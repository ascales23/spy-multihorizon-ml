"""
Walk-forward backtester for the SPY direction MLP.

For each retrain step:
  1. Train ensemble of N seeds on ALL valid history up to step t (expanding window)
  2. Predict daily direction probabilities for the next `retrain_freq` days
  3. Slide forward, repeat

Trading rule: position = 1.0 (long) if P(up @ horizon) > threshold, else 0.0.
Execution timing: prediction made at EOD t-1, position effective for day t,
realizes daily_return[t] = close[t]/close[t-1] - 1. (positions shifted by 1
so we never use today's prediction for today's return.)

Cost model: cost_bps per side, paid on |position[t] - position[t-1]| > 0.

Baselines: buy-and-hold (always long, one entry cost), always-long-flat,
vol-targeted SPY @ 15% annualized realized vol.

    python backtest.py --symbol SPY --horizon 5 --seeds 5 --retrain-freq 21
"""
from __future__ import annotations
import os

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as feat
from dataset import TabularDataset
from models.mlp import MultiHorizonMLP

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("backtest")

# Validated minimal-redundancy feature set
DROP_FEATURES = {"tenkan", "kijun", "senkou_a", "senkou_b", "days_to_month_end"}
DEFAULT_HORIZONS = [1, 5, 21]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Training (single seed) ─────────────────────────────────────────────────
def train_one_seed(X_tr, y_tr, X_val, y_val, *, seed, epochs=80, lr=1e-3, bs=128):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MultiHorizonMLP(X_tr.shape[1], y_tr.shape[1]).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TabularDataset(X_tr, y_tr),
                        batch_size=min(bs, len(X_tr)),
                        shuffle=True)
    best_auc, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            loss = loss_fn(model(X), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(torch.from_numpy(X_val).to(DEVICE))).cpu().numpy()
        try:
            v = float(np.nanmean([roc_auc_score(y_val[:, i], vp[:, i])
                                  for i in range(y_val.shape[1])]))
        except Exception:
            v = -1.0
        if v > best_auc:
            best_auc, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 12:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model


def ensemble_predict(models, X):
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(torch.sigmoid(m(torch.from_numpy(X).to(DEVICE))).cpu().numpy())
    return np.stack(preds).mean(axis=0)


# ─── Walk-forward driver ────────────────────────────────────────────────────
def run_walkforward(df: pd.DataFrame, horizon: int, *,
                    train_min: int, retrain_freq: int, n_seeds: int):
    feat_cols = [c for c in df.columns
                 if c not in ("close",)
                 and c not in DROP_FEATURES
                 and not c.startswith("label_")
                 and not c.startswith("fwd_return_")]
    label_cols = [f"label_{h}d" for h in DEFAULT_HORIZONS]
    h_idx = DEFAULT_HORIZONS.index(horizon)

    # Drop only rows missing labels (we need them for training); features handled per-fold
    df = df.sort_index().dropna(subset=label_cols + feat_cols).copy()
    n = len(df)
    logger.info("Backtest universe: %d daily rows, %d features", n, len(feat_cols))

    predictions = pd.Series(index=df.index, dtype=float)
    n_retrains = 0
    cursor = train_min
    while cursor < n:
        end = min(cursor + retrain_freq, n)
        train_df = df.iloc[:cursor]
        n_val = max(20, int(len(train_df) * 0.15))
        tr_part = train_df.iloc[:-n_val]
        val_part = train_df.iloc[-n_val:]

        sc = StandardScaler()
        X_tr = sc.fit_transform(tr_part[feat_cols].values).astype(np.float32)
        X_val = sc.transform(val_part[feat_cols].values).astype(np.float32)
        y_tr = tr_part[label_cols].values.astype(np.float32)
        y_val = val_part[label_cols].values.astype(np.float32)

        pred_block = df.iloc[cursor:end]
        X_pred = sc.transform(pred_block[feat_cols].values).astype(np.float32)

        models = [train_one_seed(X_tr, y_tr, X_val, y_val, seed=s) for s in range(n_seeds)]
        ens = ensemble_predict(models, X_pred)
        predictions.iloc[cursor:end] = ens[:, h_idx]

        n_retrains += 1
        if n_retrains % 5 == 0 or cursor == train_min:
            logger.info("  retrain %d at row %d/%d (date %s)  predicting [%d..%d)",
                        n_retrains, cursor, n,
                        str(df.index[cursor])[:10], cursor, end)
        cursor = end

    return predictions, df


# ─── Strategy returns + baselines ───────────────────────────────────────────
def build_strategies(df: pd.DataFrame, predictions: pd.Series, *,
                     threshold: float, cost_bps: float):
    """All series indexed by df.index. cost_bps is per side."""
    close = df["close"]
    daily_ret = close.pct_change()
    cost_per_unit = cost_bps / 10_000.0

    # ── NN strategy ──
    raw_signal = (predictions > threshold).astype(float)
    # Position in force during day t was decided at EOD t-1
    pos_nn = raw_signal.shift(1).fillna(0.0)
    pos_change = pos_nn.diff().abs()
    # First day: paid cost = pos_nn[0] (whichever fraction of initial position)
    pos_change.iloc[0] = pos_nn.iloc[0] if len(pos_nn) else 0.0
    cost_nn = pos_change * cost_per_unit
    ret_nn = pos_nn * daily_ret - cost_nn

    # ── Buy & hold ──
    pos_bh = pd.Series(1.0, index=df.index)
    cost_bh = pd.Series(0.0, index=df.index)
    cost_bh.iloc[0] = cost_per_unit   # one entry cost
    ret_bh = pos_bh * daily_ret - cost_bh

    # ── Vol-targeted SPY at 15% ann vol ──
    realized_vol = daily_ret.rolling(20).std() * np.sqrt(252)
    target_vol = 0.15
    pos_vt = (target_vol / realized_vol.clip(lower=0.05)).clip(upper=2.0).shift(1).fillna(0.0)
    pos_vt_change = pos_vt.diff().abs().fillna(pos_vt.iloc[0])
    cost_vt = pos_vt_change * cost_per_unit
    ret_vt = pos_vt * daily_ret - cost_vt

    return {
        "close": close,
        "daily_ret": daily_ret,
        "nn": {"position": pos_nn, "return": ret_nn, "cost": cost_nn},
        "bh": {"position": pos_bh, "return": ret_bh, "cost": cost_bh},
        "vt": {"position": pos_vt, "return": ret_vt, "cost": cost_vt},
    }


# ─── Metrics ────────────────────────────────────────────────────────────────
def metrics_of(returns: pd.Series, name: str = "") -> dict:
    r = returns.dropna()
    if len(r) == 0:
        return {"name": name, "n_days": 0}
    cum = (1 + r).cumprod()
    total = float(cum.iloc[-1] - 1)
    n_days = len(r)
    ann_ret = (1 + total) ** (252.0 / n_days) - 1 if total > -1 else -1
    ann_vol = float(r.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    downside = float(r[r < 0].std() * np.sqrt(252)) if (r < 0).any() else 0.0
    sortino = ann_ret / downside if downside > 0 else 0.0
    rolling_max = cum.cummax()
    dd_series = (cum - rolling_max) / rolling_max
    max_dd = float(dd_series.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    hit_rate = float((r > 0).mean())
    pos_sum = float(r[r > 0].sum())
    neg_sum = float(-r[r < 0].sum())
    profit_factor = pos_sum / neg_sum if neg_sum > 0 else float("inf")

    return {
        "name":             name,
        "total_return_pct": round(total * 100, 2),
        "ann_return_pct":   round(ann_ret * 100, 2),
        "ann_vol_pct":      round(ann_vol * 100, 2),
        "sharpe":           round(sharpe, 3),
        "sortino":          round(sortino, 3),
        "max_dd_pct":       round(max_dd * 100, 2),
        "calmar":           round(calmar, 3),
        "hit_rate_pct":     round(hit_rate * 100, 2),
        "profit_factor":    round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "n_days":           n_days,
    }


def trade_stats(position: pd.Series, returns: pd.Series, close: pd.Series) -> dict:
    pos = position.fillna(0.0)
    changes = pos.diff().fillna(pos.iloc[0])
    trades = []
    in_trade = False
    entry_idx, entry_close = None, None
    for i, idx in enumerate(pos.index):
        ch = float(changes.iloc[i])
        if not in_trade and ch > 0.5:    # treating binary 0/1 transitions
            in_trade = True
            entry_idx = idx
            entry_close = float(close.iloc[i])
        elif in_trade and (ch < -0.5 or i == len(pos) - 1):
            exit_close = float(close.iloc[i])
            duration = (idx - entry_idx).days if entry_idx else 0
            t_ret = exit_close / entry_close - 1 if entry_close else 0.0
            trades.append({"entry": str(entry_idx)[:10], "exit": str(idx)[:10],
                           "days": duration, "return_pct": round(t_ret * 100, 3)})
            in_trade = False
    if not trades:
        return {"n_trades": 0}
    wins = [t for t in trades if t["return_pct"] > 0]
    losses = [t for t in trades if t["return_pct"] <= 0]
    return {
        "n_trades":         len(trades),
        "avg_trade_days":   round(mean(t["days"] for t in trades), 1),
        "win_rate_pct":     round(len(wins) / len(trades) * 100, 2),
        "avg_win_pct":      round(mean(t["return_pct"] for t in wins), 3) if wins else 0.0,
        "avg_loss_pct":     round(mean(t["return_pct"] for t in losses), 3) if losses else 0.0,
        "best_trade_pct":   round(max(t["return_pct"] for t in trades), 3),
        "worst_trade_pct":  round(min(t["return_pct"] for t in trades), 3),
    }


# ─── Plot ───────────────────────────────────────────────────────────────────
def plot_results(out_dir: Path, all_returns: dict, horizon: int):
    """3-panel: equity (log), drawdown, rolling 1y Sharpe."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    colors = {"NN": "#1f77b4", "Buy&Hold": "#888888", "VolTarget15": "#2ca02c"}

    ax = axes[0]
    for name, r in all_returns.items():
        cum = (1 + r).cumprod()
        ax.plot(cum.index, cum.values, label=name, linewidth=1.7,
                color=colors.get(name, "k"))
    ax.set_yscale("log")
    ax.set_ylabel("Equity (log)")
    ax.set_title(f"SPY {horizon}-day MLP backtest, equity curves (walk-forward, ensembled)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    for name, r in all_returns.items():
        cum = (1 + r).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax() * 100
        ax.plot(dd.index, dd.values, label=name, linewidth=1.5,
                color=colors.get(name, "k"))
    ax.set_ylabel("Drawdown %")
    ax.set_title("Drawdowns")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for name, r in all_returns.items():
        rs = r.rolling(252).mean() / r.rolling(252).std() * np.sqrt(252)
        ax.plot(rs.index, rs.values, label=name, linewidth=1.5,
                color=colors.get(name, "k"))
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.set_ylabel("Rolling 1y Sharpe")
    ax.set_title("Rolling 1-year Sharpe")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Date")

    plt.tight_layout()
    out = out_dir / "equity_curves.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    return out


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",        default="SPY")
    ap.add_argument("--horizon",       type=int,   default=5)
    ap.add_argument("--train-min",     type=int,   default=250)
    ap.add_argument("--retrain-freq",  type=int,   default=21,
                    help="Days between retrains. 21 = monthly.")
    ap.add_argument("--seeds",         type=int,   default=5)
    ap.add_argument("--threshold",     type=float, default=0.5)
    ap.add_argument("--cost-bps",      type=float, default=1.0,
                    help="Per side. Round-trip = 2x this. 1.0 ≈ retail SPY.")
    ap.add_argument("--out-root",      default=str(Path(__file__).resolve().parent / "runs"))
    args = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"backtest_{args.symbol}_{args.horizon}d_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output → %s", out_dir)

    # Build features
    logger.info("Loading features for %s ...", args.symbol)
    df = feat.build(args.symbol)
    df = feat.add_labels(df, horizons=DEFAULT_HORIZONS)
    logger.info("Feature matrix raw: %s", df.shape)

    # Walk-forward predict
    logger.info("Walk-forward (horizon=%dd, retrain_freq=%d, seeds=%d, device=%s)...",
                args.horizon, args.retrain_freq, args.seeds, DEVICE)
    predictions, df_used = run_walkforward(
        df, args.horizon,
        train_min=args.train_min,
        retrain_freq=args.retrain_freq,
        n_seeds=args.seeds,
    )

    # Build strategies
    strat = build_strategies(df_used, predictions,
                             threshold=args.threshold,
                             cost_bps=args.cost_bps)

    # Restrict everything to the prediction window so comparison is apples-to-apples
    first = predictions.first_valid_index()
    last = predictions.last_valid_index()
    pred_slice = slice(first, last)

    returns = {
        "NN":          strat["nn"]["return"].loc[pred_slice],
        "Buy&Hold":    strat["bh"]["return"].loc[pred_slice],
        "VolTarget15": strat["vt"]["return"].loc[pred_slice],
    }
    # Adjust BH so its FIRST cost is paid on the first day of the prediction period
    bh_adj = strat["daily_ret"].loc[pred_slice].copy()
    bh_adj.iloc[0] -= args.cost_bps / 10_000.0
    returns["Buy&Hold"] = bh_adj

    # Compute metrics
    summary = {n: metrics_of(r, n) for n, r in returns.items()}
    summary["NN"]["trade_stats"] = trade_stats(
        strat["nn"]["position"].loc[pred_slice],
        strat["nn"]["return"].loc[pred_slice],
        strat["close"].loc[pred_slice],
    )
    summary["NN"]["avg_exposure"] = round(float(strat["nn"]["position"].loc[pred_slice].mean()), 3)

    # ── Print pretty table ─────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"BACKTEST RESULTS, SPY {args.horizon}-day horizon, walk-forward, "
          f"{args.seeds}-seed ensemble, {args.cost_bps}bp/side cost")
    print(f"Period: {str(first)[:10]} → {str(last)[:10]}  ({summary['NN']['n_days']} trading days)")
    print("=" * 92)
    header = f"{'Metric':<22} {'NN':>14} {'Buy&Hold':>14} {'VolTarget15':>14}"
    print("\n" + header)
    print("-" * len(header))
    for k in ["total_return_pct", "ann_return_pct", "ann_vol_pct", "sharpe",
              "sortino", "max_dd_pct", "calmar", "hit_rate_pct", "profit_factor"]:
        nn = summary["NN"][k]; bh = summary["Buy&Hold"][k]; vt = summary["VolTarget15"][k]
        print(f"{k:<22} {str(nn):>14} {str(bh):>14} {str(vt):>14}")
    print(f"\nNN avg exposure: {summary['NN']['avg_exposure']}")
    print(f"NN trade stats:  {summary['NN']['trade_stats']}")

    # Save artifacts
    daily = pd.DataFrame({
        "close":            strat["close"].loc[pred_slice],
        "daily_return":     strat["daily_ret"].loc[pred_slice],
        "prediction":       predictions.loc[pred_slice],
        "nn_position":      strat["nn"]["position"].loc[pred_slice],
        "nn_cost":          strat["nn"]["cost"].loc[pred_slice],
        "nn_return":        strat["nn"]["return"].loc[pred_slice],
        "bh_return":        returns["Buy&Hold"],
        "vt_return":        strat["vt"]["return"].loc[pred_slice],
    })
    daily.to_csv(out_dir / "daily.csv")

    summary_full = {
        "config":           vars(args),
        "period":           {"start": str(first)[:10], "end": str(last)[:10]},
        "metrics":          summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_full, indent=2, default=str))

    plot_path = plot_results(out_dir, returns, args.horizon)
    print(f"\nArtifacts:")
    print(f"  summary.json  , full metrics + config")
    print(f"  daily.csv     , every day's predictions/positions/returns")
    print(f"  {plot_path.name}, equity, drawdown, rolling Sharpe (3-panel)")
    print(f"  Path: {out_dir}")


if __name__ == "__main__":
    main()
