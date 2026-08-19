"""
Walk-forward backtester for the intraday MLP (1-day-ahead direction).

Predictions made at every 15-min bar (model trained on multi-horizon labels:
15min, 1h, 1d). The TRADING DECISION is made once per day at the last bar
of the regular session (15:45 ET timestamp) using the 26-bar (1-day-ahead)
prediction. Position is taken at the next bar's open and held until the
following day's decision. This keeps trading frequency ≤ 1 round-trip per
day, so cost impact stays manageable.

Walk-forward:
  - Retrain every `retrain_bars` (default 130 = ~5 trading days)
  - Expanding window, each retrain uses ALL history up to that point
  - 3-seed ensemble per retrain (matches daily backtest pattern)

Baselines: buy-and-hold SPY (intraday compounding), vol-targeted SPY at 15%
annualized realized vol.

    python backtest_intraday.py --symbol SPY --retrain-bars 130 --seeds 3
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features_intraday as fi
from dataset import TabularDataset
from models.mlp import MultiHorizonMLP

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("backtest_intraday")

HORIZONS_BARS = [1, 4, 26]   # 15min, 1h, 1d
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BARS_PER_YEAR = 252 * 26    # ~6,552, for annualizing intraday returns
BARS_PER_DAY = 26


# ─── Train one seed ───────────────────────────────────────────────────────
def train_one_seed(X_tr, y_tr, X_val, y_val, *, seed, epochs=50, lr=1e-3, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MultiHorizonMLP(X_tr.shape[1], y_tr.shape[1]).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TabularDataset(X_tr, y_tr),
                        batch_size=min(bs, len(X_tr)), shuffle=True)
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
            v = float(np.nanmean([roc_auc_score(y_val[:,i], vp[:,i])
                                   for i in range(y_val.shape[1])]))
        except Exception:
            v = -1.0
        if v > best_auc:
            best_auc, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 8:
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


# ─── Walk-forward predictions ─────────────────────────────────────────────
def walk_forward_predict(df: pd.DataFrame, *,
                          train_min: int, retrain_bars: int, n_seeds: int):
    """Generate predictions for every bar after train_min via expanding-window CV."""
    label_cols = [f"label_{h}b" for h in HORIZONS_BARS]
    feat_cols = [c for c in df.columns
                 if c not in ("close",)
                 and not c.startswith("label_")
                 and not c.startswith("fwd_return_")
                 and df[c].isna().sum() / len(df) < 0.4]
    dropped = [c for c in df.columns
               if c not in ("close",)
               and not c.startswith("label_")
               and not c.startswith("fwd_return_")
               and df[c].isna().sum() / len(df) >= 0.4]
    logger.info("Using %d features (dropped %d >40%% NaN: %s)",
                len(feat_cols), len(dropped), dropped[:6])

    df = df.sort_index().dropna(subset=label_cols + feat_cols).copy()
    n = len(df)
    logger.info("Backtest universe: %d bars (range %s → %s)",
                n, str(df.index[0])[:16], str(df.index[-1])[:16])

    # We'll predict 26b (1-day) for the trading decision
    h_idx = HORIZONS_BARS.index(26)

    predictions = pd.Series(index=df.index, dtype=float)
    n_retrains = 0
    cursor = train_min
    while cursor < n:
        end = min(cursor + retrain_bars, n)
        train_df = df.iloc[:cursor]
        n_val = max(100, int(len(train_df) * 0.15))
        tr_part = train_df.iloc[:-n_val]
        val_part = train_df.iloc[-n_val:]

        sc = StandardScaler()
        X_tr = sc.fit_transform(tr_part[feat_cols].values).astype(np.float32)
        X_val = sc.transform(val_part[feat_cols].values).astype(np.float32)
        y_tr = tr_part[label_cols].values.astype(np.float32)
        y_val = val_part[label_cols].values.astype(np.float32)

        pred_block = df.iloc[cursor:end]
        X_pred = sc.transform(pred_block[feat_cols].values).astype(np.float32)

        models = [train_one_seed(X_tr, y_tr, X_val, y_val, seed=s)
                  for s in range(n_seeds)]
        ens = ensemble_predict(models, X_pred)
        predictions.iloc[cursor:end] = ens[:, h_idx]

        n_retrains += 1
        if n_retrains % 3 == 0 or cursor == train_min:
            logger.info("  retrain %d  train=%d  predicting [%d..%d)",
                        n_retrains, cursor, cursor, end)
        cursor = end

    return predictions, df


# ─── Trading rule: once-per-day decision at 15:45 ET last bar ─────────────
def build_strategies(df: pd.DataFrame, predictions: pd.Series, *,
                      threshold: float, cost_bps: float,
                      decision_hour: int = 15, decision_minute: int = 45):
    close = df["close"]
    bar_returns = close.pct_change()
    cost_per_unit = cost_bps / 10_000.0

    # Identify decision bars (15:45 ET = 19:45 or 20:45 UTC depending on DST)
    idx_et = predictions.index.tz_convert("America/New_York")
    is_decision = (idx_et.hour == decision_hour) & (idx_et.minute == decision_minute)

    raw_signal = pd.Series(np.nan, index=predictions.index)
    valid_decisions = predictions[is_decision & predictions.notna()]
    raw_signal.loc[valid_decisions.index] = (valid_decisions > threshold).astype(float)
    # ffill the binary signal between decisions; shift 1 bar so position takes
    # effect AFTER the decision bar closes (we trade at next bar's open).
    filled = raw_signal.ffill().fillna(0.0)
    position = filled.shift(1).fillna(0.0)

    # Cost on position changes (typically once per day)
    pos_change = position.diff().abs()
    pos_change.iloc[0] = position.iloc[0] if len(position) else 0.0
    cost_nn = pos_change * cost_per_unit
    ret_nn = position * bar_returns - cost_nn

    # Buy & hold (intraday compounding)
    pos_bh = pd.Series(1.0, index=df.index)
    cost_bh = pd.Series(0.0, index=df.index)
    cost_bh.iloc[0] = cost_per_unit
    ret_bh = pos_bh * bar_returns - cost_bh

    # Vol-targeted SPY @ 15%, bar-level, 5-day rolling annualized vol
    rolling_window = BARS_PER_DAY * 5
    realized_ann_vol = bar_returns.rolling(rolling_window).std() * np.sqrt(BARS_PER_YEAR)
    pos_vt = (0.15 / realized_ann_vol.clip(lower=0.05)).clip(upper=2.0).shift(1).fillna(0.0)
    pos_vt_change = pos_vt.diff().abs().fillna(pos_vt.iloc[0])
    cost_vt = pos_vt_change * cost_per_unit
    ret_vt = pos_vt * bar_returns - cost_vt

    return {
        "close": close,
        "bar_returns": bar_returns,
        "nn":   {"position": position, "cost": cost_nn, "return": ret_nn,
                 "n_decisions": int(is_decision.sum())},
        "bh":   {"position": pos_bh,   "cost": cost_bh, "return": ret_bh},
        "vt":   {"position": pos_vt,   "cost": cost_vt, "return": ret_vt},
    }


# ─── Metrics (intraday-aware annualization) ────────────────────────────────
def metrics_of(returns: pd.Series, name: str = "") -> dict:
    r = returns.dropna()
    if len(r) == 0:
        return {"name": name, "n_bars": 0}
    cum = (1 + r).cumprod()
    total = float(cum.iloc[-1] - 1)
    n_bars = len(r)
    # Annualize using bars-per-year
    ann_ret = (1 + total) ** (BARS_PER_YEAR / n_bars) - 1 if total > -1 else -1
    ann_vol = float(r.std() * np.sqrt(BARS_PER_YEAR))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    downside = float(r[r < 0].std() * np.sqrt(BARS_PER_YEAR)) if (r < 0).any() else 0.0
    sortino = ann_ret / downside if downside > 0 else 0.0
    cummax = cum.cummax()
    dd = (cum - cummax) / cummax
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
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
        "profit_factor":    round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "n_bars":           n_bars,
    }


# ─── Plot ───────────────────────────────────────────────────────────────────
def plot_results(out_dir: Path, all_returns: dict):
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    colors = {"NN": "#1f77b4", "Buy&Hold": "#888888", "VolTarget15": "#2ca02c"}

    ax = axes[0]
    for name, r in all_returns.items():
        cum = (1 + r).cumprod()
        ax.plot(cum.index, cum.values, label=name, linewidth=1.5,
                color=colors.get(name, "k"))
    ax.set_yscale("log")
    ax.set_ylabel("Equity (log)")
    ax.set_title("SPY intraday MLP backtest, 1-day signal, once-per-day trading")
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
    ax.legend(loc="lower left"); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for name, r in all_returns.items():
        # Rolling 1-month Sharpe (1 month ~ 545 bars at 15-min)
        rs = r.rolling(545).mean() / r.rolling(545).std() * np.sqrt(BARS_PER_YEAR)
        ax.plot(rs.index, rs.values, label=name, linewidth=1.3,
                color=colors.get(name, "k"))
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.set_ylabel("Rolling 1-month Sharpe")
    ax.set_title("Rolling 1-month Sharpe (intraday-annualized)")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    ax.set_xlabel("Bar timestamp (UTC)")

    plt.tight_layout()
    p = out_dir / "equity_curves_intraday.png"
    plt.savefig(p, dpi=110, bbox_inches="tight")
    plt.close()
    return p


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--train-min", type=int, default=2000,
                    help="Min training bars for first retrain. 2000 ≈ 75 days.")
    ap.add_argument("--retrain-bars", type=int, default=130,
                    help="Bars between retrains. 130 ≈ 5 days.")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parent / "runs"))
    args = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"intraday_backtest_{args.symbol}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output → %s", out_dir)

    logger.info("Loading intraday features for %s...", args.symbol)
    df = fi.build(args.symbol, timeframe="15mins")
    df = fi.add_labels(df, horizons_bars=HORIZONS_BARS)
    logger.info("Feature matrix raw: %s (NaN tail will be trimmed)", df.shape)

    logger.info("Walk-forward (train_min=%d, retrain_bars=%d, seeds=%d)...",
                args.train_min, args.retrain_bars, args.seeds)
    predictions, df_used = walk_forward_predict(
        df,
        train_min=args.train_min,
        retrain_bars=args.retrain_bars,
        n_seeds=args.seeds,
    )

    strat = build_strategies(df_used, predictions,
                              threshold=args.threshold,
                              cost_bps=args.cost_bps)

    first = predictions.first_valid_index()
    last = predictions.last_valid_index()
    pred_slice = slice(first, last)

    returns = {
        "NN":          strat["nn"]["return"].loc[pred_slice],
        "Buy&Hold":    strat["bar_returns"].loc[pred_slice].copy(),
        "VolTarget15": strat["vt"]["return"].loc[pred_slice],
    }
    returns["Buy&Hold"].iloc[0] -= args.cost_bps / 10_000.0

    summary = {n: metrics_of(r, n) for n, r in returns.items()}
    summary["NN"]["n_decisions_made"] = strat["nn"]["n_decisions"]
    summary["NN"]["avg_exposure"] = round(float(strat["nn"]["position"].loc[pred_slice].mean()), 3)

    print("\n" + "=" * 92)
    print(f"INTRADAY BACKTEST, SPY 1-day-ahead signal, ONCE-PER-DAY trading, "
          f"{args.seeds}-seed ensemble, {args.cost_bps}bp/side cost")
    print(f"Period: {str(first)[:16]} → {str(last)[:16]}  "
          f"({summary['NN']['n_bars']:,} bars ≈ {summary['NN']['n_bars']//BARS_PER_DAY} trading days)")
    print(f"Decisions made: {summary['NN']['n_decisions_made']}")
    print("=" * 92)
    header = f"{'Metric':<22} {'NN':>14} {'Buy&Hold':>14} {'VolTarget15':>14}"
    print("\n" + header); print("-" * len(header))
    for k in ["total_return_pct", "ann_return_pct", "ann_vol_pct", "sharpe",
              "sortino", "max_dd_pct", "calmar", "profit_factor"]:
        nn = summary["NN"][k]; bh = summary["Buy&Hold"][k]; vt = summary["VolTarget15"][k]
        print(f"{k:<22} {str(nn):>14} {str(bh):>14} {str(vt):>14}")
    print(f"\nNN avg exposure: {summary['NN']['avg_exposure']}")

    # Save daily-aggregated CSV for sanity (one row per day)
    daily = pd.DataFrame({
        "close":         strat["close"].loc[pred_slice],
        "bar_return":    strat["bar_returns"].loc[pred_slice],
        "prediction":    predictions.loc[pred_slice],
        "nn_position":   strat["nn"]["position"].loc[pred_slice],
        "nn_return":     strat["nn"]["return"].loc[pred_slice],
        "bh_return":     returns["Buy&Hold"],
        "vt_return":     strat["vt"]["return"].loc[pred_slice],
    })
    daily.to_csv(out_dir / "bars.csv")

    (out_dir / "summary.json").write_text(json.dumps({
        "config":  vars(args),
        "period":  {"start": str(first)[:16], "end": str(last)[:16]},
        "metrics": summary,
    }, indent=2, default=str))

    plot_path = plot_results(out_dir, returns)
    print(f"\nArtifacts:")
    print(f"  summary.json")
    print(f"  bars.csv")
    print(f"  {plot_path.name}")
    print(f"  Path: {out_dir}")


if __name__ == "__main__":
    main()
