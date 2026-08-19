"""
Walk-forward cross-validation for the SPY direction classifier.

Expanding window: train [0:end_k], val [end_k-val_size:end_k],
                  test [end_k:end_k+test_size]. Slide by `step` days per fold.

Per-fold metrics for XGBoost and MLP. Aggregate mean ± std across folds for
each horizon. Gives a much more honest read than the single 70/15/15 holdout -
tells us whether a "good" model is consistently good or just lucky on one window.

    python walk_forward.py --symbol SPY --horizons 1,5,21 --train-min 180 --test 30 --step 21
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import features as feat
from dataset import TabularDataset
from models.mlp import MultiHorizonMLP

logger = logging.getLogger("walk_forward")


# ── per-fold splitting & metrics ────────────────────────────────────────────
def fold_split(df: pd.DataFrame, train_idx, test_idx, horizons, val_frac=0.15):
    """Build per-fold scaled arrays. Val is the tail of train (for MLP early stop)."""
    label_cols = [f"label_{h}d" for h in horizons]
    feat_cols = [c for c in df.columns
                 if c not in ("close",)
                 and not c.startswith("label_")
                 and not c.startswith("fwd_return_")]
    fwd_cols = [f"fwd_return_{h}d" for h in horizons]

    train_df = df.loc[train_idx].dropna(subset=feat_cols + label_cols)
    test_df  = df.loc[test_idx].dropna(subset=feat_cols + label_cols)
    if len(train_df) < 50 or len(test_df) < 5:
        return None

    n_val = max(20, int(len(train_df) * val_frac))
    val_df = train_df.iloc[-n_val:]
    tr_df  = train_df.iloc[:-n_val]

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(tr_df[feat_cols].values).astype(np.float32)
    X_val  = scaler.transform(val_df[feat_cols].values).astype(np.float32)
    X_test = scaler.transform(test_df[feat_cols].values).astype(np.float32)

    return {
        "X_tr":   X_tr,
        "y_tr":   tr_df[label_cols].values.astype(np.float32),
        "X_val":  X_val,
        "y_val":  val_df[label_cols].values.astype(np.float32),
        "X_test": X_test,
        "y_test": test_df[label_cols].values.astype(np.float32),
        "fwd_test": test_df[fwd_cols].values,
        "feat_cols": feat_cols,
        "n_train": len(tr_df),
        "n_val":   len(val_df),
        "n_test":  len(test_df),
    }


def directional_sharpe(probs: np.ndarray, fwd: np.ndarray) -> float:
    pos = (probs > 0.5).astype(float)
    pnl = pos * fwd
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) == 0 or pnl.std() == 0:
        return 0.0
    return float(math.sqrt(252) * pnl.mean() / pnl.std())


def per_horizon_metrics(probs, y, fwd, horizons):
    out = {}
    for i, h in enumerate(horizons):
        try:
            auc = roc_auc_score(y[:, i], probs[:, i])
        except ValueError:
            auc = float("nan")     # single-class test set
        out[f"{h}d"] = {
            "auc":    float(auc),
            "acc":    float(accuracy_score(y[:, i], probs[:, i] > 0.5)),
            "sharpe": directional_sharpe(probs[:, i], fwd[:, i]),
            "base":   float(y[:, i].mean()),
        }
    return out


# ── per-fold trainers ───────────────────────────────────────────────────────
def fit_xgboost(fold, horizons):
    import xgboost as xgb
    test_probs = np.zeros_like(fold["y_test"])
    for i, h in enumerate(horizons):
        model = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="auc", tree_method="hist", n_jobs=-1,
            early_stopping_rounds=30,
        )
        model.fit(
            fold["X_tr"], fold["y_tr"][:, i],
            eval_set=[(fold["X_val"], fold["y_val"][:, i])],
            verbose=False,
        )
        test_probs[:, i] = model.predict_proba(fold["X_test"])[:, 1]
    return per_horizon_metrics(test_probs, fold["y_test"], fold["fwd_test"], horizons)


def fit_mlp(fold, horizons, *, epochs=80, lr=1e-3, batch_size=128, device="cuda"):
    model = MultiHorizonMLP(
        n_features=fold["X_tr"].shape[1],
        n_horizons=len(horizons),
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr_loader = DataLoader(TabularDataset(fold["X_tr"], fold["y_tr"]),
                           batch_size=min(batch_size, len(fold["X_tr"])),
                           shuffle=True, drop_last=False)

    best_auc, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            loss = loss_fn(model(X), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_p = torch.sigmoid(
                model(torch.from_numpy(fold["X_val"]).to(device))
            ).cpu().numpy()
        try:
            v = float(np.nanmean([roc_auc_score(fold["y_val"][:, i], val_p[:, i])
                                  for i in range(len(horizons))]))
        except Exception:
            v = -1.0
        if v > best_auc:
            best_auc, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 12:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_p = torch.sigmoid(
            model(torch.from_numpy(fold["X_test"]).to(device))
        ).cpu().numpy()
    return per_horizon_metrics(test_p, fold["y_test"], fold["fwd_test"], horizons)


# ── walk-forward driver ────────────────────────────────────────────────────
def walk_forward(
    df: pd.DataFrame,
    horizons: list[int],
    *,
    train_min: int,
    test_size: int,
    step: int,
    device: str,
):
    df = df.sort_index()
    label_cols = [f"label_{h}d" for h in horizons]
    df = df.dropna(subset=label_cols)
    n = len(df)
    if train_min + test_size > n:
        raise ValueError(f"Need {train_min + test_size} rows, have {n}. "
                         f"Reduce --train-min or --test.")

    fold_xgb: list[dict] = []
    fold_mlp: list[dict] = []
    fold_meta: list[dict] = []

    k = 0
    while True:
        train_end = train_min + k * step
        test_end = train_end + test_size
        if test_end > n:
            break
        train_idx = df.index[:train_end]
        test_idx  = df.index[train_end:test_end]
        fold = fold_split(df, train_idx, test_idx, horizons)
        if fold is None:
            logger.warning("Fold %d skipped (too small after dropna)", k)
            k += 1
            continue

        logger.info("Fold %d  train=%d  val=%d  test=%d  test_range=%s..%s",
                    k, fold["n_train"], fold["n_val"], fold["n_test"],
                    str(test_idx[0])[:10], str(test_idx[-1])[:10])

        try:
            xgb_m = fit_xgboost(fold, horizons)
        except Exception as e:
            logger.exception("XGB failed fold %d: %s", k, e)
            xgb_m = None
        try:
            mlp_m = fit_mlp(fold, horizons, device=device)
        except Exception as e:
            logger.exception("MLP failed fold %d: %s", k, e)
            mlp_m = None

        fold_xgb.append(xgb_m or {})
        fold_mlp.append(mlp_m or {})
        fold_meta.append({
            "fold": k,
            "test_start": str(test_idx[0])[:10],
            "test_end":   str(test_idx[-1])[:10],
            "n_train":    fold["n_train"],
            "n_test":     fold["n_test"],
        })
        k += 1

    return aggregate(fold_xgb, fold_mlp, fold_meta, horizons)


def aggregate(fold_xgb, fold_mlp, fold_meta, horizons):
    """Compute per-horizon mean / std across folds, ignoring NaN-AUC folds."""
    def agg(folds, metric):
        out = {}
        for h in horizons:
            vals = [f.get(f"{h}d", {}).get(metric, float("nan")) for f in folds]
            vals = [v for v in vals if v == v]  # not NaN
            if not vals:
                out[f"{h}d"] = {"mean": None, "std": None, "n_folds": 0}
                continue
            out[f"{h}d"] = {
                "mean":     round(mean(vals), 4),
                "std":      round(stdev(vals), 4) if len(vals) > 1 else 0.0,
                "n_folds":  len(vals),
                "per_fold": [round(v, 4) for v in vals],
            }
        return out

    return {
        "n_folds": len(fold_meta),
        "folds":   fold_meta,
        "xgboost": {
            "auc":    agg(fold_xgb, "auc"),
            "acc":    agg(fold_xgb, "acc"),
            "sharpe": agg(fold_xgb, "sharpe"),
        },
        "mlp": {
            "auc":    agg(fold_mlp, "auc"),
            "acc":    agg(fold_mlp, "acc"),
            "sharpe": agg(fold_mlp, "sharpe"),
        },
    }


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--horizons", default="1,5,21")
    ap.add_argument("--train-min", type=int, default=180,
                    help="Min training rows for fold 0 (expanding window)")
    ap.add_argument("--test", type=int, default=30, help="Test rows per fold")
    ap.add_argument("--step", type=int, default=21, help="Days between fold starts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "runs"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    horizons = [int(h) for h in args.horizons.split(",")]

    df = feat.build(args.symbol)
    df = feat.add_labels(df, horizons=horizons)
    logger.info("Loaded %s: %s", args.symbol, df.shape)

    result = walk_forward(
        df, horizons,
        train_min=args.train_min,
        test_size=args.test,
        step=args.step,
        device=args.device,
    )

    print("\n=== Walk-forward CV summary ===")
    print(f"Folds: {result['n_folds']}")
    for model in ("xgboost", "mlp"):
        print(f"\n--- {model.upper()} ---")
        for h in horizons:
            a = result[model]["auc"].get(f"{h}d", {})
            s = result[model]["sharpe"].get(f"{h}d", {})
            if a.get("mean") is None:
                print(f"  {h:>3}d  AUC: -- (all folds NaN)")
                continue
            print(f"  {h:>3}d  AUC {a['mean']:.3f} ± {a['std']:.3f}  "
                  f"Sharpe {s['mean']:.2f} ± {s['std']:.2f}  "
                  f"(n={a['n_folds']})")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    path = out / f"walk_forward_{args.symbol}_{ts}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
