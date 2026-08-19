"""
Train both an XGBoost baseline and the multi-horizon MLP on the same
features+splits. Compares them on per-horizon AUC and a simple long/flat
directional Sharpe.

    python train.py --symbol SPY --horizons 1,5,21 --epochs 80
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader

import features as feat
from dataset import make_splits, TabularDataset
from models.mlp import MultiHorizonMLP

logger = logging.getLogger("train")


# ── Metrics ──────────────────────────────────────────────────────────────────
def directional_sharpe(probs: np.ndarray, fwd_returns: np.ndarray) -> float:
    """
    Long when prob > 0.5, flat otherwise. Annualized Sharpe over the
    realized forward returns. (No tx-cost / no scaling.)
    """
    pos = (probs > 0.5).astype(float)
    pnl = pos * fwd_returns
    pnl = pnl[~np.isnan(pnl)]
    if pnl.std() == 0:
        return 0.0
    return float(np.sqrt(252) * pnl.mean() / pnl.std())


def eval_per_horizon(probs: np.ndarray, y: np.ndarray, fwd: np.ndarray,
                     horizons: list[int]) -> dict:
    out = {}
    for i, h in enumerate(horizons):
        out[f"{h}d"] = {
            "auc": float(roc_auc_score(y[:, i], probs[:, i])),
            "acc": float(accuracy_score(y[:, i], probs[:, i] > 0.5)),
            "sharpe_long_flat": directional_sharpe(probs[:, i], fwd[:, i]),
            "base_rate_pos": float(y[:, i].mean()),
        }
    return out


# ── XGBoost baseline ─────────────────────────────────────────────────────────
def train_xgboost(splits, horizons: list[int]) -> dict:
    import xgboost as xgb
    metrics = {}
    val_probs = np.zeros_like(splits.y_val)
    test_probs = np.zeros_like(splits.y_test)
    for i, h in enumerate(horizons):
        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            early_stopping_rounds=30,
        )
        model.fit(
            splits.X_train, splits.y_train[:, i],
            eval_set=[(splits.X_val, splits.y_val[:, i])],
            verbose=False,
        )
        val_probs[:, i]  = model.predict_proba(splits.X_val)[:, 1]
        test_probs[:, i] = model.predict_proba(splits.X_test)[:, 1]
        logger.info("XGB %dd best_iter=%s", h, model.best_iteration)
    metrics["val"]  = eval_per_horizon(val_probs,  splits.y_val,  splits._fwd_val,  horizons)
    metrics["test"] = eval_per_horizon(test_probs, splits.y_test, splits._fwd_test, horizons)
    return metrics


# ── MLP trainer ──────────────────────────────────────────────────────────────
def train_mlp(splits, horizons: list[int], *, epochs: int, lr: float,
              batch_size: int, device: str) -> dict:
    model = MultiHorizonMLP(
        n_features=splits.X_train.shape[1],
        n_horizons=len(horizons),
    ).to(device)
    logger.info("MLP params: %d", model.num_parameters)

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_loader = DataLoader(TabularDataset(splits.X_train, splits.y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TabularDataset(splits.X_val, splits.y_val),
                              batch_size=512)
    test_loader  = DataLoader(TabularDataset(splits.X_test, splits.y_test),
                              batch_size=512)

    best_val_auc = -1.0
    best_state = None
    patience = 12
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(X)
        train_loss /= len(train_loader.dataset)
        sched.step()

        # Val
        model.eval()
        val_probs_all = []
        with torch.no_grad():
            for X, _ in val_loader:
                val_probs_all.append(torch.sigmoid(model(X.to(device))).cpu().numpy())
        val_probs = np.concatenate(val_probs_all, axis=0)
        val_auc = np.mean([roc_auc_score(splits.y_val[:, i], val_probs[:, i])
                           for i in range(len(horizons))])

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if ep % 5 == 0 or ep == 1:
            logger.info("ep %3d  loss %.4f  val_auc %.4f  (best %.4f)",
                        ep, train_loss, val_auc, best_val_auc)
        if bad_epochs >= patience:
            logger.info("Early stop at ep %d", ep)
            break

    # Restore best, evaluate test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs  = torch.sigmoid(model(torch.from_numpy(splits.X_val).to(device))).cpu().numpy()
        test_probs = torch.sigmoid(model(torch.from_numpy(splits.X_test).to(device))).cpu().numpy()
    return {
        "val":  eval_per_horizon(val_probs,  splits.y_val,  splits._fwd_val,  horizons),
        "test": eval_per_horizon(test_probs, splits.y_test, splits._fwd_test, horizons),
        "n_params": model.num_parameters,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--horizons", default="1,5,21")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "runs"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    horizons = [int(h) for h in args.horizons.split(",")]
    logger.info("Building features for %s, horizons=%s", args.symbol, horizons)
    f = feat.build(args.symbol)
    f = feat.add_labels(f, horizons=horizons)
    logger.info("Feature matrix: %s (NaN warmup will be trimmed)", f.shape)

    splits = make_splits(f, horizons=horizons)
    # Attach forward returns for Sharpe calc on each split
    fwd_cols = [f"fwd_return_{h}d" for h in horizons]
    f_with_idx = f.loc[splits.train_idx.union(splits.val_idx).union(splits.test_idx)]
    splits._fwd_val  = f.loc[splits.val_idx,  fwd_cols].values
    splits._fwd_test = f.loc[splits.test_idx, fwd_cols].values

    logger.info("Train %d | Val %d | Test %d | features=%d",
                len(splits.train_idx), len(splits.val_idx), len(splits.test_idx),
                len(splits.feature_cols))

    logger.info("=== XGBoost baseline ===")
    xgb_metrics = train_xgboost(splits, horizons)
    logger.info("XGBoost: %s", json.dumps(xgb_metrics, indent=2))

    logger.info("=== MLP (device=%s) ===", args.device)
    mlp_metrics = train_mlp(splits, horizons, epochs=args.epochs, lr=args.lr,
                            batch_size=args.batch_size, device=args.device)
    logger.info("MLP: %s", json.dumps(mlp_metrics, indent=2))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"run_{args.symbol}_{ts}.json"
    summary = {
        "symbol": args.symbol,
        "horizons": horizons,
        "n_features": len(splits.feature_cols),
        "feature_cols": splits.feature_cols,
        "splits": {
            "train": len(splits.train_idx),
            "val":   len(splits.val_idx),
            "test":  len(splits.test_idx),
        },
        "xgboost": xgb_metrics,
        "mlp":     mlp_metrics,
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Wrote summary → %s", out_path)


if __name__ == "__main__":
    main()
