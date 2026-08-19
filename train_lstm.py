"""
Train the multi-scale, multi-horizon LSTM on SR-gated ternary labels.

Modes
-----
  Static split (fast sanity check):
      python train_lstm.py --symbol SPY

  Walk-forward CV (proper evaluation):
      python train_lstm.py --symbol SPY --walk-forward

  Multi-symbol training (shared weights):
      python train_lstm.py --symbol SPY,QQQ,IWM --walk-forward

Examples
--------
  python train_lstm.py --symbol SPY --epochs 60 --threshold 0.3
  python train_lstm.py --symbol SPY --walk-forward --min-train 500 --val-bars 63
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.utils.data import DataLoader

import features as feat
from dataset import (
    make_sequence_arrays, split_sequence_arrays,
    MultiWindowDataset, WalkForwardSplitter, WalkForwardFold,
)
from models.lstm import MultiScaleLSTM

logger = logging.getLogger("train_lstm")

N_CLASSES = 3    # 0=short, 1=abstain, 2=long
HORIZONS = [1, 5, 21]
LABEL_NAMES = {0: "short", 1: "abstain", 2: "long"}


# ── Metrics ──────────────────────────────────────────────────────────────────
def directional_sharpe(
    probs: np.ndarray,      # (N, 3) softmax probs
    fwd_returns: np.ndarray,
    long_thresh: float = 0.45,
    short_thresh: float = 0.45,
) -> float:
    """
    P(long) - P(short) → signed position, filtered by confidence threshold.
    Annualized Sharpe on realized forward returns.
    """
    long_p  = probs[:, 2]   # class 2 = long
    short_p = probs[:, 0]   # class 0 = short
    edge = long_p - short_p
    pos = np.where(long_p > long_thresh, 1.0,
          np.where(short_p > short_thresh, -1.0, 0.0))
    pnl = pos * fwd_returns
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) == 0 or pnl.std() == 0:
        return 0.0
    return float(np.sqrt(252) * pnl.mean() / pnl.std())


def win_rate(probs: np.ndarray, fwd_returns: np.ndarray,
             long_thresh: float = 0.45, short_thresh: float = 0.45) -> float:
    long_p  = probs[:, 2]
    short_p = probs[:, 0]
    traded = (long_p > long_thresh) | (short_p > short_thresh)
    if traded.sum() == 0:
        return float("nan")
    pos = np.where(long_p > long_thresh, 1.0, -1.0)[traded]
    pnl = pos * fwd_returns[traded]
    return float((pnl > 0).mean())


def eval_horizon(
    probs: np.ndarray,         # (N, 3)
    y_true: np.ndarray,        # (N,) class indices 0/1/2
    fwd_returns: np.ndarray | None,
) -> dict:
    pred = probs.argmax(axis=1)
    # AUC for long-vs-rest (class 2)
    try:
        auc_long = float(roc_auc_score((y_true == 2).astype(int), probs[:, 2]))
    except Exception:
        auc_long = float("nan")
    out = {
        "auc_long":       auc_long,
        "acc":            float(accuracy_score(y_true, pred)),
        "f1_macro":       float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "abstain_rate":   float((pred == 1).mean()),
        "long_rate":      float((pred == 2).mean()),
        "short_rate":     float((pred == 0).mean()),
        "label_dist":     {LABEL_NAMES[k]: int((y_true == k).sum()) for k in range(3)},
    }
    if fwd_returns is not None:
        out["sharpe"]   = directional_sharpe(probs, fwd_returns)
        out["win_rate"] = win_rate(probs, fwd_returns)
    return out


# ── Training loop ─────────────────────────────────────────────────────────────
def _build_class_weights(y: np.ndarray, device: str) -> torch.Tensor:
    """Inverse-frequency weights to handle abstain-heavy imbalance."""
    counts = np.bincount(y.flatten(), minlength=N_CLASSES).astype(float)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (N_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_fold(
    train_arr,
    val_arr,
    fwd_val: np.ndarray | None,
    *,
    n_features: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    patience: int = 15,
) -> tuple[MultiScaleLSTM, dict]:

    model = MultiScaleLSTM(n_features=n_features, n_horizons=len(HORIZONS)).to(device)
    logger.info("Model parameters: %d", model.num_parameters)

    class_weights = _build_class_weights(train_arr.y, device)
    loss_fns = [nn.CrossEntropyLoss(weight=class_weights) for _ in HORIZONS]

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_loader = DataLoader(
        MultiWindowDataset(train_arr),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(MultiWindowDataset(val_arr), batch_size=512)

    best_auc  = -1.0
    best_state = None
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for (x1d, x1w, x1m), y_batch in train_loader:
            x1d, x1w, x1m = x1d.to(device), x1w.to(device), x1m.to(device)
            y_batch = y_batch.to(device)  # (B, n_horizons)
            logits_list = model(x1d, x1w, x1m)
            loss = sum(
                loss_fns[i](logits_list[i], y_batch[:, i])
                for i in range(len(HORIZONS))
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(y_batch)
        ep_loss /= len(train_loader.dataset)
        sched.step()

        # Validation AUC
        model.eval()
        all_probs = [[] for _ in HORIZONS]
        all_y     = []
        with torch.no_grad():
            for (x1d, x1w, x1m), y_batch in val_loader:
                x1d, x1w, x1m = x1d.to(device), x1w.to(device), x1m.to(device)
                prob_list = model.predict_proba(x1d, x1w, x1m)
                for i, p in enumerate(prob_list):
                    all_probs[i].append(p.cpu().numpy())
                all_y.append(y_batch.numpy())

        all_y = np.concatenate(all_y, axis=0)
        val_aucs = []
        for i in range(len(HORIZONS)):
            probs_i = np.concatenate(all_probs[i], axis=0)
            try:
                val_aucs.append(roc_auc_score((all_y[:, i] == 2).astype(int), probs_i[:, 2]))
            except Exception:
                val_aucs.append(0.5)
        mean_auc = float(np.mean(val_aucs))

        if mean_auc > best_auc:
            best_auc   = mean_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if ep % 5 == 0 or ep == 1:
            logger.info("ep %3d  loss %.4f  val_auc %.4f  (best %.4f)",
                        ep, ep_loss, mean_auc, best_auc)
        if bad_epochs >= patience:
            logger.info("Early stop at epoch %d", ep)
            break

    model.load_state_dict(best_state)
    model.eval()

    # Full val metrics
    all_probs = [[] for _ in HORIZONS]
    all_y = []
    with torch.no_grad():
        for (x1d, x1w, x1m), y_batch in val_loader:
            x1d, x1w, x1m = x1d.to(device), x1w.to(device), x1m.to(device)
            for i, p in enumerate(model.predict_proba(x1d, x1w, x1m)):
                all_probs[i].append(p.cpu().numpy())
            all_y.append(y_batch.numpy())
    all_y = np.concatenate(all_y, axis=0)

    metrics = {}
    for i, h in enumerate(HORIZONS):
        probs_i = np.concatenate(all_probs[i], axis=0)
        fwd_i = fwd_val[:, i] if fwd_val is not None else None
        metrics[f"{h}d"] = eval_horizon(probs_i, all_y[:, i], fwd_i)

    return model, metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY",
                    help="Comma-separated list for multi-symbol training")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="SR threshold for ternary label (default 0.3)")
    ap.add_argument("--walk-forward", action="store_true",
                    help="Use walk-forward CV instead of static split")
    ap.add_argument("--min-train", type=int, default=500,
                    help="Min train bars for walk-forward (default 500)")
    ap.add_argument("--val-bars", type=int, default=63,
                    help="Val window size in bars for walk-forward (default 63 ≈ 1Q)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "runs"))
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logger.info("Device: %s | threshold: %.2f", args.device, args.threshold)

    # Build feature matrix (concatenate multiple symbols if requested)
    symbols = [s.strip() for s in args.symbol.split(",")]
    frames = []
    for sym in symbols:
        logger.info("Building features for %s", sym)
        f = feat.build(sym)
        f = feat.add_sr_labels(f, horizons=HORIZONS, threshold=args.threshold)
        f["_symbol"] = sym
        frames.append(f)
    df = pd.concat(frames).sort_index() if len(frames) > 1 else frames[0]
    logger.info("Combined frame: %s rows, %s cols", len(df), len(df.columns))

    out_dir = Path(args.out_dir)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"lstm_{args.symbol.replace(',','_')}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.walk_forward:
        splitter = WalkForwardSplitter(
            min_train_bars=args.min_train,
            val_bars=args.val_bars,
        )
        fold_results = []
        best_model = None
        best_fold_auc = -1.0

        for fold in splitter.split(df, horizons=HORIZONS, ternary=True):
            logger.info("=== Fold %d | train=%d val=%d ===",
                        fold.fold, len(fold.train_idx), len(fold.val_idx))

            # Build sequence arrays for this fold
            train_df = df.loc[fold.train_idx]
            val_df   = df.loc[fold.val_idx]

            train_arr, scaler = make_sequence_arrays(
                train_df, horizons=HORIZONS, ternary=True, fit_scaler=True
            )
            val_arr, _ = make_sequence_arrays(
                val_df, horizons=HORIZONS, ternary=True,
                scaler=scaler, fit_scaler=False,
            )

            if len(train_arr.dates) < 50 or len(val_arr.dates) < 5:
                logger.warning("Fold %d too small, skipping", fold.fold)
                continue

            fwd_val = fold.fwd_val

            model, metrics = train_one_fold(
                train_arr, val_arr, fwd_val,
                n_features=train_arr.x_1d.shape[1],
                epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size, device=args.device,
            )

            mean_auc = np.mean([v["auc_long"] for v in metrics.values()
                                if not np.isnan(v["auc_long"])])
            fold_results.append({"fold": fold.fold, "metrics": metrics,
                                 "mean_auc": float(mean_auc)})
            logger.info("Fold %d  mean_auc=%.4f", fold.fold, mean_auc)
            logger.info("Metrics: %s", json.dumps(metrics, indent=2, default=str))

            if mean_auc > best_fold_auc:
                best_fold_auc = mean_auc
                best_model = model

        summary = {
            "mode": "walk_forward", "symbol": args.symbol,
            "horizons": HORIZONS, "threshold": args.threshold,
            "n_folds": len(fold_results),
            "mean_auc_across_folds": float(
                np.mean([r["mean_auc"] for r in fold_results])
            ) if fold_results else 0.0,
            "folds": fold_results,
        }

    else:
        # Static split, fast development mode
        arr, scaler = make_sequence_arrays(df, horizons=HORIZONS, ternary=True)
        splits = split_sequence_arrays(arr)
        train_arr, val_arr, test_arr = splits["train"], splits["val"], splits["test"]
        logger.info("Static split, train=%d val=%d test=%d",
                    len(train_arr.dates), len(val_arr.dates), len(test_arr.dates))

        model, val_metrics = train_one_fold(
            train_arr, val_arr, fwd_val=None,
            n_features=train_arr.x_1d.shape[1],
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=args.device,
        )
        best_model = model
        summary = {
            "mode": "static", "symbol": args.symbol,
            "horizons": HORIZONS, "threshold": args.threshold,
            "val": val_metrics,
        }

    # Save model + summary
    model_path = run_dir / "model.pt"
    if best_model is not None:
        torch.save(best_model.state_dict(), model_path)
        logger.info("Model saved → %s", model_path)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Summary → %s", summary_path)
    logger.info("\n%s", json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
