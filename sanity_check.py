"""
Quick end-to-end sanity check. Run this after the backfill completes.
Verifies: data loads, features build, labels compute, dataset assembles,
model forward pass runs, label distribution looks sane.

    python sanity_check.py
    python sanity_check.py --symbol QQQ --threshold 0.25
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

import features as feat
from dataset import make_sequence_arrays, split_sequence_arrays, MultiWindowDataset
from models.lstm import MultiScaleLSTM
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("sanity")

HORIZONS = [1, 5, 21]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()

    log.info("── Features ─────────────────────────────────")
    f = feat.build(args.symbol)
    log.info("Raw feature matrix: %d rows × %d cols", *f.shape)
    missing = f.isna().sum()
    bad = missing[missing > 0]
    if len(bad):
        log.info("Columns with NaN (expected for macro/cross-asset near start):")
        for col, n in bad.items():
            log.info("  %-30s  %d NaN", col, n)

    log.info("\n── Labels (threshold=%.2f) ─────────────────", args.threshold)
    f = feat.add_sr_labels(f, horizons=HORIZONS, threshold=args.threshold)
    for h in HORIZONS:
        col = f[f"sr_label_{h}d"].dropna()
        dist = col.value_counts().sort_index().to_dict()
        abstain_pct = (col == 0).mean()
        log.info("  %2dd: short=%d  abstain=%d  long=%d  (abstain %.0f%%)",
                 h, dist.get(-1, 0), dist.get(0, 0), dist.get(1, 0),
                 abstain_pct * 100)

    log.info("\n── Sequence arrays ─────────────────────────")
    arr, scaler = make_sequence_arrays(f, horizons=HORIZONS, ternary=True)
    splits = split_sequence_arrays(arr)
    train_arr = splits["train"]
    log.info("  x_1d: %s  x_1w: %s  x_1m: %s  y: %s",
             train_arr.x_1d.shape, train_arr.x_1w.shape,
             train_arr.x_1m.shape, train_arr.y.shape)
    log.info("  feature count: %d", len(arr.feature_cols))
    log.info("  train=%d  val=%d  test=%d",
             len(splits["train"].dates), len(splits["val"].dates), len(splits["test"].dates))

    log.info("\n── Model forward pass ──────────────────────")
    n_features = arr.x_1d.shape[1]
    model = MultiScaleLSTM(n_features=n_features, n_horizons=len(HORIZONS))
    log.info("  Parameters: %d", model.num_parameters)

    loader = DataLoader(MultiWindowDataset(train_arr), batch_size=32, shuffle=True)
    (x1d, x1w, x1m), y = next(iter(loader))
    logits = model(x1d, x1w, x1m)
    assert len(logits) == len(HORIZONS), "Expected one logit tensor per horizon"
    assert logits[0].shape == (len(x1d), 3), f"Unexpected logit shape: {logits[0].shape}"
    log.info("  Forward pass OK, logit shapes: %s", [l.shape for l in logits])

    log.info("\n── All checks passed ✓ ──────────────────────")
    log.info("Ready to train:  python train_lstm.py --symbol %s --threshold %.2f",
             args.symbol, args.threshold)
    log.info("Walk-forward CV: python train_lstm.py --symbol %s --threshold %.2f --walk-forward",
             args.symbol, args.threshold)


if __name__ == "__main__":
    main()
