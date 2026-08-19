"""
Sanity & overfitting diagnostics on top of walk_forward.

Four tests, run sequentially:
  1. shuffled-label CV , labels permuted, model should collapse to AUC ~0.5
  2. multi-seed CV     , N seeds per fold, within-fold variance vs across-fold variance
  3. permutation importance, shuffle one feature at a time, measure AUC drop
  4. feature correlation matrix, find redundant features

Usage:
    python sanity.py --symbol SPY --horizons 1,5,21 --seeds 10
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import features as feat
from dataset import TabularDataset
from models.mlp import MultiHorizonMLP

logger = logging.getLogger("sanity")


# ── Shared trainer ───────────────────────────────────────────────────────────
def train_one_mlp(X_tr, y_tr, X_val, y_val, *, seed: int, epochs=80,
                  lr=1e-3, batch_size=128, device="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MultiHorizonMLP(n_features=X_tr.shape[1], n_horizons=y_tr.shape[1]).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    loader = DataLoader(TabularDataset(X_tr, y_tr),
                        batch_size=min(batch_size, len(X_tr)),
                        shuffle=True, drop_last=False)
    best_auc, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            loss = loss_fn(model(X), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_p = torch.sigmoid(model(torch.from_numpy(X_val).to(device))).cpu().numpy()
        try:
            v = float(np.nanmean([roc_auc_score(y_val[:, i], val_p[:, i])
                                  for i in range(y_val.shape[1])]))
        except Exception:
            v = -1.0
        if v > best_auc:
            best_auc, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 12:
            break
    model.load_state_dict(best_state)
    return model


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X).to(device))).cpu().numpy()


# ── Fold splitting (matches walk_forward.fold_split) ─────────────────────────
def prep_fold(df, train_idx, test_idx, horizons, val_frac=0.15):
    label_cols = [f"label_{h}d" for h in horizons]
    feat_cols = [c for c in df.columns
                 if c not in ("close",)
                 and not c.startswith("label_")
                 and not c.startswith("fwd_return_")]

    tr = df.loc[train_idx].dropna(subset=feat_cols + label_cols)
    te = df.loc[test_idx].dropna(subset=feat_cols + label_cols)
    if len(tr) < 50 or len(te) < 5:
        return None
    n_val = max(20, int(len(tr) * val_frac))
    val_part = tr.iloc[-n_val:]
    tr_part = tr.iloc[:-n_val]
    sc = StandardScaler()
    X_tr = sc.fit_transform(tr_part[feat_cols].values).astype(np.float32)
    X_val = sc.transform(val_part[feat_cols].values).astype(np.float32)
    X_te = sc.transform(te[feat_cols].values).astype(np.float32)
    return {
        "X_tr": X_tr, "y_tr": tr_part[label_cols].values.astype(np.float32),
        "X_val": X_val, "y_val": val_part[label_cols].values.astype(np.float32),
        "X_test": X_te, "y_test": te[label_cols].values.astype(np.float32),
        "feat_cols": feat_cols,
    }


def iterate_folds(df, horizons, train_min, test_size, step):
    df = df.sort_index().dropna(subset=[f"label_{h}d" for h in horizons])
    n = len(df)
    k = 0
    while True:
        end_train = train_min + k * step
        end_test = end_train + test_size
        if end_test > n:
            return
        yield k, df.index[:end_train], df.index[end_train:end_test]
        k += 1


# ── Test 1: shuffled labels ──────────────────────────────────────────────────
def test_shuffled_labels(df, horizons, *, train_min, test_size, step,
                          n_repeats=3, device="cuda"):
    """Permute label columns within each fold's training set. AUC should be ~0.5."""
    logger.info("=== Shuffled-label test (n_repeats=%d) ===", n_repeats)
    results = defaultdict(list)
    for rep in range(n_repeats):
        rng = np.random.default_rng(rep + 100)
        per_fold = defaultdict(list)
        for fold_k, tr_idx, te_idx in iterate_folds(df, horizons, train_min, test_size, step):
            f = prep_fold(df, tr_idx, te_idx, horizons)
            if f is None:
                continue
            # Shuffle each label column independently in TRAIN+VAL only (test unchanged
            # because we evaluate against true labels)
            y_tr = f["y_tr"].copy()
            y_val = f["y_val"].copy()
            for col in range(y_tr.shape[1]):
                rng.shuffle(y_tr[:, col])
                rng.shuffle(y_val[:, col])
            model = train_one_mlp(f["X_tr"], y_tr, f["X_val"], y_val, seed=rep, device=device)
            probs = predict(model, f["X_test"], device)
            for i, h in enumerate(horizons):
                try:
                    auc = roc_auc_score(f["y_test"][:, i], probs[:, i])
                    per_fold[h].append(auc)
                except ValueError:
                    pass
        for h in horizons:
            clean = [v for v in per_fold[h] if v == v]   # drop NaN
            if clean:
                results[h].append(mean(clean))
                logger.info("  rep %d  %dd  mean AUC across folds: %.3f (n=%d/%d valid)",
                            rep, h, mean(clean), len(clean), len(per_fold[h]))
    summary = {}
    for h in horizons:
        clean = [v for v in results[h] if v == v]
        summary[h] = {
            "mean":   round(mean(clean), 4) if clean else None,
            "std":    round(stdev(clean), 4) if len(clean) > 1 else 0.0,
            "values": [round(v, 4) for v in clean],
            "n_reps": len(clean),
        }
    return summary


# ── Test 2: multi-seed CV ────────────────────────────────────────────────────
def test_multi_seed(df, horizons, *, train_min, test_size, step,
                    n_seeds=10, device="cuda"):
    """For each fold, train n_seeds models. Report within-fold AUC std vs ensemble AUC."""
    logger.info("=== Multi-seed CV (n_seeds=%d) ===", n_seeds)
    per_fold_metrics = []
    for fold_k, tr_idx, te_idx in iterate_folds(df, horizons, train_min, test_size, step):
        f = prep_fold(df, tr_idx, te_idx, horizons)
        if f is None:
            continue
        # Train n_seeds models, collect test predictions
        all_probs = np.zeros((n_seeds, f["X_test"].shape[0], len(horizons)))
        for s in range(n_seeds):
            model = train_one_mlp(f["X_tr"], f["y_tr"], f["X_val"], f["y_val"],
                                  seed=s, device=device)
            all_probs[s] = predict(model, f["X_test"], device)
        ensemble = all_probs.mean(axis=0)

        per_seed_auc = {h: [] for h in horizons}
        ensemble_auc = {}
        for i, h in enumerate(horizons):
            for s in range(n_seeds):
                try:
                    per_seed_auc[h].append(roc_auc_score(f["y_test"][:, i], all_probs[s, :, i]))
                except ValueError:
                    pass
            try:
                ensemble_auc[h] = roc_auc_score(f["y_test"][:, i], ensemble[:, i])
            except ValueError:
                ensemble_auc[h] = None
        per_fold_metrics.append({
            "fold": fold_k,
            "per_seed_auc": {h: per_seed_auc[h] for h in horizons},
            "ensemble_auc": ensemble_auc,
        })
        for h in horizons:
            clean = [v for v in per_seed_auc[h] if v == v]
            if clean:
                within = stdev(clean) if len(clean) > 1 else 0.0
                ens = ensemble_auc.get(h)
                ens_str = f"{ens:.3f}" if (ens is not None and ens == ens) else "nan"
                logger.info("  fold %d  %dd  within-fold std=%.3f  ensemble=%s",
                            fold_k, h, within, ens_str)

    # Aggregate
    out = {}
    for h in horizons:
        within_stds = []
        for m in per_fold_metrics:
            clean = [v for v in m["per_seed_auc"][h] if v == v]
            if len(clean) > 1:
                within_stds.append(stdev(clean))
        ensemble_aucs = [m["ensemble_auc"][h] for m in per_fold_metrics
                         if m["ensemble_auc"][h] is not None and m["ensemble_auc"][h] == m["ensemble_auc"][h]]
        out[h] = {
            "mean_within_fold_std": round(mean(within_stds), 4) if within_stds else None,
            "ensemble_auc_mean":    round(mean(ensemble_aucs), 4) if ensemble_aucs else None,
            "ensemble_auc_std":     round(stdev(ensemble_aucs), 4) if len(ensemble_aucs) > 1 else 0.0,
            "n_folds_eval":         len(ensemble_aucs),
        }
    return out


# ── Test 3: permutation feature importance ───────────────────────────────────
def test_permutation_importance(df, horizons, *, train_min, test_size, step,
                                 n_perm=3, device="cuda"):
    """For each feature, shuffle its test-set values, measure AUC drop per horizon."""
    logger.info("=== Permutation importance (n_perm=%d) ===", n_perm)
    drop = defaultdict(lambda: defaultdict(list))   # drop[feature][horizon] = list of AUC drops
    baseline = defaultdict(list)

    rng = np.random.default_rng(42)
    feat_cols = None
    for fold_k, tr_idx, te_idx in iterate_folds(df, horizons, train_min, test_size, step):
        f = prep_fold(df, tr_idx, te_idx, horizons)
        if f is None:
            continue
        feat_cols = f["feat_cols"]
        model = train_one_mlp(f["X_tr"], f["y_tr"], f["X_val"], f["y_val"],
                              seed=0, device=device)
        probs = predict(model, f["X_test"], device)
        # Baseline AUC per horizon
        for i, h in enumerate(horizons):
            try:
                baseline[h].append(roc_auc_score(f["y_test"][:, i], probs[:, i]))
            except ValueError:
                continue
        # Permute each feature
        for j, fname in enumerate(feat_cols):
            for rep in range(n_perm):
                X_perm = f["X_test"].copy()
                rng.shuffle(X_perm[:, j])
                probs_p = predict(model, X_perm, device)
                for i, h in enumerate(horizons):
                    try:
                        auc_p = roc_auc_score(f["y_test"][:, i], probs_p[:, i])
                        baseline_auc = roc_auc_score(f["y_test"][:, i], probs[:, i])
                        drop[fname][h].append(baseline_auc - auc_p)
                    except ValueError:
                        continue
        logger.info("  fold %d done", fold_k)

    # Aggregate
    importance = {}
    for fname in feat_cols:
        per_h = {}
        for h in horizons:
            clean = [v for v in drop[fname][h] if v == v]
            per_h[h] = round(mean(clean), 4) if clean else None
        importance[fname] = per_h
    return {"baseline_auc": {h: round(mean([v for v in baseline[h] if v == v]), 4)
                                       if [v for v in baseline[h] if v == v] else None
                              for h in horizons},
            "importance": importance,
            "n_folds_evaluated": len(baseline[horizons[0]])}


# ── Test 4: feature correlation ──────────────────────────────────────────────
def test_correlations(df):
    """Pearson correlation matrix among features; return top redundant pairs."""
    logger.info("=== Feature correlations ===")
    label_or_fwd = lambda c: c.startswith("label_") or c.startswith("fwd_return_")
    feat_cols = [c for c in df.columns if c != "close" and not label_or_fwd(c)]
    sub = df[feat_cols].dropna()
    if len(sub) < 50:
        return {"error": "not enough rows after dropna"}
    corr = sub.corr().abs()
    # Find pairs above 0.9
    pairs = []
    for i, a in enumerate(feat_cols):
        for b in feat_cols[i + 1:]:
            r = corr.loc[a, b]
            if r >= 0.9:
                pairs.append((a, b, round(float(r), 3)))
    pairs.sort(key=lambda x: -x[2])
    return {"n_rows": len(sub),
            "high_corr_pairs": pairs[:30],
            "n_high_pairs":  len(pairs)}


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--horizons", default="1,5,21")
    ap.add_argument("--train-min", type=int, default=250)
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--step", type=int, default=21)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--shuffle-reps", type=int, default=3)
    ap.add_argument("--perm-reps", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "runs"))
    ap.add_argument("--skip-multi-seed", action="store_true",
                    help="Skip the multi-seed CV (slow on small datasets)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    horizons = [int(h) for h in args.horizons.split(",")]
    df = feat.build(args.symbol)
    df = feat.add_labels(df, horizons=horizons)
    logger.info("Loaded %s: %s", args.symbol, df.shape)

    summary = {}

    summary["correlations"] = test_correlations(df)
    summary["shuffled_labels"] = test_shuffled_labels(
        df, horizons,
        train_min=args.train_min, test_size=args.test, step=args.step,
        n_repeats=args.shuffle_reps, device=args.device,
    )
    if not args.skip_multi_seed:
        summary["multi_seed"] = test_multi_seed(
            df, horizons,
            train_min=args.train_min, test_size=args.test, step=args.step,
            n_seeds=args.seeds, device=args.device,
        )
    summary["permutation_importance"] = test_permutation_importance(
        df, horizons,
        train_min=args.train_min, test_size=args.test, step=args.step,
        n_perm=args.perm_reps, device=args.device,
    )

    # ── Print pretty summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SANITY DIAGNOSTICS, SUMMARY")
    print("=" * 70)
    print("\n[1] Correlation: %d feature pairs |r| ≥ 0.9 (top shown)"
          % summary["correlations"].get("n_high_pairs", 0))
    for a, b, r in summary["correlations"].get("high_corr_pairs", [])[:10]:
        print(f"      r={r:.3f}   {a}  ↔  {b}")

    print("\n[2] Shuffled-label test (AUC should be ~0.50 if no leak):")
    for h in horizons:
        s = summary["shuffled_labels"][h]
        if s["mean"] is None:
            print(f"      {h:>3}d   (no valid folds)")
        else:
            tag = " ⚠ LEAK SUSPECT" if s["mean"] > 0.55 else " ✓ clean"
            print(f"      {h:>3}d   AUC {s['mean']:.3f} ± {s['std']:.3f}{tag}")

    if "multi_seed" in summary:
        print("\n[3] Multi-seed CV (n_seeds=%d):" % args.seeds)
        for h in horizons:
            m = summary["multi_seed"][h]
            print(f"      {h:>3}d   ensemble AUC {m['ensemble_auc_mean']} ± {m['ensemble_auc_std']} "
                  f"| within-fold std {m['mean_within_fold_std']} | n_folds={m['n_folds_eval']}")

    print("\n[4] Permutation importance (mean AUC drop when feature shuffled):")
    imp = summary["permutation_importance"]["importance"]
    baseline = summary["permutation_importance"]["baseline_auc"]
    print(f"     Baseline AUC: " + " | ".join(f"{h}d={baseline[h]}" for h in horizons))
    for h in horizons:
        print(f"\n      Top features for {h}d horizon:")
        ranked = sorted(imp.items(), key=lambda kv: -kv[1][h])[:12]
        for fname, drops in ranked:
            print(f"        {fname:<20s}  drop={drops[h]:+.4f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    p = out / f"sanity_{args.symbol}_{ts}.json"
    p.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved → {p}")


if __name__ == "__main__":
    main()
