"""
XGBoost variant, same data, same diagnostics, different algorithm.

XGBoost's tree-based splits + reg_alpha/reg_lambda penalize over-reliance on
a single feature (the macro_treasury_2y problem the MLP hit). Train one model
per horizon, evaluate the 1-day-ahead model in detail.

Outputs match diagnose.py:
  1. Prediction histogram
  2. Threshold sweep
  3. Calibration plot
  4. Feature importance (XGBoost-native, uses gain not perturbation)
  5. Backtest with optimal threshold
"""
from __future__ import annotations
import os

import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features_intraday as fi

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("diagnose_xgb")

HORIZONS_BARS = [1, 4, 26]
BARS_PER_YEAR = 252 * 26
BARS_PER_DAY = 26


# ── Setup ────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).resolve().parent / "runs" / f"diagnose_xgb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
out_dir.mkdir(parents=True, exist_ok=True)
logger.info("Output → %s", out_dir)

df = fi.build("SPY", timeframe="15mins")
df = fi.add_labels(df, horizons_bars=HORIZONS_BARS)
label_cols = [f"label_{h}b" for h in HORIZONS_BARS]
feat_cols = [c for c in df.columns
             if c not in ("close",)
             and not c.startswith("label_")
             and not c.startswith("fwd_return_")
             and df[c].isna().sum() / len(df) < 0.15]
df = df.sort_index().dropna(subset=label_cols + feat_cols).copy()
logger.info("Universe: %d bars × %d features", len(df), len(feat_cols))

# Same split as MLP diagnostic for direct comparison
n = len(df)
n_test = int(n * 0.20)
n_val  = int((n - n_test) * 0.15)
tr_part = df.iloc[:n - n_test - n_val]
val_part = df.iloc[n - n_test - n_val: n - n_test]
te_part  = df.iloc[n - n_test:]
logger.info("Train %d | Val %d | Test %d", len(tr_part), len(val_part), len(te_part))

# XGBoost doesn't need scaling but it doesn't hurt for direct comparison
scaler = StandardScaler()
X_tr = scaler.fit_transform(tr_part[feat_cols].values).astype(np.float32)
X_val = scaler.transform(val_part[feat_cols].values).astype(np.float32)
X_te = scaler.transform(te_part[feat_cols].values).astype(np.float32)
y_tr_1d = tr_part["label_26b"].values.astype(int)
y_val_1d = val_part["label_26b"].values.astype(int)
y_te_1d = te_part["label_26b"].values.astype(int)
fwd_ret_1d = te_part["fwd_return_26b"].values

logger.info("Training XGBoost (1-day-ahead head, early-stop on val AUC)...")
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=8,
    reg_alpha=0.2,
    reg_lambda=1.5,
    eval_metric="auc",
    tree_method="hist",
    n_jobs=-1,
    early_stopping_rounds=40,
)
model.fit(X_tr, y_tr_1d, eval_set=[(X_val, y_val_1d)], verbose=False)
logger.info("Best iteration: %d", model.best_iteration)

test_preds = model.predict_proba(X_te)[:, 1]
test_auc = roc_auc_score(y_te_1d, test_preds)
base_rate = float(y_te_1d.mean())
logger.info("Test AUC: %.4f  |  base rate: %.3f  |  mean pred: %.3f  |  median: %.3f",
            test_auc, base_rate, float(np.mean(test_preds)), float(np.median(test_preds)))


# ─── 1. Prediction histogram ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
ax.hist(test_preds, bins=40, color="#1f77b4", alpha=0.7, edgecolor="k")
ax.axvline(0.5, color="r", linestyle="--", label="threshold = 0.5")
ax.axvline(np.mean(test_preds), color="green", label=f"mean = {np.mean(test_preds):.3f}")
ax.axvline(base_rate, color="purple", linestyle=":", label=f"base rate = {base_rate:.3f}")
ax.set_xlabel("Predicted prob (1-day-ahead up)")
ax.set_title(f"XGBoost prediction distribution (test, AUC={test_auc:.3f})")
ax.legend()

ax = axes[1]
ups = test_preds[y_te_1d == 1]
downs = test_preds[y_te_1d == 0]
ax.hist(ups, bins=30, alpha=0.6, label=f"True UP (n={len(ups)})", color="green")
ax.hist(downs, bins=30, alpha=0.6, label=f"True DOWN (n={len(downs)})", color="red")
ax.axvline(0.5, color="k", linestyle="--", alpha=0.5)
ax.set_xlabel("Predicted prob")
ax.set_title("XGBoost predictions split by actual outcome")
ax.legend()
plt.tight_layout()
plt.savefig(out_dir / "1_prediction_histogram.png", dpi=110, bbox_inches="tight")
plt.close()


# ─── 2. Threshold sweep ────────────────────────────────────────────────────
def at_decision_time(idx):
    et = idx.tz_convert("America/New_York")
    return (et.hour == 15) & (et.minute == 45)

test_index = te_part.index
close = te_part["close"]
bar_returns = close.pct_change().fillna(0)
is_decision = at_decision_time(test_index)

preds_series = pd.Series(test_preds, index=test_index)
threshold_results = []
for thresh in np.arange(0.35, 0.66, 0.025):
    raw = pd.Series(np.nan, index=test_index)
    raw.loc[is_decision] = (preds_series[is_decision] > thresh).astype(float)
    filled = raw.ffill().fillna(0.0)
    position = filled.shift(1).fillna(0.0)
    pos_change = position.diff().abs(); pos_change.iloc[0] = position.iloc[0]
    cost = pos_change * 1.0 / 10_000.0
    strat_ret = position * bar_returns - cost
    cum = (1 + strat_ret).cumprod()
    total = float(cum.iloc[-1] - 1)
    ann_vol = float(strat_ret.std() * np.sqrt(BARS_PER_YEAR))
    ann_ret = (1 + total) ** (BARS_PER_YEAR / len(strat_ret)) - 1 if total > -1 else -1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    avg_exp = float(position.mean())
    threshold_results.append({
        "threshold": round(thresh, 3),
        "total_return_pct": round(total * 100, 2),
        "ann_return_pct": round(ann_ret * 100, 2),
        "sharpe": round(sharpe, 3),
        "avg_exposure": round(avg_exp, 3),
    })

threshold_df = pd.DataFrame(threshold_results)
print("\n" + "=" * 80)
print("XGBOOST THRESHOLD SWEEP")
print("=" * 80)
print(threshold_df.to_string(index=False))

bh_total = float((1 + bar_returns).cumprod().iloc[-1] - 1)
bh_ann_vol = float(bar_returns.std() * np.sqrt(BARS_PER_YEAR))
bh_ann_ret = (1 + bh_total) ** (BARS_PER_YEAR / len(bar_returns)) - 1
bh_sharpe = bh_ann_ret / bh_ann_vol if bh_ann_vol > 0 else 0
print(f"\nBuy & Hold same period: total={bh_total*100:.2f}%, ann_ret={bh_ann_ret*100:.2f}%, Sharpe={bh_sharpe:.3f}")

best_row = threshold_df.iloc[threshold_df["sharpe"].idxmax()]
print(f"\nBest threshold by Sharpe: {best_row['threshold']} → "
      f"total {best_row['total_return_pct']}%, Sharpe {best_row['sharpe']}, "
      f"exposure {best_row['avg_exposure']}")

fig, ax = plt.subplots(figsize=(11, 5))
ax2 = ax.twinx()
ax.bar(threshold_df["threshold"], threshold_df["total_return_pct"], width=0.018,
       color="#1f77b4", alpha=0.7, label="Total return %")
ax2.plot(threshold_df["threshold"], threshold_df["sharpe"], "o-", color="orange",
         linewidth=2, label="Sharpe")
ax.axhline(bh_total * 100, color="gray", linestyle="--", label=f"B&H total {bh_total*100:.1f}%")
ax.set_xlabel("Threshold"); ax.set_ylabel("Total return %", color="#1f77b4")
ax2.set_ylabel("Sharpe", color="orange")
ax.set_title("XGBoost threshold sweep")
ax.grid(True, alpha=0.3); ax.legend(loc="upper left"); ax2.legend(loc="upper right")
plt.tight_layout()
plt.savefig(out_dir / "2_threshold_sweep.png", dpi=110, bbox_inches="tight")
plt.close()


# ─── 3. Calibration plot ───────────────────────────────────────────────────
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
cal_pred, cal_actual, cal_n = [], [], []
for i in range(n_bins):
    in_bin = (test_preds >= bin_edges[i]) & (test_preds < bin_edges[i + 1])
    if in_bin.sum() < 5:
        cal_pred.append(None); cal_actual.append(None); cal_n.append(0)
        continue
    cal_pred.append(float(test_preds[in_bin].mean()))
    cal_actual.append(float(y_te_1d[in_bin].mean()))
    cal_n.append(int(in_bin.sum()))

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
xs = [p for p in cal_pred if p is not None]
ys = [a for a, p in zip(cal_actual, cal_pred) if p is not None]
ns = [n for n, p in zip(cal_n, cal_pred) if p is not None]
ax.scatter(xs, ys, s=[max(40, n * 1.5) for n in ns], alpha=0.6, color="#1f77b4")
for x, y, n in zip(xs, ys, ns):
    ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
ax.axhline(base_rate, color="g", linestyle=":", alpha=0.7, label=f"base rate = {base_rate:.3f}")
ax.set_xlabel("Mean predicted prob in bin"); ax.set_ylabel("Actual up rate in bin")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("XGBoost calibration plot")
ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(out_dir / "3_calibration.png", dpi=110, bbox_inches="tight")
plt.close()


# ─── 4. Feature importance (gain-based, XGBoost native) ────────────────────
booster = model.get_booster()
importance_dict = booster.get_score(importance_type="gain")
importance_pairs = sorted(
    [(feat_cols[int(f.replace("f", ""))], v) for f, v in importance_dict.items()],
    key=lambda x: -x[1],
)
print("\n" + "=" * 80)
print("XGBOOST FEATURE IMPORTANCE (top 20 by gain)")
print("=" * 80)
total_gain = sum(g for _, g in importance_pairs)
for name, gain in importance_pairs[:20]:
    bar = "█" * int(gain / importance_pairs[0][1] * 50)
    print(f"  {name:<28s}  {gain:>8.1f}  ({gain/total_gain*100:.1f}%)  {bar}")

# Plot
fig, ax = plt.subplots(figsize=(11, 8))
top = importance_pairs[:20][::-1]
names = [n for n, _ in top]
gains = [g for _, g in top]
ax.barh(names, gains, color="#1f77b4")
ax.set_xlabel("Gain (XGBoost)")
ax.set_title("XGBoost feature importance (top 20)")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(out_dir / "4_feature_importance.png", dpi=110, bbox_inches="tight")
plt.close()


# ─── 5. Summary + save ─────────────────────────────────────────────────────
summary = {
    "model": "XGBoost (gbtree)",
    "params": model.get_xgb_params(),
    "best_iteration": int(model.best_iteration),
    "config": {
        "train_period": [str(tr_part.index[0])[:16], str(tr_part.index[-1])[:16]],
        "val_period":   [str(val_part.index[0])[:16], str(val_part.index[-1])[:16]],
        "test_period":  [str(te_part.index[0])[:16], str(te_part.index[-1])[:16]],
        "n_features": len(feat_cols),
    },
    "test_metrics": {
        "auc_1d": round(test_auc, 4),
        "base_rate": round(base_rate, 4),
        "mean_pred": round(float(np.mean(test_preds)), 4),
        "median_pred": round(float(np.median(test_preds)), 4),
        "frac_pred_above_05": round(float((test_preds > 0.5).mean()), 4),
        "calibration_bias": round(float(np.mean(test_preds)) - base_rate, 4),
    },
    "threshold_sweep": threshold_results,
    "best_threshold": {
        "threshold": float(best_row["threshold"]),
        "total_return_pct": float(best_row["total_return_pct"]),
        "sharpe": float(best_row["sharpe"]),
        "avg_exposure": float(best_row["avg_exposure"]),
    },
    "buy_hold_test_period": {
        "total_return_pct": round(bh_total * 100, 2),
        "sharpe": round(bh_sharpe, 3),
    },
    "top_20_features_by_gain": [
        {"feature": n, "gain": round(g, 1), "pct": round(g/total_gain*100, 2)}
        for n, g in importance_pairs[:20]
    ],
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

print("\n" + "=" * 80)
print("XGBOOST vs MLP KEY COMPARISON")
print("=" * 80)
print(f"  {'Metric':<32}{'MLP':>12}{'XGBoost':>12}")
print(f"  {'Test AUC':<32}{'0.5713':>12}{test_auc:>12.4f}")
print(f"  {'Mean prediction':<32}{'0.052':>12}{float(np.mean(test_preds)):>12.4f}")
print(f"  {'Median prediction':<32}{'0.001':>12}{float(np.median(test_preds)):>12.4f}")
print(f"  {'Frac preds > 0.5':<32}{'0.036':>12}{float((test_preds > 0.5).mean()):>12.4f}")
print(f"  {'Base rate (target)':<32}{'0.623':>12}{base_rate:>12.4f}")

print(f"\nArtifacts → {out_dir}")
