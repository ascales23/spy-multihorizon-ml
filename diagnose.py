"""
Model diagnostics, figure out WHY the NN goes flat during rallies.

Four analyses on a single train-once model (trained on first 2000 bars of
non-NaN data, evaluated on the remaining bars):

  1. Prediction histogram, is the model's median prediction below 0.5?
  2. Threshold sweep, total return / Sharpe at thresholds [0.35..0.60]
  3. Calibration plot, when model says X, does it go up X% of the time?
  4. Feature attribution, on wrong-call days (model says < 0.5, market went up
     in next 26 bars), which features pushed prediction down the most?

Output: PNG charts + text summary to ~/ml_pipeline/runs/diagnose_<ts>/
"""
from __future__ import annotations
import os

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
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("diagnose")

HORIZONS_BARS = [1, 4, 26]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BARS_PER_YEAR = 252 * 26
BARS_PER_DAY = 26


def train_seed(X_tr, y_tr, X_val, y_val, *, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MultiHorizonMLP(X_tr.shape[1], y_tr.shape[1]).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TabularDataset(X_tr, y_tr),
                        batch_size=min(256, len(X_tr)), shuffle=True)
    best, best_state, bad = -1.0, None, 0
    for _ in range(50):
        m.train()
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            loss = loss_fn(m(X), y); opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vp = torch.sigmoid(m(torch.from_numpy(X_val).to(DEVICE))).cpu().numpy()
        try:
            v = float(np.nanmean([roc_auc_score(y_val[:,i], vp[:,i]) for i in range(y_val.shape[1])]))
        except Exception:
            v = -1.0
        if v > best:
            best, best_state, bad = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 8:
            break
    if best_state:
        m.load_state_dict(best_state)
    return m


def predict(models, X):
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(torch.sigmoid(m(torch.from_numpy(X).to(DEVICE))).cpu().numpy())
    return np.stack(preds).mean(axis=0)


# ── Setup ────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).resolve().parent / "runs" / f"diagnose_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
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

# Single-shot training: use first 70% for train, next 15% val, last 15% test
n = len(df)
n_test = int(n * 0.20)
n_val  = int((n - n_test) * 0.15)
tr_part = df.iloc[:n - n_test - n_val]
val_part = df.iloc[n - n_test - n_val: n - n_test]
te_part  = df.iloc[n - n_test:]
logger.info("Train: %d bars (%s → %s)",
            len(tr_part), str(tr_part.index[0])[:16], str(tr_part.index[-1])[:16])
logger.info("Val:   %d bars (%s → %s)",
            len(val_part), str(val_part.index[0])[:16], str(val_part.index[-1])[:16])
logger.info("Test:  %d bars (%s → %s)",
            len(te_part), str(te_part.index[0])[:16], str(te_part.index[-1])[:16])

scaler = StandardScaler()
X_tr = scaler.fit_transform(tr_part[feat_cols].values).astype(np.float32)
X_val = scaler.transform(val_part[feat_cols].values).astype(np.float32)
X_te = scaler.transform(te_part[feat_cols].values).astype(np.float32)
y_tr = tr_part[label_cols].values.astype(np.float32)
y_val = val_part[label_cols].values.astype(np.float32)
y_te = te_part[label_cols].values.astype(np.float32)

logger.info("Training 5-seed ensemble...")
models = [train_seed(X_tr, y_tr, X_val, y_val, seed=s) for s in range(5)]
test_preds = predict(models, X_te)
# 1-day-ahead is horizon index 2 (HORIZONS_BARS = [1,4,26])
h_idx = 2
preds_1d = test_preds[:, h_idx]
y_test_1d = y_te[:, h_idx]
fwd_ret_1d = te_part["fwd_return_26b"].values

test_auc = roc_auc_score(y_test_1d, preds_1d)
logger.info("Test AUC for 1-day-ahead (26 bars): %.4f", test_auc)


# ─── 1. Prediction histogram ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
ax.hist(preds_1d, bins=40, color="#1f77b4", alpha=0.7, edgecolor="k")
ax.axvline(0.5, color="r", linestyle="--", label="threshold = 0.5")
ax.axvline(np.mean(preds_1d), color="green", linestyle="-", label=f"mean = {np.mean(preds_1d):.3f}")
ax.axvline(np.median(preds_1d), color="orange", linestyle="-", label=f"median = {np.median(preds_1d):.3f}")
ax.set_xlabel("Predicted prob (1-day-ahead up)")
ax.set_ylabel("Count")
ax.set_title(f"Prediction distribution (test period, AUC={test_auc:.3f})")
ax.legend()

# Split histogram by actual outcome
ax = axes[1]
ups = preds_1d[y_test_1d == 1]
downs = preds_1d[y_test_1d == 0]
ax.hist(ups, bins=30, alpha=0.6, label=f"True UP (n={len(ups)})", color="green")
ax.hist(downs, bins=30, alpha=0.6, label=f"True DOWN (n={len(downs)})", color="red")
ax.axvline(0.5, color="k", linestyle="--", alpha=0.5)
ax.set_xlabel("Predicted prob")
ax.set_ylabel("Count")
ax.set_title("Predictions split by actual outcome")
ax.legend()
plt.tight_layout()
plt.savefig(out_dir / "1_prediction_histogram.png", dpi=110, bbox_inches="tight")
plt.close()

base_rate = float(y_test_1d.mean())
logger.info("Base rate (actual up rate): %.3f | Mean prediction: %.3f | Median: %.3f",
            base_rate, float(np.mean(preds_1d)), float(np.median(preds_1d)))

# ─── 2. Threshold sweep ────────────────────────────────────────────────────
def at_decision_time(idx):
    idx_et = idx.tz_convert("America/New_York")
    return (idx_et.hour == 15) & (idx_et.minute == 45)

test_index = te_part.index
close = te_part["close"]
bar_returns = close.pct_change().fillna(0)
is_decision = at_decision_time(test_index)

preds_series = pd.Series(preds_1d, index=test_index)
threshold_results = []
for thresh in np.arange(0.35, 0.61, 0.025):
    raw = pd.Series(np.nan, index=test_index)
    raw.loc[is_decision] = (preds_series[is_decision] > thresh).astype(float)
    filled = raw.ffill().fillna(0.0)
    position = filled.shift(1).fillna(0.0)
    pos_change = position.diff().abs()
    pos_change.iloc[0] = position.iloc[0]
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
print("THRESHOLD SWEEP")
print("=" * 80)
print(threshold_df.to_string(index=False))

# Buy & hold over same period for comparison
bh_total = float((1 + bar_returns).cumprod().iloc[-1] - 1)
bh_ann_vol = float(bar_returns.std() * np.sqrt(BARS_PER_YEAR))
bh_ann_ret = (1 + bh_total) ** (BARS_PER_YEAR / len(bar_returns)) - 1
bh_sharpe = bh_ann_ret / bh_ann_vol if bh_ann_vol > 0 else 0
print(f"\nBuy & Hold same period: total={bh_total*100:.2f}%, ann_ret={bh_ann_ret*100:.2f}%, Sharpe={bh_sharpe:.3f}")

fig, ax = plt.subplots(figsize=(11, 5))
ax2 = ax.twinx()
ax.bar(threshold_df["threshold"], threshold_df["total_return_pct"], width=0.018,
       color="#1f77b4", alpha=0.7, label="Total return %")
ax2.plot(threshold_df["threshold"], threshold_df["sharpe"], "o-", color="orange",
         linewidth=2, label="Sharpe")
ax.axhline(bh_total * 100, color="gray", linestyle="--", label=f"B&H total {bh_total*100:.1f}%")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Total return %", color="#1f77b4")
ax2.set_ylabel("Sharpe", color="orange")
ax.set_title("Threshold sweep, total return + Sharpe vs decision threshold")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left"); ax2.legend(loc="upper right")
plt.tight_layout()
plt.savefig(out_dir / "2_threshold_sweep.png", dpi=110, bbox_inches="tight")
plt.close()


# ─── 3. Calibration plot ───────────────────────────────────────────────────
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
cal_pred = []
cal_actual = []
cal_n = []
for i in range(n_bins):
    in_bin = (preds_1d >= bin_edges[i]) & (preds_1d < bin_edges[i + 1])
    if in_bin.sum() < 5:
        cal_pred.append(None); cal_actual.append(None); cal_n.append(0)
        continue
    cal_pred.append(float(preds_1d[in_bin].mean()))
    cal_actual.append(float(y_test_1d[in_bin].mean()))
    cal_n.append(int(in_bin.sum()))

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
xs = [p for p in cal_pred if p is not None]
ys = [a for a, p in zip(cal_actual, cal_pred) if p is not None]
ns = [n for n, p in zip(cal_n, cal_pred) if p is not None]
ax.scatter(xs, ys, s=[max(40, n*1.5) for n in ns], alpha=0.6, color="#1f77b4")
for x, y, n in zip(xs, ys, ns):
    ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
ax.axhline(base_rate, color="g", linestyle=":", alpha=0.7, label=f"actual base rate = {base_rate:.3f}")
ax.set_xlabel("Mean predicted prob in bin")
ax.set_ylabel("Actual up rate in bin")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Calibration plot, predicted vs realized")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(out_dir / "3_calibration.png", dpi=110, bbox_inches="tight")
plt.close()


# ─── 4. Feature attribution on wrong-call days ─────────────────────────────
# A "wrong-call day" = decision bar where prediction was clearly bearish (< 0.45)
# but the realized next-day return was strongly positive (> +0.5%)
test_decision_idx = test_index[is_decision]
test_decision_preds = preds_series[is_decision]
test_decision_fwd = pd.Series(fwd_ret_1d, index=test_index).loc[test_decision_idx]

wrong = pd.DataFrame({
    "prediction": test_decision_preds,
    "fwd_return": test_decision_fwd,
})
wrong = wrong[(wrong["prediction"] < 0.45) & (wrong["fwd_return"] > 0.005)]
wrong = wrong.sort_values("fwd_return", ascending=False).head(5)

print("\n" + "=" * 80)
print(f"FEATURE ATTRIBUTION, top 5 wrong-call days (predicted < 0.45, market went UP > +0.5%)")
print("=" * 80)
if len(wrong) == 0:
    print("(no wrong-call days matching criteria)")
    attribution_summary = []
else:
    # Build a single-seed model for fast feature ablation
    single_model = models[0]
    feat_names = feat_cols
    attribution_summary = []
    for idx in wrong.index:
        # Position in te_part
        loc = te_part.index.get_loc(idx)
        x_row = X_te[loc:loc+1]
        baseline_logit = single_model(torch.from_numpy(x_row).to(DEVICE))
        baseline_prob = float(torch.sigmoid(baseline_logit)[0, h_idx].item())

        deltas = []
        for j, name in enumerate(feat_names):
            x_perturbed = x_row.copy()
            x_perturbed[0, j] = 0.0  # 0 in scaled space = training mean
            with torch.no_grad():
                logit_p = single_model(torch.from_numpy(x_perturbed).to(DEVICE))
                prob_p = float(torch.sigmoid(logit_p)[0, h_idx].item())
            deltas.append((name, baseline_prob - prob_p))   # positive = feature pushed prediction UP
        deltas.sort(key=lambda x: x[1])   # most negative first (feature pushed pred DOWN)
        bearish_features = deltas[:8]      # features that pushed pred down most

        attribution_summary.append({
            "date": str(idx)[:16],
            "baseline_pred": round(baseline_prob, 4),
            "actual_fwd_ret_pct": round(wrong.loc[idx, "fwd_return"] * 100, 3),
            "bearish_features": [{"feature": n, "delta": round(d, 4)} for n, d in bearish_features],
        })
        print(f"\n{str(idx)[:16]}  pred={baseline_prob:.3f}  fwd_return={wrong.loc[idx,'fwd_return']*100:+.2f}%")
        print("  features pushing prediction DOWN (Δ prob if feature ablated to training mean):")
        for name, d in bearish_features:
            bar = "█" * max(1, int(-d * 200))
            print(f"    {name:<28s}  Δ={d:+.4f}  {bar}")

# Save text summary
summary = {
    "config": {
        "train_period": [str(tr_part.index[0])[:16], str(tr_part.index[-1])[:16]],
        "val_period":   [str(val_part.index[0])[:16], str(val_part.index[-1])[:16]],
        "test_period":  [str(te_part.index[0])[:16], str(te_part.index[-1])[:16]],
        "n_features": len(feat_cols),
        "n_train_bars": len(tr_part),
        "n_test_bars":  len(te_part),
    },
    "test_set_metrics": {
        "auc_1d": round(test_auc, 4),
        "base_rate":     round(base_rate, 4),
        "mean_pred":     round(float(np.mean(preds_1d)), 4),
        "median_pred":   round(float(np.median(preds_1d)), 4),
        "frac_pred_above_05": round(float((preds_1d > 0.5).mean()), 4),
    },
    "threshold_sweep": threshold_results,
    "buy_hold_test_period": {
        "total_return_pct": round(bh_total * 100, 2),
        "sharpe": round(bh_sharpe, 3),
    },
    "calibration": [
        {"bin_center": round(bc, 2), "n": n, "pred": p, "actual": a}
        for bc, p, a, n in zip(bin_centers, cal_pred, cal_actual, cal_n)
    ],
    "wrong_call_attribution": attribution_summary,
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

print("\n" + "=" * 80)
print("KEY DIAGNOSTIC NUMBERS")
print("=" * 80)
print(f"  Test set base rate (actual UP days): {base_rate*100:.1f}%")
print(f"  Mean model prediction:                {float(np.mean(preds_1d))*100:.1f}%   "
      f"(should be ~equal to base rate if calibrated)")
print(f"  Median model prediction:              {float(np.median(preds_1d))*100:.1f}%")
print(f"  Fraction of predictions > 0.5:        {float((preds_1d > 0.5).mean())*100:.1f}%")
print(f"  Test AUC (ranking quality):           {test_auc:.4f}")
print(f"  → Calibration bias: model predicts {float(np.mean(preds_1d)) - base_rate:+.3f} vs actual rate")

print(f"\nArtifacts saved to {out_dir}")
print(f"  1_prediction_histogram.png , where do predictions sit?")
print(f"  2_threshold_sweep.png      , what threshold maximizes returns?")
print(f"  3_calibration.png          , when model says X%, does it happen X%?")
print(f"  summary.json               , full diagnostic data + feature attributions")
