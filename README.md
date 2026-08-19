# SPY Multi-Horizon Classifier

**A multi-horizon directional model for SPY, and an honest account of why daily direction is
close to unpredictable, with the walk-forward rigor and diagnostics to prove it rather than
p-hack around it.**

This is the research pipeline that predates and motivates the
[overnight gap-risk work](https://github.com/ascales23/overnight-gap-risk): before concluding
that *risk* is the predictable edge, I tested *return* direction properly, across horizons and
model families, with leak checks and out-of-sample walk-forward validation. The answer was a
clean negative, and documenting that honestly is the point.

> Research code. Results are walk-forward out-of-sample. No trading, no real capital.

---

## The finding

Daily SPY direction is barely more predictable than a coin flip, and the models that *look*
best in-sample fall apart out-of-sample or are miscalibrated. Reproduced locally from the
bundled data (expanding-window walk-forward, XGBoost, OOS from 2016):

| Horizon | OOS AUC        |
|--------:|:--------------:|
| 1 day   | 0.504 ± 0.027  |
| 5 day   | 0.539 ± 0.052  |
| 21 day  | 0.507 ± 0.118  |

56 engineered features, 3,774 daily rows. The point is not the tiny edge; it is that a rigorous
setup refuses to manufacture one.

### Diagnostics: the honest-negative evidence (`runs/`)

The included run artifacts (from the GPU box) are the actual receipts:

- **The MLP is severely miscalibrated.** Against a ~62% up base rate it predicts a mean
  probability near 5%, with a single feature (a Treasury yield) dominating its output. High
  in-sample AUC, useless probabilities.
- **XGBoost honestly gives up.** On the same data its best boosting iteration is 1, it finds no
  signal, and at any sane threshold it is 97% long, so the "strategy" is buy-and-hold by
  accident.
- **The backtest confirms it.** The NN trading on these signals returned ~0.7% while buy-and-hold
  returned ~12.9% over the test window; it simply refused to be long during a rally.
- **A label leak was found and removed.** The Ichimoku `chikou` span (`close.shift(-26)`) leaked
  the future and inflated a holdout AUC to 0.79; deleting it collapsed the apparent signal, which
  is how you know it was leakage, not alpha.

This is what pointed the research toward predicting *risk* instead of *return*.

## Quick start (daily pipeline, bundled data)

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas pyarrow scikit-learn xgboost matplotlib
python features.py --symbol SPY        # build + inspect the 56-feature matrix
python walk_forward.py --symbol SPY    # expanding walk-forward CV (see note on torch below)
```

The bundled `data/marketdata/` (SPY + cross-asset + sectors + VIX complex + VIX-futures term
structure + FRED macro, daily) makes the daily pipeline self-contained. To point at your own
store instead, set `MARKETDATA_DIR=/path/to/marketdata`.

**Torch / MLP note.** `walk_forward.py` and `train.py` also fit a PyTorch MLP and were built for
a CUDA box (Python 3.11–3.13). The XGBoost path is the one verified in the table above; the MLP
head needs a stable torch build (torch's brand-new 3.14 wheels are not). `pip install torch` for
that head, or run the XGBoost analysis without it.

## Repository layout

```
spy-multihorizon-ml/
├── features.py            # 56-feature daily matrix: price/MA/RSI/Ichimoku, realized vol + VIX +
│                          #   VRP, VIX-futures term structure, FRED macro, cross-asset, sectors
├── walk_forward.py        # expanding-window walk-forward CV, XGBoost + MLP per horizon
├── train.py               # single-run training (XGBoost baseline + MLP), writes runs/ metrics
├── dataset.py             # time-series splits + StandardScaler (fit on train only)
├── diagnose.py            # MLP diagnostics: calibration, threshold sweep, feature perturbation
├── diagnose_xgb.py        # XGBoost diagnostics: calibration, feature importance
├── backtest.py            # daily once-per-day backtest from the model signals
├── sanity.py              # leakage + multicollinearity + shuffled-label sanity checks
├── models/                # MultiHorizonMLP and an LSTM variant
├── data/
│   ├── yahoo.py           # public-data fetcher into the marketdata layout
│   ├── resample.py        # 15m -> 1h/4h/1D resampling
│   └── marketdata/        # bundled daily parquet store (the self-contained input)
├── features_intraday.py + backtest_intraday.py + train_lstm.py
│                          #   intraday / sequence track (needs a 15-min store, NOT bundled)
├── runs/                  # example outputs: metrics JSON + diagnostic PNGs (the receipts)
└── requirements.txt · LICENSE · README.md
```

The intraday and LSTM track is included for completeness but is not runnable from this repo alone;
it needs a 15-minute bar store that is not bundled here.

## Method notes

- **No lookahead.** Expanding-window walk-forward; StandardScaler is fit on train folds only;
  Ichimoku spans are lagged and the leaking `chikou` is deliberately excluded.
- **Two label schemes.** Binary direction, and a Sharpe-ratio-gated ternary label that abstains
  on low-conviction (chop) days, with horizon-scaled vol lookbacks.
- **Model choice is earned, not assumed.** XGBoost baseline alongside the MLP; on this small
  tabular data the trees are at least as good and far better calibrated.

## License

MIT. See [LICENSE](LICENSE).

Author: **Alexander Scales**
