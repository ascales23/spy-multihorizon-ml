"""
Feature engineering on top of the Parquet store.

Reads ~/marketdata/, returns a single feature DataFrame indexed by date,
ready for ML. Source-agnostic, works whether bars came from Yahoo or IBKR.

Features:
  Price/MA:           log_return_1d, ma_15, ma_30, ma_15_minus_30_pct
  Momentum:           rsi_14, return_5d, return_21d, return_63d
  Ichimoku:           tenkan, kijun, senkou_a, senkou_b,
                      close_vs_cloud (-1/0/+1), tk_cross (-1/0/+1)
  Volatility:         realized_vol_20, vix, vrp (vix - realized_vol_20)
  VIX term struct:    vx1..vx8, vx_level, vx_slope, vx_curvature, vx_contango_ratio
  Macro (FRED):       yield_spread_10y2y, yield_spread_10y3m, treasury_10y,
                      real_yield_10y, credit_bbb_oas, credit_hy_oas,
                      credit_hy_minus_bbb, nfci_weekly, dollar_twi
  Cross-asset:        hyg_ret5, tlt_ret5, gld_ret5, dxy_ret5,
                      eem_ret5, uup_ret5
  Sector rotation:    xle_vs_spy_5d, xlk_vs_spy_5d, xlf_vs_spy_5d,
                      xlu_vs_spy_5d, xly_vs_spy_5d, xlv_vs_spy_5d
  Calendar:           dow, dom, days_to_month_end
"""
from __future__ import annotations
import os

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(os.environ.get("MARKETDATA_DIR", Path(__file__).resolve().parent / "data" / "marketdata"))
logger = logging.getLogger(__name__)


# ── Loaders ──────────────────────────────────────────────────────────────────
def _load_symbol(instrument_type: str, symbol: str,
                 timeframe: str = "1day") -> pd.DataFrame:
    pattern = str(DATA_ROOT / instrument_type / symbol / timeframe / "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning("No parquet for %s/%s/%s", instrument_type, symbol, timeframe)
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _load_close(instrument_type: str, symbol: str,
                target_index: pd.Index) -> pd.Series:
    """Load close prices for a symbol and align to target_index."""
    df = _load_symbol(instrument_type, symbol)
    if df.empty:
        return pd.Series(np.nan, index=target_index)
    return df.set_index("date")["close"].reindex(target_index, method="ffill")


def _load_macro_series(alias: str, target_index: pd.Index) -> pd.Series:
    """Load a FRED macro series (stored under marketdata/macro/<alias>/1day/)."""
    pattern = str(DATA_ROOT / "macro" / alias / "1day" / "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning("No macro data for %s", alias)
        return pd.Series(np.nan, index=target_index)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    col = "value" if "value" in df.columns else "close"
    return df.set_index("date")[col].reindex(target_index, method="ffill")


# ── Indicators ───────────────────────────────────────────────────────────────
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun  = (high.rolling(26).max() + low.rolling(26).min()) / 2
    # Senkou A/B are derived from data t-26 bars back → no future leak.
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    # Chikou (close.shift(-26)) leaks the future label, deliberately excluded.

    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    close_vs_cloud = np.where(close > cloud_top, 1,
                              np.where(close < cloud_bot, -1, 0))

    diff = tenkan - kijun
    prev = diff.shift(1)
    tk_cross = np.where((prev < 0) & (diff > 0), 1,
                np.where((prev > 0) & (diff < 0), -1, 0))

    return pd.DataFrame({
        "tenkan": tenkan, "kijun": kijun,
        "senkou_a": senkou_a, "senkou_b": senkou_b,
        "close_vs_cloud": close_vs_cloud,
        "tk_cross": tk_cross,
    }, index=df.index)


# ── VIX term-structure features ───────────────────────────────────────────────
def _vix_term_structure(target_dates: pd.Index) -> pd.DataFrame:
    """
    Rank-based VX term structure: VX1 = front month, VX2 = next, etc.
    Falls back to spot VIX indices when futures not available.
    """
    target_dates = pd.to_datetime(target_dates, utc=True)
    out = pd.DataFrame(index=target_dates)

    vx_dir = DATA_ROOT / "vix_futures"
    monthly_files = sorted(vx_dir.glob("VX*/1day/*.parquet")) if vx_dir.exists() else []

    if monthly_files:
        frames = []
        for f in monthly_files:
            df = pd.read_parquet(f, columns=["date", "close", "expiry"])
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["symbol"] = f.parent.parent.name
            frames.append(df)
        vx = pd.concat(frames, ignore_index=True)
        vx = vx.drop_duplicates(["date", "symbol"]).sort_values("date")

        vx["expiry_dt"] = pd.to_datetime(vx["expiry"] + "-01", utc=True)
        vx["months_diff"] = ((vx["expiry_dt"].dt.year - vx["date"].dt.year) * 12
                             + (vx["expiry_dt"].dt.month - vx["date"].dt.month))
        vx = vx[vx["months_diff"] >= 1]
        vx["rank"] = vx.groupby("date")["months_diff"].rank(method="dense").astype(int)

        wide = (vx.pivot_table(index="date", columns="rank", values="close", aggfunc="last")
                .sort_index())
        keep_ranks = [r for r in wide.columns if r <= 8]
        wide = wide[keep_ranks]
        wide.columns = [f"vx{r}" for r in keep_ranks]
        wide = wide.reindex(target_dates, method="ffill")
        out = out.join(wide)

        if "vx1" in wide.columns:
            front = wide["vx1"]
            back = wide.apply(
                lambda row: row.dropna().iloc[-1] if row.notna().any() else np.nan, axis=1
            )
            out["vx_level"] = wide.mean(axis=1)
            out["vx_slope"] = back - front
            out["vx_contango_ratio"] = front / back
            if "vx4" in wide.columns:
                out["vx_curvature"] = (front + back) / 2 - wide["vx4"]
    else:
        v9 = _load_symbol("vix_index", "VIX9D").set_index("date")["close"]
        vs = _load_symbol("vix_index", "VIX").set_index("date")["close"]
        vm = _load_symbol("vix_index", "VXMT").set_index("date")["close"]
        for s, name in [(v9, "vx_9d"), (vs, "vx_spot"), (vm, "vx_mt")]:
            if s is not None and not s.empty:
                out[name] = s.reindex(target_dates, method="ffill")
        if "vx_9d" in out.columns and "vx_mt" in out.columns:
            out["vx_level"] = out[["vx_9d", "vx_spot", "vx_mt"]].mean(axis=1)
            out["vx_slope"] = out["vx_mt"] - out["vx_9d"]
            out["vx_contango_ratio"] = out["vx_9d"] / out["vx_mt"]
            out["vx_curvature"] = (out["vx_9d"] + out["vx_mt"]) / 2 - out["vx_spot"]

    return out


# ── Main build ────────────────────────────────────────────────────────────────
def build(symbol: str = "SPY") -> pd.DataFrame:
    """Build the full feature matrix for `symbol`. Returns date-indexed DataFrame."""
    px = _load_symbol("equities", symbol).set_index("date")
    if px.empty:
        raise FileNotFoundError(f"No data for {symbol} in ~/marketdata/equities/")

    f = pd.DataFrame(index=px.index)
    f["close"] = px["close"]
    f["log_return_1d"] = np.log(px["close"] / px["close"].shift(1))
    f["ma_15"] = px["close"].rolling(15).mean()
    f["ma_30"] = px["close"].rolling(30).mean()
    f["ma_15_minus_30_pct"] = (f["ma_15"] - f["ma_30"]) / f["ma_30"]

    f["rsi_14"] = _rsi(px["close"], 14)
    f["return_5d"]  = px["close"].pct_change(5)
    f["return_21d"] = px["close"].pct_change(21)
    f["return_63d"] = px["close"].pct_change(63)

    ich = _ichimoku(px)
    for c in ich.columns:
        f[c] = ich[c]

    f["realized_vol_20"] = f["log_return_1d"].rolling(20).std() * np.sqrt(252) * 100

    vix = _load_symbol("vix_index", "VIX").set_index("date")["close"]
    f["vix"] = vix.reindex(f.index, method="ffill")
    f["vrp"] = f["vix"] - f["realized_vol_20"]

    ts = _vix_term_structure(f.index)
    for c in ts.columns:
        f[c] = ts[c]

    # ── FRED macro ───────────────────────────────────────────────────────────
    macro_map = {
        "spread_10y_2y":   "yield_spread_10y2y",
        "spread_10y_3m":   "yield_spread_10y3m",
        "treasury_10y":    "treasury_10y",
        "treasury_real_10y": "real_yield_10y",
        "credit_bbb_oas":  "credit_bbb_oas",
        "credit_hy_oas":   "credit_hy_oas",
        "nfci_weekly":     "nfci",
        "dollar_twi_broad": "dollar_twi",
    }
    for alias, col_name in macro_map.items():
        s = _load_macro_series(alias, f.index)
        f[col_name] = s

    # Derived macro: HY-BBB spread compression/widening signal
    if "credit_hy_oas" in f.columns and "credit_bbb_oas" in f.columns:
        f["credit_hy_minus_bbb"] = f["credit_hy_oas"] - f["credit_bbb_oas"]

    # Yield curve inversion flag
    if "yield_spread_10y2y" in f.columns:
        f["curve_inverted"] = (f["yield_spread_10y2y"] < 0).astype(float)

    # ── Cross-asset 5d returns ────────────────────────────────────────────────
    cross_asset = [
        ("equities", "HYG", "hyg_ret5"),
        ("equities", "TLT", "tlt_ret5"),
        ("equities", "GLD", "gld_ret5"),
        ("equities", "UUP", "uup_ret5"),
        ("equities", "EEM", "eem_ret5"),
        ("equities", "IWM", "iwm_ret5"),
        ("equities", "SHY", "shy_ret5"),
    ]
    for instr_type, sym, col_name in cross_asset:
        s = _load_close(instr_type, sym, f.index)
        f[col_name] = s.pct_change(5)

    # ── Sector rotation (relative to SPY) ────────────────────────────────────
    spy_ret5 = f["return_5d"]
    sectors = ["XLE", "XLK", "XLF", "XLU", "XLY", "XLV", "XLI", "XLB"]
    for sec in sectors:
        s = _load_close("equities", sec, f.index)
        if s.notna().sum() > 50:
            f[f"{sec.lower()}_vs_spy_5d"] = s.pct_change(5) - spy_ret5

    # ── Calendar ─────────────────────────────────────────────────────────────
    idx = f.index
    f["dow"] = idx.dayofweek
    f["dom"] = idx.day
    f["days_to_month_end"] = idx.daysinmonth - idx.day

    return f


# ── Label construction ────────────────────────────────────────────────────────
def add_sr_labels(
    f: pd.DataFrame,
    horizons: list[int] = (1, 5, 21),
    threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Sharpe-ratio-gated ternary labels per horizon.

    For horizon h at time t:
      fwd_return = (close[t+h] - close[t]) / close[t]
      vol        = rolling std of log returns over vol_lookback[h], annualized
      sr         = fwd_return / vol

      label = +1  if sr > +threshold   (strong risk-adjusted up)
              -1  if sr < -threshold   (strong risk-adjusted down)
               0  otherwise            (noise / chop, abstain)

    Vol lookbacks are horizon-scaled so the Sharpe is dimensionally consistent:
      1d  → 21-day realized vol  (1-month vol)
      5d  → 63-day realized vol  (1-quarter vol)
      21d → 126-day realized vol (6-month vol)

    Stores fwd_return_{h}d, sr_{h}d, and sr_label_{h}d columns.
    """
    f = f.copy()
    vol_lookback = {1: 21, 5: 63, 21: 126}
    log_ret = np.log(f["close"] / f["close"].shift(1))

    for h in horizons:
        fwd_return = f["close"].shift(-h) / f["close"] - 1
        lookback = vol_lookback.get(h, max(h * 3, 21))
        vol = log_ret.rolling(lookback).std() * np.sqrt(252)
        sr = fwd_return / vol.replace(0, np.nan)

        f[f"fwd_return_{h}d"] = fwd_return
        f[f"sr_{h}d"] = sr
        f[f"sr_label_{h}d"] = np.where(
            sr > threshold, 1, np.where(sr < -threshold, -1, 0)
        ).astype(np.int8)

    return f


def add_labels(f: pd.DataFrame, horizons: list[int] = (1, 5, 21)) -> pd.DataFrame:
    """Binary direction labels. Kept for backward compatibility."""
    f = f.copy()
    for h in horizons:
        fwd_return = f["close"].shift(-h) / f["close"] - 1
        f[f"fwd_return_{h}d"] = fwd_return
        f[f"label_{h}d"] = (fwd_return > 0).astype(int)
    return f


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()
    f = build(args.symbol)
    f = add_sr_labels(f, threshold=args.threshold)
    print(f.tail(5).to_string())
    print(f"\nShape: {f.shape}")
    print(f"NaN per col (top 10):\n{f.isna().sum().sort_values(ascending=False).head(10)}")
    for h in [1, 5, 21]:
        col = f[f"sr_label_{h}d"].dropna()
        print(f"\n{h}d label dist: {col.value_counts().to_dict()} "
              f"(abstain rate: {(col==0).mean():.1%})")
