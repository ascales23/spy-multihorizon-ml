"""
Feature engineering for INTRADAY (15-min default) bars.

Reads from ~/marketdata/, joins:
  - SPY intraday OHLC + derived (returns, MAs, vol, RSI)
  - Cross-asset intraday returns (HYG, TLT, UUP, GLD, etc.)
  - Daily VIX spot + VRP (forward-filled to intraday)
  - Daily VIX futures term structure (forward-filled; or intraday if available)
  - FRED macro daily series (forward-filled)
  - Time-of-day / day-of-week / day-of-month

Labels are next-N-bar binary direction (default: 1, 4, 26 bars = 15m / 1h / 1d).
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
def _load_bars(instrument_type: str, symbol: str,
               timeframe: str = "15mins") -> pd.DataFrame:
    pattern = str(DATA_ROOT / instrument_type / symbol / timeframe / "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _load_macro(alias: str) -> pd.Series:
    df = _load_bars("macro", alias, "1day")
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("date")["value"]


# ── Indicators ───────────────────────────────────────────────────────────────
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ── VIX term structure (forward-fill from daily; reuse features.py logic) ──
def _vix_term_structure(target_dates: pd.Index) -> pd.DataFrame:
    """Same rank-based pivot as features.py, but more tolerant of intraday index."""
    target_dates = pd.to_datetime(target_dates, utc=True)
    out = pd.DataFrame(index=target_dates)

    vx_dir = DATA_ROOT / "vix_futures"
    # Prefer intraday VX if present, fall back to daily
    intraday_files = sorted(vx_dir.glob("VX*/15mins/*.parquet")) if vx_dir.exists() else []
    if intraday_files:
        files = intraday_files
        tf_used = "15mins"
    else:
        files = sorted(vx_dir.glob("VX*/1day/*.parquet")) if vx_dir.exists() else []
        tf_used = "1day"
    if not files:
        return out

    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=["date", "close", "expiry"])
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["symbol"] = f.parent.parent.name
        frames.append(df)
    vx = pd.concat(frames, ignore_index=True)
    vx = vx.drop_duplicates(["date", "symbol"]).sort_values("date")

    # For intraday: collapse to daily snapshots first (use last close of each day per contract)
    if tf_used == "15mins":
        vx["day"] = vx["date"].dt.date
        vx = (vx.sort_values("date")
                .groupby(["symbol", "day"], as_index=False)
                .agg({"date": "last", "close": "last", "expiry": "last"}))

    vx["expiry_dt"] = pd.to_datetime(vx["expiry"] + "-01", utc=True)
    vx["months_diff"] = ((vx["expiry_dt"].dt.year - vx["date"].dt.year) * 12
                         + (vx["expiry_dt"].dt.month - vx["date"].dt.month))
    vx = vx[vx["months_diff"] >= 1]
    vx["rank"] = vx.groupby("date")["months_diff"].rank(method="dense").astype(int)

    wide = (vx.pivot_table(index="date", columns="rank", values="close", aggfunc="last")
              .sort_index())
    keep = [r for r in wide.columns if r <= 8]
    wide = wide[keep]
    wide.columns = [f"vx{r}" for r in keep]
    wide = wide.reindex(target_dates, method="ffill")
    out = out.join(wide)

    if "vx1" in wide.columns:
        front = wide["vx1"]
        back = wide.apply(lambda row: row.dropna().iloc[-1] if row.notna().any() else np.nan, axis=1)
        out["vx_level"] = wide.mean(axis=1)
        out["vx_slope"] = back - front
        out["vx_contango_ratio"] = front / back
        if "vx4" in wide.columns:
            out["vx_curvature"] = (front + back) / 2 - wide["vx4"]
    return out


# ── Main feature build ───────────────────────────────────────────────────────
# Bar counts at 15-min frequency
# 1 day RTH = 26 bars, 1 week = ~130, 1 month = ~545
BAR_PERIODS = {
    "1d":  26,    # one trading day
    "1w":  130,   # one trading week
    "1m":  545,   # ~21 trading days
}


def build(symbol: str = "SPY", *, timeframe: str = "15mins") -> pd.DataFrame:
    """Build intraday feature matrix for `symbol` at `timeframe`."""
    px = _load_bars("equities", symbol, timeframe)
    if px.empty:
        raise FileNotFoundError(f"No data at equities/{symbol}/{timeframe}")
    px = px.set_index("date").sort_index()

    f = pd.DataFrame(index=px.index)
    f["close"] = px["close"]

    # ── Returns at multiple horizons ──
    f["log_return_1b"] = np.log(px["close"] / px["close"].shift(1))
    for label, n in BAR_PERIODS.items():
        f[f"return_{label}"] = px["close"].pct_change(n)
        f[f"ma_{label}"] = px["close"].rolling(n).mean()
    f["ma_1d_vs_1w_pct"] = (f["ma_1d"] - f["ma_1w"]) / f["ma_1w"]
    f["ma_1w_vs_1m_pct"] = (f["ma_1w"] - f["ma_1m"]) / f["ma_1m"]

    # ── Volume ──
    if "volume" in px.columns:
        f["volume"] = px["volume"]
        f["volume_ma_1d"] = px["volume"].rolling(BAR_PERIODS["1d"]).mean()
        f["volume_ratio"] = px["volume"] / f["volume_ma_1d"].replace(0, np.nan)

    # ── RSI ──
    f["rsi_14"] = _rsi(px["close"], 14)

    # ── Realized vol, bar-count based ──
    n_per_year = 252 * BAR_PERIODS["1d"]   # bars per year
    f["realized_vol_1d"] = f["log_return_1b"].rolling(BAR_PERIODS["1d"]).std() * np.sqrt(n_per_year)
    f["realized_vol_1w"] = f["log_return_1b"].rolling(BAR_PERIODS["1w"]).std() * np.sqrt(n_per_year)

    # ── Spot VIX (daily, ffill) ──
    vix = _load_bars("vix_index", "VIX", timeframe="1day")
    if not vix.empty:
        vs = vix.set_index("date")["close"]
        f["vix"] = vs.reindex(f.index, method="ffill")
        f["vrp"] = f["vix"] - f["realized_vol_1d"]

    # ── VIX term structure (daily or intraday ffilled) ──
    ts = _vix_term_structure(f.index)
    for c in ts.columns:
        f[c] = ts[c]

    # ── FRED macro (daily, ffill) ──
    macro_aliases = [
        "treasury_10y", "treasury_2y", "treasury_3m",
        "spread_10y_2y", "spread_10y_3m", "treasury_real_10y",
        "credit_bbb_oas", "credit_hy_oas", "dollar_twi_broad",
    ]
    for alias in macro_aliases:
        s = _load_macro(alias)
        if not s.empty:
            f[f"macro_{alias}"] = s.reindex(f.index, method="ffill")

    # ── Cross-asset returns at 1d (intraday) ──
    cross = [("HYG", "hyg"), ("TLT", "tlt"), ("UUP", "dxy"), ("GLD", "gld"),
             ("XLF", "xlf"), ("XLK", "xlk"), ("XLE", "xle"), ("QQQ", "qqq"),
             ("IWM", "iwm"), ("EEM", "eem")]
    for sym, alias in cross:
        df_c = _load_bars("equities", sym, timeframe)
        if df_c.empty:
            continue
        c = df_c.set_index("date")["close"]
        # Forward-fill onto SPY's index so cross-asset gaps don't drop SPY rows
        c_aligned = c.reindex(f.index, method="ffill")
        f[f"{alias}_return_1d"] = c_aligned.pct_change(BAR_PERIODS["1d"])
        f[f"{alias}_return_1w"] = c_aligned.pct_change(BAR_PERIODS["1w"])

    # ── Calendar / time-of-day ──
    idx_local = f.index.tz_convert("America/New_York")
    f["minute_of_day"] = idx_local.hour * 60 + idx_local.minute
    f["bar_of_day"] = ((f["minute_of_day"] - 9 * 60 - 30) // 15).clip(lower=0, upper=25)
    f["dow"] = idx_local.dayofweek
    f["dom"] = idx_local.day
    f["days_to_month_end"] = idx_local.daysinmonth - idx_local.day

    return f


# ── Labels ───────────────────────────────────────────────────────────────────
def add_labels(f: pd.DataFrame, horizons_bars=(1, 4, 26)) -> pd.DataFrame:
    """Forward-N-bar binary up labels. 1b=15min, 4b=1h, 26b=1d."""
    f = f.copy()
    for h in horizons_bars:
        fwd = f["close"].shift(-h) / f["close"] - 1
        f[f"fwd_return_{h}b"] = fwd
        f[f"label_{h}b"] = (fwd > 0).astype(int)
    return f


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--timeframe", default="15mins")
    args = ap.parse_args()
    f = build(args.symbol, timeframe=args.timeframe)
    f = add_labels(f)
    print(f"\nShape: {f.shape}")
    print(f"Date range: {f.index.min()} → {f.index.max()}")
    print(f"\nTail:")
    print(f.tail(5)[["close", "log_return_1b", "rsi_14", "vix", "vx_slope",
                     "macro_spread_10y_2y", "minute_of_day", "label_26b"]].to_string())
    print(f"\nNaN per col (top 15):")
    print(f.isna().sum().sort_values(ascending=False).head(15))
    print(f"\nRows with NO NaN: {(~f.isna().any(axis=1)).sum()} / {len(f)}")
