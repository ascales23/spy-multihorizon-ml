"""
Bar resampling utilities. Aggregate finer-resolution bars to coarser.
Used at feature-build time so we don't duplicate data on disk.
"""
from __future__ import annotations
import os

import glob
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(os.environ.get("MARKETDATA_DIR", Path(__file__).resolve().parent / "marketdata"))


def aggregate(df: pd.DataFrame, rule: str = "1h") -> pd.DataFrame:
    """OHLCV resampling. `rule` is a pandas offset string ('1h', '4h', '1D').
    Volume sums, OHLC standard agg. Volume-weighted average if present."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    out = df.resample(rule, label="right", closed="right").agg(agg)
    if "average" in df.columns and "volume" in df.columns:
        num = (df["average"] * df["volume"]).resample(rule, label="right", closed="right").sum()
        den = df["volume"].resample(rule, label="right", closed="right").sum().replace(0, pd.NA)
        out["average"] = (num / den)
    out = out.dropna(subset=["close"]).reset_index()
    return out


def load_resampled(instrument_type: str, symbol: str, *,
                   base_timeframe: str = "15mins",
                   target_rule: str = "1h") -> pd.DataFrame:
    """Load base bars from disk and resample to target_rule."""
    pattern = str(DATA_ROOT / instrument_type / symbol / base_timeframe / "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    if target_rule.lower() in ("none", "raw", base_timeframe):
        return df
    return aggregate(df, rule=target_rule)
