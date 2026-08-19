"""
Yahoo data source, works today with zero IBKR setup.

Pulls daily OHLCV for SPY, VIX (index), and the VIX futures continuous proxy.
Writes to the same Parquet layout the IBKR puller uses, so the trainer is
source-agnostic.

CAVEAT: Yahoo's VIX *futures term structure* is incomplete. We can get spot
VIX (`^VIX`) and a few legs like `^VIX9D`, `^VXMT`, but per-month VX contracts
(VX1, VX2, ...) aren't reliably available on Yahoo. So when running on
Yahoo-only data the term-structure features fall back to a coarse 3-point
curve: short-dated (^VIX9D), spot (^VIX), medium-dated (^VXMT, ~6mo).
The IBKR puller fills in the proper monthly term structure.
"""
from __future__ import annotations
import os

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.environ.get("MARKETDATA_DIR", Path(__file__).resolve().parent / "marketdata"))


# Tickers and where they land in the Parquet layout.
YAHOO_SYMBOLS = {
    # ticker        instrument_type      stored_as
    "^VIX":         ("vix_index",        "VIX"),
    "^VIX9D":       ("vix_index",        "VIX9D"),     # 9-day VIX
    "^VIX6M":      ("vix_index",        "VXMT"),      # 6-month CBOE vol (^VXMT was delisted)
    # cross-asset
    "DX-Y.NYB":     ("dollar",           "DXY"),
}


def _write_parquet(df: pd.DataFrame, instrument_type: str, symbol: str,
                   timeframe: str = "1day") -> Path:
    """Match storage.py's layout: ~/marketdata/{type}/{symbol}/{tf}/{YYYY-MM}.parquet"""
    if df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df.index, utc=True)
    df = df.reset_index(drop=True)
    # Yahoo columns: Open High Low Close Adj Close Volume → lowercase
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    df = df[[c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]]
    df["symbol"] = symbol
    df["timeframe"] = timeframe

    df["_month"] = df["date"].dt.strftime("%Y-%m")
    last_path = None
    for month, chunk in df.groupby("_month"):
        chunk = chunk.drop(columns="_month")
        path = DATA_ROOT / instrument_type / symbol / timeframe / f"{month}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            merged = pd.concat([existing, chunk], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep="last")
            merged = merged.sort_values("date").reset_index(drop=True)
        else:
            merged = chunk.sort_values("date").reset_index(drop=True)
        merged.to_parquet(path, index=False)
        last_path = path
        logger.info("Wrote %d rows → %s", len(chunk), path)
    return last_path


def pull_all(years: int = 10) -> dict[str, int]:
    """Download `years` of daily history for every symbol in YAHOO_SYMBOLS."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=years * 366)
    results: dict[str, int] = {}
    for ticker, (instr_type, stored) in YAHOO_SYMBOLS.items():
        logger.info("Downloading %s (→ %s/%s)", ticker, instr_type, stored)
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            _write_parquet(df, instr_type, stored)
            results[ticker] = len(df)
        except Exception as e:
            logger.exception("Failed %s: %s", ticker, e)
            results[ticker] = -1
    return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    args = ap.parse_args()
    r = pull_all(years=args.years)
    print("\n=== Summary ===")
    for k, n in r.items():
        print(f"  {k}: {n} rows")
