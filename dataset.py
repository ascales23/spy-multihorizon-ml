"""
Dataset construction for the multi-scale, multi-horizon trading model.

Supports two modes:
  1. Tabular (flat), backward-compatible with MLP/XGBoost baseline
  2. Multi-window sequences, for the LSTM model:
       1D window  : single bar feature vector (no sequence)
       1W window  : last 5 bars as a sequence
       1M window  : last 21 bars as a sequence

Walk-forward cross-validation is provided by WalkForwardSplitter,
which generates rolling expanding-train / fixed-val windows. This is
the correct approach for time-series, no random shuffling, no leakage.

Label convention (ternary SR-gated):
  -1 → short / risk-adjusted down move
   0 → abstain / chop
  +1 → long / risk-adjusted up move

  Mapped to class indices 0/1/2 for PyTorch CrossEntropyLoss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# Columns excluded from the feature matrix
LABEL_COL_PREFIXES = ("label_", "fwd_return_", "sr_label_", "sr_")
DROP_COLS = ("close",)

# Window sizes in trading days
WIN_1D = 1
WIN_1W = 5
WIN_1M = 21


# ── Helpers ───────────────────────────────────────────────────────────────────
def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in DROP_COLS
        and not any(c.startswith(p) for p in LABEL_COL_PREFIXES)
    ]


def _label_cols(df: pd.DataFrame, horizons: list[int], ternary: bool) -> list[str]:
    prefix = "sr_label_" if ternary else "label_"
    return [f"{prefix}{h}d" for h in horizons]


# ── Tabular split (backward-compatible) ──────────────────────────────────────
@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val:   np.ndarray
    y_val:   np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray
    feature_cols: list[str]
    label_cols:   list[str]
    scaler: StandardScaler
    train_idx: pd.Index
    val_idx:   pd.Index
    test_idx:  pd.Index
    _fwd_val:  Optional[np.ndarray] = None
    _fwd_test: Optional[np.ndarray] = None


def make_splits(
    df: pd.DataFrame,
    horizons: list[int] = (1, 5, 21),
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    ternary: bool = True,
) -> Splits:
    """Static train/val/test split. Use WalkForwardSplitter for proper CV."""
    df = df.copy()
    feat_cols = _feature_cols(df)
    lbl_cols  = _label_cols(df, horizons, ternary)
    fwd_cols  = [f"fwd_return_{h}d" for h in horizons]

    df = df.dropna(subset=lbl_cols + feat_cols)
    n = len(df)
    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)
    n_train = n - n_val - n_test
    if n_train < 100:
        raise ValueError(f"Only {n_train} training rows after dropna, need more history")

    train = df.iloc[:n_train]
    val   = df.iloc[n_train:n_train + n_val]
    test  = df.iloc[n_train + n_val:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train[feat_cols].values).astype(np.float32)
    X_val   = scaler.transform(val[feat_cols].values).astype(np.float32)
    X_test  = scaler.transform(test[feat_cols].values).astype(np.float32)

    def _labels(split):
        if ternary:
            # Map -1/0/+1 → class index 0/1/2
            return (split[lbl_cols].values.astype(np.int64) + 1)
        return split[lbl_cols].values.astype(np.float32)

    return Splits(
        X_train=X_train, y_train=_labels(train),
        X_val=X_val,     y_val=_labels(val),
        X_test=X_test,   y_test=_labels(test),
        feature_cols=feat_cols,
        label_cols=lbl_cols,
        scaler=scaler,
        train_idx=train.index,
        val_idx=val.index,
        test_idx=test.index,
        _fwd_val=val[fwd_cols].values if all(c in val for c in fwd_cols) else None,
        _fwd_test=test[fwd_cols].values if all(c in test for c in fwd_cols) else None,
    )


# ── Walk-forward CV ───────────────────────────────────────────────────────────
@dataclass
class WalkForwardFold:
    fold: int
    train_idx: pd.Index
    val_idx: pd.Index
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    feature_cols: list[str]
    label_cols: list[str]
    scaler: StandardScaler
    fwd_val: Optional[np.ndarray] = None


class WalkForwardSplitter:
    """
    Expanding-window walk-forward CV for time series.

    Parameters
    ----------
    min_train_bars : int
        Minimum number of bars in the initial training window.
    val_bars : int
        Fixed number of bars in each validation window.
    step_bars : int
        Number of bars to advance the window each fold.
        Defaults to val_bars (non-overlapping validation windows).
    """
    def __init__(
        self,
        min_train_bars: int = 500,
        val_bars: int = 63,
        step_bars: int | None = None,
    ):
        self.min_train_bars = min_train_bars
        self.val_bars = val_bars
        self.step_bars = step_bars or val_bars

    def split(
        self,
        df: pd.DataFrame,
        horizons: list[int] = (1, 5, 21),
        ternary: bool = True,
    ) -> Generator[WalkForwardFold, None, None]:
        feat_cols = _feature_cols(df)
        lbl_cols  = _label_cols(df, horizons, ternary)
        fwd_cols  = [f"fwd_return_{h}d" for h in horizons]

        df_clean = df.dropna(subset=feat_cols + lbl_cols).copy()
        n = len(df_clean)

        fold = 0
        train_end = self.min_train_bars
        while train_end + self.val_bars <= n:
            val_end = min(train_end + self.val_bars, n)
            train_slice = df_clean.iloc[:train_end]
            val_slice   = df_clean.iloc[train_end:val_end]

            if len(val_slice) == 0:
                break

            scaler  = StandardScaler()
            X_train = scaler.fit_transform(train_slice[feat_cols].values).astype(np.float32)
            X_val   = scaler.transform(val_slice[feat_cols].values).astype(np.float32)

            def _labels(split):
                raw = split[lbl_cols].values.astype(np.int64)
                return raw + 1 if ternary else raw.astype(np.float32)

            yield WalkForwardFold(
                fold=fold,
                train_idx=train_slice.index,
                val_idx=val_slice.index,
                X_train=X_train,
                y_train=_labels(train_slice),
                X_val=X_val,
                y_val=_labels(val_slice),
                feature_cols=feat_cols,
                label_cols=lbl_cols,
                scaler=scaler,
                fwd_val=val_slice[fwd_cols].values
                    if all(c in val_slice for c in fwd_cols) else None,
            )
            fold += 1
            train_end += self.step_bars


# ── Multi-window sequence dataset ─────────────────────────────────────────────
@dataclass
class SequenceArrays:
    """Raw arrays for multi-scale LSTM. Each row is one sample."""
    x_1d: np.ndarray   # (N, n_features)          , single-bar snapshot
    x_1w: np.ndarray   # (N, WIN_1W, n_features)  , 5-bar sequence
    x_1m: np.ndarray   # (N, WIN_1M, n_features)  , 21-bar sequence
    y: np.ndarray      # (N, n_horizons)            , class indices (ternary) or floats
    dates: pd.Index
    feature_cols: list[str]
    label_cols: list[str]


def make_sequence_arrays(
    df: pd.DataFrame,
    horizons: list[int] = (1, 5, 21),
    ternary: bool = True,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[SequenceArrays, StandardScaler]:
    """
    Build multi-window sequence arrays from a feature DataFrame.

    Each sample at index i uses features[i-WIN_1M+1 : i+1] as context.
    The minimum valid index is WIN_1M - 1 (bars before that lack enough history).

    Returns (SequenceArrays, fitted_scaler). Pass a pre-fitted scaler
    (fit_scaler=False) for val/test to avoid leakage.
    """
    feat_cols = _feature_cols(df)
    lbl_cols  = _label_cols(df, horizons, ternary)
    fwd_cols  = [f"fwd_return_{h}d" for h in horizons]

    df = df.dropna(subset=feat_cols + lbl_cols).reset_index(drop=False)
    n = len(df)
    X_raw = df[feat_cols].values.astype(np.float32)

    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
    else:
        X_scaled = scaler.transform(X_raw).astype(np.float32)

    # Build sliding windows; only valid from index WIN_1M - 1 onward
    start = WIN_1M - 1
    valid_n = n - start

    x_1d = X_scaled[start:]                                       # (N, F)
    x_1w = np.stack([X_scaled[i - WIN_1W + 1: i + 1]             # (N, 5, F)
                     for i in range(start, n)])
    x_1m = np.stack([X_scaled[i - WIN_1M + 1: i + 1]             # (N, 21, F)
                     for i in range(start, n)])

    raw_labels = df.iloc[start:][lbl_cols].values.astype(np.int64)
    y = (raw_labels + 1) if ternary else raw_labels.astype(np.float32)

    dates = df.iloc[start:].set_index("date").index

    return SequenceArrays(
        x_1d=x_1d, x_1w=x_1w, x_1m=x_1m,
        y=y, dates=dates,
        feature_cols=feat_cols, label_cols=lbl_cols,
    ), scaler


def split_sequence_arrays(
    arr: SequenceArrays,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict[str, SequenceArrays]:
    """Time-ordered split of SequenceArrays into train/val/test."""
    n = len(arr.dates)
    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)
    n_train = n - n_val - n_test

    def _slice(s, e):
        return SequenceArrays(
            x_1d=arr.x_1d[s:e], x_1w=arr.x_1w[s:e], x_1m=arr.x_1m[s:e],
            y=arr.y[s:e], dates=arr.dates[s:e],
            feature_cols=arr.feature_cols, label_cols=arr.label_cols,
        )

    return {
        "train": _slice(0, n_train),
        "val":   _slice(n_train, n_train + n_val),
        "test":  _slice(n_train + n_val, n),
    }


# ── PyTorch Datasets ──────────────────────────────────────────────────────────
class TabularDataset(Dataset):
    """Flat feature dataset for MLP/XGBoost baseline."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


class MultiWindowDataset(Dataset):
    """
    Multi-scale sequence dataset for the LSTM model.
    Each item: (x_1d, x_1w, x_1m), y
    """
    def __init__(self, arrays: SequenceArrays):
        self.x_1d = torch.from_numpy(arrays.x_1d)   # (N, F)
        self.x_1w = torch.from_numpy(arrays.x_1w)   # (N, 5, F)
        self.x_1m = torch.from_numpy(arrays.x_1m)   # (N, 21, F)
        self.y    = torch.from_numpy(arrays.y)       # (N, n_horizons)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return (self.x_1d[i], self.x_1w[i], self.x_1m[i]), self.y[i]
