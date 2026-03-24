"""
kaggle_loader.py
================
Data loader for the Kaggle "High Frequency Crypto Limit Order Book Data"
dataset by Martin Søgaard Nielsen.

Dataset: https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data

Structure (after unzip into ``data/``):
    data/
    ├── btc_usdt/
    │   ├── btc_usdt_1sec.csv
    │   ├── btc_usdt_1min.csv
    │   └── btc_usdt_5min.csv
    ├── eth_usdt/
    │   ├── eth_usdt_1sec.csv
    │   ├── eth_usdt_1min.csv
    │   └── eth_usdt_5min.csv
    └── ada_usdt/
        ├── ada_usdt_1sec.csv
        ├── ada_usdt_1min.csv
        └── ada_usdt_5min.csv

Each CSV has 15 bid/ask levels with columns:
    midpoint, spread,
    bids_distance_{0..14}, asks_distance_{0..14},
    bids_market_notional_{0..14}, bids_limit_notional_{0..14},
    bids_cancel_notional_{0..14},
    asks_market_notional_{0..14}, asks_limit_notional_{0..14},
    asks_cancel_notional_{0..14}

This module transforms raw LOB data into the multi-asset mid-price
matrix and feature tensors consumed by the Hankel-ADMM pipeline.
"""

from __future__ import annotations

import os
import sys
import glob
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

DATA_DIR = Path(__file__).parent / "data"

ASSETS = ["btc_usdt", "eth_usdt", "ada_usdt"]
FREQS  = {"1sec": "1sec", "1min": "1min", "5min": "5min"}
N_LEVELS = 15


# ──────────────────────────────────────────────────────────────
#  Dataset download helper
# ──────────────────────────────────────────────────────────────

def download_dataset(dest: str | Path | None = None) -> Path:
    """Attempt to download the dataset via Kaggle API.

    Requires:
      1. ``pip install kaggle``
      2. ``~/.kaggle/kaggle.json`` with valid API credentials
         (or KAGGLE_USERNAME / KAGGLE_KEY env vars)

    Falls back to manual-download instructions if the API is
    unavailable.
    """
    dest = Path(dest) if dest else DATA_DIR
    dest.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"  ⏳ Downloading dataset to {dest} (≈1 GB)...")
        api.dataset_download_files(
            "martinsn/high-frequency-crypto-limit-order-book-data",
            path=str(dest),
            unzip=True,
        )
        print("  ✓ Download complete.")
        return dest
    except Exception as e:
        print(f"\n{'─' * 60}")
        print("  ⚠  Kaggle API download failed:")
        print(f"     {e}")
        print()
        print("  Manual download steps:")
        print("  1. Visit: https://www.kaggle.com/datasets/martinsn/"
              "high-frequency-crypto-limit-order-book-data")
        print("  2. Click 'Download' (requires Kaggle account)")
        print(f"  3. Unzip the archive into: {dest.resolve()}")
        print()
        print("  Expected structure:")
        for a in ASSETS:
            print(f"    {dest}/{a}/{a}_1sec.csv")
        print(f"{'─' * 60}\n")
        raise FileNotFoundError(
            f"Dataset not found at {dest}. Follow instructions above."
        ) from e


# ──────────────────────────────────────────────────────────────
#  Discovery: locate CSV files
# ──────────────────────────────────────────────────────────────

def _discover_csvs(data_dir: Path) -> dict[str, dict[str, Path]]:
    """Map (asset, freq) → CSV path by searching the data directory."""
    found: dict[str, dict[str, Path]] = {}

    # Try structured subfolders first (btc_usdt/btc_usdt_1sec.csv)
    for asset in ASSETS:
        found[asset] = {}
        for freq_key, freq_suffix in FREQS.items():
            candidates = [
                data_dir / asset / f"{asset}_{freq_suffix}.csv",
                data_dir / f"{asset}_{freq_suffix}.csv",
                data_dir / asset.upper() / f"{asset}_{freq_suffix}.csv",
            ]
            # Also search recursively
            recursive = list(data_dir.rglob(f"*{asset}*{freq_suffix}*.csv"))
            candidates.extend(recursive)

            for c in candidates:
                if c.exists():
                    found[asset][freq_key] = c
                    break

    return found


def check_data_available(data_dir: Path | None = None) -> bool:
    """Return True if at least one asset's CSV file is found."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    if not data_dir.exists():
        return False
    csvs = _discover_csvs(data_dir)
    return any(bool(freqs) for freqs in csvs.values())


# ──────────────────────────────────────────────────────────────
#  CSV loading (numpy-only fallback when pandas unavailable)
# ──────────────────────────────────────────────────────────────

def _load_csv_numpy(path: Path, max_rows: int | None = None) -> tuple[NDArray, list[str]]:
    """Load CSV with numpy only (header + float data)."""
    with open(path, "r") as f:
        header = f.readline().strip().split(",")

    data = np.genfromtxt(
        path, delimiter=",", skip_header=1,
        max_rows=max_rows, filling_values=np.nan,
    )
    return data, header


def _load_csv_pandas(path: Path, max_rows: int | None = None) -> tuple[NDArray, list[str]]:
    """Load CSV with pandas for better dtype inference and speed."""
    df = pd.read_csv(path, nrows=max_rows)
    # Drop any timestamp/date columns, keep numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols].values.astype(np.float64), numeric_cols


def load_csv(path: Path, max_rows: int | None = None) -> tuple[NDArray, list[str]]:
    """Load a single LOB CSV. Returns (data_array, column_names)."""
    if HAS_PANDAS:
        return _load_csv_pandas(path, max_rows)
    return _load_csv_numpy(path, max_rows)


# ──────────────────────────────────────────────────────────────
#  Mid-price extraction
# ──────────────────────────────────────────────────────────────

def extract_midprices(
    data_dir: Path | None = None,
    freq: str = "1min",
    max_rows: int | None = None,
    align_length: bool = True,
) -> tuple[NDArray, list[str]]:
    """Extract mid-price time-series for all available assets.

    Parameters
    ----------
    data_dir   : root of the unzipped dataset
    freq       : one of '1sec', '1min', '5min'
    max_rows   : limit rows per asset (useful for testing)
    align_length : if True, truncate all series to the shortest length

    Returns
    -------
    midprices : (n_assets, T) matrix of mid-prices
    asset_names : list of asset tickers
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    csvs = _discover_csvs(data_dir)

    series = []
    names = []
    for asset in ASSETS:
        freq_map = csvs.get(asset, {})
        if freq not in freq_map:
            continue
        data, cols = load_csv(freq_map[freq], max_rows=max_rows)
        # Find midpoint column
        mid_idx = None
        for i, c in enumerate(cols):
            if "midpoint" in c.lower() or "mid" in c.lower():
                mid_idx = i
                break
        if mid_idx is None:
            # Fallback: use first column
            mid_idx = 0

        mid = data[:, mid_idx]
        # Drop leading/trailing NaN
        valid = ~np.isnan(mid)
        if valid.any():
            first_valid = np.argmax(valid)
            last_valid = len(valid) - np.argmax(valid[::-1])
            mid = mid[first_valid:last_valid]

        series.append(mid)
        names.append(asset.replace("_usdt", "").upper())

    if not series:
        raise FileNotFoundError(
            f"No valid CSV files found in {data_dir}. "
            f"Run download_dataset() or manually place data."
        )

    if align_length:
        min_len = min(len(s) for s in series)
        series = [s[:min_len] for s in series]

    return np.array(series, dtype=np.float64), names


# ──────────────────────────────────────────────────────────────
#  Feature tensor extraction (full LOB depth)
# ──────────────────────────────────────────────────────────────

def extract_lob_features(
    data_dir: Path | None = None,
    asset: str = "btc_usdt",
    freq: str = "1min",
    max_rows: int | None = None,
    levels: int = 5,
) -> dict[str, NDArray]:
    """Extract rich LOB feature set for a single asset.

    Returns a dict with keys:
        'midpoint'           : (T,)
        'spread'             : (T,)
        'bid_distances'      : (T, levels)
        'ask_distances'      : (T, levels)
        'bid_limit_notional' : (T, levels)
        'ask_limit_notional' : (T, levels)
        'bid_market_notional': (T, levels)
        'ask_market_notional': (T, levels)
        'bid_cancel_notional': (T, levels)
        'ask_cancel_notional': (T, levels)
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    csvs = _discover_csvs(data_dir)

    if asset not in csvs or freq not in csvs[asset]:
        raise FileNotFoundError(f"No data for {asset} @ {freq}")

    data, cols = load_csv(csvs[asset][freq], max_rows=max_rows)
    col_idx = {c.strip(): i for i, c in enumerate(cols)}

    levels = min(levels, N_LEVELS)

    def _get_col(name: str) -> NDArray:
        for key, idx in col_idx.items():
            if name.lower() in key.lower():
                return data[:, idx]
        return np.full(data.shape[0], np.nan)

    def _get_levels(prefix: str) -> NDArray:
        arr = np.full((data.shape[0], levels), np.nan)
        for lv in range(levels):
            for key, idx in col_idx.items():
                if prefix in key.lower() and str(lv) in key:
                    # Match exact level number
                    # e.g., "bids_distance_3" should match level 3
                    parts = key.split("_")
                    try:
                        level_num = int(parts[-1])
                        if level_num == lv:
                            arr[:, lv] = data[:, idx]
                            break
                    except ValueError:
                        continue
        return arr

    features = {
        "midpoint": _get_col("midpoint"),
        "spread": _get_col("spread"),
        "bid_distances": _get_levels("bids_distance"),
        "ask_distances": _get_levels("asks_distance"),
        "bid_limit_notional": _get_levels("bids_limit_notional"),
        "ask_limit_notional": _get_levels("asks_limit_notional"),
        "bid_market_notional": _get_levels("bids_market_notional"),
        "ask_market_notional": _get_levels("asks_market_notional"),
        "bid_cancel_notional": _get_levels("bids_cancel_notional"),
        "ask_cancel_notional": _get_levels("asks_cancel_notional"),
    }

    return features


# ──────────────────────────────────────────────────────────────
#  Liquidity depth matrix (multi-level, multi-asset)
# ──────────────────────────────────────────────────────────────

def build_liquidity_depth_matrix(
    data_dir: Path | None = None,
    freq: str = "1min",
    levels: int = 5,
    max_rows: int | None = None,
) -> tuple[NDArray, list[str]]:
    """Build a composite liquidity-depth matrix.

    For each asset, computes total bid+ask limit notional across
    the top `levels` price levels, producing one row per asset.

    Returns
    -------
    depth_matrix : (n_assets, T) — total depth per asset over time
    asset_names  : list of asset tickers
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    csvs = _discover_csvs(data_dir)

    rows = []
    names = []
    for asset in ASSETS:
        if asset not in csvs or freq not in csvs[asset]:
            continue
        feats = extract_lob_features(data_dir, asset, freq, max_rows, levels)
        bid_depth = np.nansum(feats["bid_limit_notional"], axis=1)
        ask_depth = np.nansum(feats["ask_limit_notional"], axis=1)
        total = bid_depth + ask_depth
        rows.append(total)
        names.append(asset.replace("_usdt", "").upper())

    if not rows:
        raise FileNotFoundError("No LOB data found.")

    min_len = min(len(r) for r in rows)
    rows = [r[:min_len] for r in rows]
    return np.array(rows, dtype=np.float64), names


# ──────────────────────────────────────────────────────────────
#  Order-flow imbalance (market microstructure signal)
# ──────────────────────────────────────────────────────────────

def compute_order_flow_imbalance(
    data_dir: Path | None = None,
    asset: str = "btc_usdt",
    freq: str = "1min",
    levels: int = 5,
    max_rows: int | None = None,
) -> NDArray:
    """Compute order-flow imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).

    This is a key microstructure signal for predicting short-term
    price movements and detecting predatory trading.
    """
    feats = extract_lob_features(data_dir, asset, freq, max_rows, levels)
    bid_depth = np.nansum(feats["bid_limit_notional"], axis=1)
    ask_depth = np.nansum(feats["ask_limit_notional"], axis=1)
    total = bid_depth + ask_depth
    total = np.where(total < 1e-12, 1e-12, total)  # avoid division by zero
    return (bid_depth - ask_depth) / total


# ──────────────────────────────────────────────────────────────
#  Summary / inspection
# ──────────────────────────────────────────────────────────────

def dataset_summary(data_dir: Path | None = None) -> str:
    """Print a summary of available data files and shapes."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    csvs = _discover_csvs(data_dir)

    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║  Kaggle LOB Dataset Summary                                 ║",
        "╠══════════════════════════════════════════════════════════════╣",
    ]

    for asset in ASSETS:
        ticker = asset.replace("_usdt", "").upper()
        freq_map = csvs.get(asset, {})
        if not freq_map:
            lines.append(f"║  {ticker:6s}  │  ❌ No data found                          ║")
            continue
        for freq, path in freq_map.items():
            try:
                data, cols = load_csv(path, max_rows=5)
                # Get full row count cheaply
                if HAS_PANDAS:
                    import pandas as pd
                    n_rows = sum(1 for _ in open(path)) - 1
                else:
                    n_rows = sum(1 for _ in open(path)) - 1
                lines.append(
                    f"║  {ticker:6s}  │  {freq:5s}  │  "
                    f"{n_rows:>8,} rows  │  {len(cols):>3} cols  ║"
                )
            except Exception as e:
                lines.append(f"║  {ticker:6s}  │  {freq:5s}  │  ⚠ Error: {e!s:.30s}  ║")

    lines.append("╚══════════════════════════════════════════════════════════════╝")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  Augmented feature matrix for depth-aware reconstruction
# ──────────────────────────────────────────────────────────────

def build_augmented_matrix(
    data_dir: Path | None = None,
    freq: str = "1min",
    levels: int = 5,
    max_rows: int | None = None,
) -> tuple[NDArray, list[str]]:
    """Stack mid-price, total depth, and OFI into one feature matrix.

    For each asset, computes three feature rows:
        1. midpoint price
        2. total bid+ask depth (sum of limit notionals)
        3. order-flow imbalance (bid-ask depth ratio)

    All rows are z-score normalised so the ADMM solver sees
    comparable scales across heterogeneous features.

    Returns
    -------
    augmented : (3 * n_assets, T) feature matrix
    labels    : list of row labels (e.g. "BTC_mid", "BTC_depth", "BTC_ofi")
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    csvs = _discover_csvs(data_dir)

    rows: list[NDArray] = []
    labels: list[str] = []

    for asset in ASSETS:
        ticker = asset.replace("_usdt", "").upper()
        if asset not in csvs or freq not in csvs[asset]:
            continue
        feats = extract_lob_features(data_dir, asset, freq, max_rows, levels)

        mid = feats["midpoint"]
        bid_depth = np.nansum(feats["bid_limit_notional"], axis=1)
        ask_depth = np.nansum(feats["ask_limit_notional"], axis=1)
        total_depth = bid_depth + ask_depth
        total_safe = np.where(total_depth < 1e-12, 1e-12, total_depth)
        ofi = (bid_depth - ask_depth) / total_safe

        rows.extend([mid, total_depth, ofi])
        labels.extend([f"{ticker}_mid", f"{ticker}_depth", f"{ticker}_ofi"])

    if not rows:
        raise FileNotFoundError("No LOB data found for augmented matrix.")

    # Align lengths
    min_len = min(len(r) for r in rows)
    rows = [r[:min_len] for r in rows]

    # Z-score normalisation per row
    mat = np.array(rows, dtype=np.float64)
    for i in range(mat.shape[0]):
        mu = np.nanmean(mat[i])
        sigma = np.nanstd(mat[i])
        if sigma > 1e-12:
            mat[i] = (mat[i] - mu) / sigma
        else:
            mat[i] -= mu

    # Replace any remaining NaN with 0
    mat = np.nan_to_num(mat, nan=0.0)

    return mat, labels


if __name__ == "__main__":
    if not check_data_available():
        print("Dataset not found. Attempting download...")
        try:
            download_dataset()
        except FileNotFoundError:
            sys.exit(1)

    print(dataset_summary())

    # Quick test: extract mid-prices
    midprices, names = extract_midprices(freq="1min", max_rows=1000)
    print(f"\nMid-prices shape: {midprices.shape}")
    print(f"Assets: {names}")
    for i, name in enumerate(names):
        print(f"  {name}: range [{midprices[i].min():.2f}, {midprices[i].max():.2f}]")
