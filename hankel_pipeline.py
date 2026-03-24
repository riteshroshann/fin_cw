"""
hankel_pipeline.py
==================
Hankel-based signal reconstruction pipeline using Singular Spectrum
Analysis (SSA).  Operates on single time-series and multi-asset
matrices to denoise and impute missing values via rank-constrained
Hankel structure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from linalg_primitives import (
    hankelise,
    dehankelise,
    truncated_svd,
    dft,
    idft,
)


def hankel_ssa_reconstruct(
    x: NDArray,
    L: int | None = None,
    rank: int | None = None,
    energy_threshold: float = 0.95,
) -> NDArray:
    """Singular Spectrum Analysis reconstruction for a single time-series.

    1) Embed x into trajectory (Hankel) matrix H ∈ ℝ^{L×K}
    2) Truncated SVD → rank-r approximation H_r
    3) De-Hankelise (anti-diagonal average) → reconstructed x̂

    Parameters
    ----------
    x : 1-D signal of length T
    L : window (embedding) length.  Default: T // 3.
    rank : number of SVD components to keep.  None → auto via energy.
    energy_threshold : cumulative singular-value energy to retain (when rank=None).

    Returns
    -------
    x_hat : reconstructed signal, same length as x
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    T = x.size

    if L is None:
        L = max(2, T // 3)
    L = min(L, T - 1)

    H = hankelise(x, L)
    U_r, s_r, Vt_r, chosen_rank = truncated_svd(H, rank=rank, energy_threshold=energy_threshold)
    H_approx = U_r @ np.diag(s_r) @ Vt_r
    return dehankelise(H_approx)


def multi_asset_hankel(
    X: NDArray,
    L: int | None = None,
    rank: int | None = None,
) -> NDArray:
    """Apply SSA reconstruction to each row (asset) of a multi-asset matrix.

    Missing values (NaN) in each row are first linearly interpolated
    before SSA, so this also serves as an imputation step.

    Parameters
    ----------
    X : (n_assets, T) matrix, may contain NaN
    L : Hankel window length
    rank : SSA truncation rank per asset

    Returns
    -------
    X_hat : (n_assets, T) reconstructed matrix with no NaN
    """
    X = np.asarray(X, dtype=np.float64).copy()
    n, T = X.shape

    if L is None:
        L = max(2, T // 3)

    X_hat = np.zeros_like(X)
    for i in range(n):
        row = X[i].copy()

        # Interpolate NaN for initial conditioning
        nans = np.isnan(row)
        if nans.any():
            valid = ~nans
            if valid.sum() < 3:
                # Fallback: fill with mean of valid or zero
                row[nans] = np.nanmean(row) if valid.any() else 0.0
            else:
                row[nans] = np.interp(
                    np.where(nans)[0],
                    np.where(valid)[0],
                    row[valid],
                )

        X_hat[i] = hankel_ssa_reconstruct(row, L=L, rank=rank)

    return X_hat


def spectral_filter(
    x: NDArray,
    keep_freqs: int | None = None,
    cutoff_ratio: float = 0.15,
) -> NDArray:
    """Frequency-domain bandpass filter via DFT → zero-out → IDFT.

    Parameters
    ----------
    x : 1-D signal of length T
    keep_freqs : number of lowest-frequency DFT bins to keep.
                 Default: int(T * cutoff_ratio).
    cutoff_ratio : used when keep_freqs is None.

    Returns
    -------
    x_filtered : real-valued filtered signal
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    T = x.size

    if keep_freqs is None:
        keep_freqs = max(1, int(T * cutoff_ratio))

    X_f = dft(x)
    # Zero-out high frequencies (keep DC + low bins + symmetric negative freqs)
    mask = np.zeros(T, dtype=bool)
    mask[:keep_freqs] = True
    mask[-keep_freqs + 1:] = True  # negative frequencies
    X_f[~mask] = 0.0

    return np.real(idft(X_f))
