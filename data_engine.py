"""
data_engine.py
==============
Synthetic FinTech data generation for limit-order-book mid-prices,
cross-asset covariance structures, and controlled anomaly injection.

Produces realistic fragmented market data that exercises every branch
of the Hankel-ADMM reconstruction pipeline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from linalg_primitives import kron


# ──────────────────────────────────────────────────────────────
#  Core LOB generator
# ──────────────────────────────────────────────────────────────

def generate_lob(
    n_assets: int = 8,
    T: int = 512,
    freq_range: tuple[float, float] = (0.01, 0.08),
    noise_sigma: float = 0.05,
    seed: int | None = 42,
) -> NDArray:
    """Generate synthetic limit-order-book mid-price matrix.

    Each asset's mid-price is a superposition of 3 sinusoidal liquidity
    cycles (different frequencies and phases) plus AR(1)-filtered
    Gaussian noise, mimicking realistic microstructure dynamics.

    Parameters
    ----------
    n_assets : number of digital assets
    T        : number of time steps
    freq_range : (lo, hi) range for random cycle frequencies
    noise_sigma : standard deviation of additive noise
    seed     : RNG seed for reproducibility

    Returns
    -------
    X : array (n_assets, T)  — clean mid-price matrix
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float64)
    X = np.zeros((n_assets, T), dtype=np.float64)

    for i in range(n_assets):
        # Baseline drift (mean-reverting random walk)
        drift = np.cumsum(rng.normal(0, 0.002, T))
        drift -= drift.mean()

        # Three periodic liquidity cycles per asset
        signal = np.zeros(T)
        for _ in range(3):
            f = rng.uniform(*freq_range)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.3, 1.0)
            signal += amp * np.sin(2 * np.pi * f * t + phase)

        # AR(1) coloured noise
        noise = np.zeros(T)
        noise[0] = rng.normal(0, noise_sigma)
        phi = rng.uniform(0.7, 0.95)
        for k in range(1, T):
            noise[k] = phi * noise[k - 1] + rng.normal(0, noise_sigma)

        X[i] = 100.0 + 2.0 * signal + drift + noise  # centred around 100

    return X


def generate_cross_asset_matrix(
    n_assets: int = 4,
    T: int = 256,
    block_size: int = 2,
    seed: int | None = 42,
) -> tuple[NDArray, NDArray]:
    """Build a multi-asset price matrix whose covariance has Kronecker structure.

    Covariance Σ = A ⊗ B, where A captures inter-sector correlations and
    B captures intra-sector correlations.  This models realistic cross-
    asset dependencies (e.g., BTC/ETH within crypto-sector, correlated
    with TradFi stablecoins in another sector).

    Returns (X, Sigma) — data matrix and ground-truth covariance.
    """
    rng = np.random.default_rng(seed)
    n_blocks = max(n_assets // block_size, 1)

    # Inter-block (sector) correlation
    A = np.eye(n_blocks) * 1.0
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            c = rng.uniform(0.1, 0.4)
            A[i, j] = c
            A[j, i] = c

    # Intra-block correlation
    B = np.eye(block_size) * 1.0
    for i in range(block_size):
        for j in range(i + 1, block_size):
            c = rng.uniform(0.5, 0.9)
            B[i, j] = c
            B[j, i] = c

    Sigma = kron(A, B)
    dim = Sigma.shape[0]

    # Cholesky factor for sampling
    L_chol = np.linalg.cholesky(Sigma + 1e-8 * np.eye(dim))
    Z = rng.standard_normal((dim, T))
    X = L_chol @ Z

    # Add baseline to make prices positive
    X += 100.0

    return X[:n_assets, :], Sigma[:n_assets, :n_assets]


# ──────────────────────────────────────────────────────────────
#  Anomaly injection
# ──────────────────────────────────────────────────────────────

def inject_anomalies(
    X: NDArray,
    ratio: float = 0.02,
    mode: str = "flash_crash",
    magnitude: float = 5.0,
    seed: int | None = 123,
) -> tuple[NDArray, NDArray]:
    """Inject controlled anomalies into a price matrix.

    Parameters
    ----------
    X : (n_assets, T) clean data
    ratio : fraction of entries to corrupt
    mode : one of 'flash_crash', 'wash_trade', 'latency_gap'
    magnitude : anomaly strength multiplier
    seed : RNG seed

    Returns
    -------
    X_corrupt : modified data
    S_true    : ground-truth sparse anomaly matrix (same shape as X)
    """
    rng = np.random.default_rng(seed)
    X_corrupt = X.copy()
    S_true = np.zeros_like(X)
    n, T = X.shape
    n_anomalies = max(1, int(n * T * ratio))

    if mode == "flash_crash":
        # Sudden spikes — simulates flash crash / pump & dump
        indices = rng.choice(n * T, n_anomalies, replace=False)
        rows, cols = np.unravel_index(indices, (n, T))
        signs = rng.choice([-1.0, 1.0], n_anomalies)
        for r, c, s in zip(rows, cols, signs):
            spike = s * magnitude * np.std(X[r])
            X_corrupt[r, c] += spike
            S_true[r, c] = spike

    elif mode == "wash_trade":
        # Flat plateaus — simulates wash-trading (no real price movement)
        for _ in range(n_anomalies // 10 + 1):
            asset = rng.integers(0, n)
            start = rng.integers(0, max(1, T - 20))
            length = rng.integers(5, min(20, T - start))
            flat_val = X_corrupt[asset, start]
            delta = flat_val - X_corrupt[asset, start:start + length]
            X_corrupt[asset, start:start + length] = flat_val
            S_true[asset, start:start + length] = delta

    elif mode == "latency_gap":
        # NaN gaps — simulates network latency / missing data
        indices = rng.choice(n * T, n_anomalies, replace=False)
        rows, cols = np.unravel_index(indices, (n, T))
        for r, c in zip(rows, cols):
            S_true[r, c] = X_corrupt[r, c]
            X_corrupt[r, c] = np.nan

    else:
        raise ValueError(f"Unknown anomaly mode: {mode}")

    return X_corrupt, S_true


# ──────────────────────────────────────────────────────────────
#  Data fragmentation (cross-venue missing data)
# ──────────────────────────────────────────────────────────────

def fragment_data(
    X: NDArray,
    missing_ratio: float = 0.20,
    seed: int | None = 77,
) -> tuple[NDArray, NDArray]:
    """Randomly mask entries to simulate fragmented cross-venue data.

    Returns
    -------
    X_frag : matrix with NaN in missing positions
    Omega  : boolean mask (True = observed, False = missing)
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(X.shape) > missing_ratio
    X_frag = X.copy()
    X_frag[~mask] = np.nan
    return X_frag, mask
