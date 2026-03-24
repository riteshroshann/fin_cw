"""
linalg_primitives.py
====================
Core linear-algebra building blocks for the Hankel-ADMM liquidity
reconstruction framework.

Implements:
    • Kronecker product & shift (lag) matrices
    • Hankelization / de-Hankelization (anti-diagonal averaging)
    • DFT / IDFT via explicit O(n²) matrix and FFT wrapper
    • Truncated SVD with explicit rank selection
    • Proximal operators: soft-thresholding (ℓ₁) and nuclear-norm (SVT)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ──────────────────────────────────────────────────────────────
#  Kronecker & Shift
# ──────────────────────────────────────────────────────────────

def kron(A: NDArray, B: NDArray) -> NDArray:
    """Kronecker product A ⊗ B without relying on np.kron internals.

    For A ∈ ℝ^{m×n} and B ∈ ℝ^{p×q}, returns C ∈ ℝ^{mp×nq}.
    """
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    m, n = A.shape
    p, q = B.shape
    return (A[:, np.newaxis, :, np.newaxis] * B[np.newaxis, :, np.newaxis, :]).reshape(m * p, n * q)


def shift_matrix(n: int, k: int = 1) -> NDArray:
    """Construct the k-step forward shift matrix S_k ∈ ℝ^{n×n}.

    (S_k)_{i,j} = 1 iff j = i + k.  Acts as a discrete lag operator:
    S_k @ x shifts x by k positions.
    """
    S = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = i + k
        if 0 <= j < n:
            S[i, j] = 1.0
    return S


# ──────────────────────────────────────────────────────────────
#  Hankelization
# ──────────────────────────────────────────────────────────────

def hankelise(x: NDArray, L: int) -> NDArray:
    """Map a 1-D signal x ∈ ℝ^T into a Hankel matrix H ∈ ℝ^{L×K}.

    K = T - L + 1.  H[i, j] = x[i + j], the trajectory matrix used
    in Singular Spectrum Analysis.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    T = x.size
    K = T - L + 1
    if K < 1:
        raise ValueError(f"Window length L={L} exceeds signal length T={T}")
    # Stride trick for zero-copy Hankel view
    from numpy.lib.stride_tricks import as_strided
    s = x.strides[0]
    return as_strided(x, shape=(L, K), strides=(s, s)).copy()


def dehankelise(H: NDArray) -> NDArray:
    """Recover a 1-D signal from a (possibly low-rank) Hankel-like matrix
    via anti-diagonal averaging.

    Given H ∈ ℝ^{L×K}, returns x ∈ ℝ^{L+K-1}.
    """
    H = np.asarray(H, dtype=np.float64)
    L, K = H.shape
    T = L + K - 1
    x = np.zeros(T, dtype=np.float64)
    counts = np.zeros(T, dtype=np.float64)
    for i in range(L):
        for j in range(K):
            x[i + j] += H[i, j]
            counts[i + j] += 1.0
    return x / counts


# ──────────────────────────────────────────────────────────────
#  DFT / IDFT
# ──────────────────────────────────────────────────────────────

def dft_matrix(n: int) -> NDArray:
    """Explicit n×n DFT matrix  F[k, j] = exp(-2πi·k·j / n) / √n."""
    idx = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(idx, idx) / n) / np.sqrt(n)


def dft_explicit(x: NDArray) -> NDArray:
    """O(n²) DFT via matrix multiplication (pedagogical reference)."""
    x = np.asarray(x, dtype=np.complex128).ravel()
    F = dft_matrix(x.size)
    return F @ x


def idft_explicit(X: NDArray) -> NDArray:
    """O(n²) IDFT via conjugate-transpose of DFT matrix."""
    X = np.asarray(X, dtype=np.complex128).ravel()
    F = dft_matrix(X.size)
    return F.conj().T @ X


def dft(x: NDArray) -> NDArray:
    """Production DFT using FFT, normalised to match dft_explicit."""
    x = np.asarray(x, dtype=np.complex128).ravel()
    return np.fft.fft(x) / np.sqrt(x.size)


def idft(X: NDArray) -> NDArray:
    """Production IDFT using IFFT, consistent normalisation."""
    X = np.asarray(X, dtype=np.complex128).ravel()
    return np.fft.ifft(X) * np.sqrt(X.size)


# ──────────────────────────────────────────────────────────────
#  SVD utilities
# ──────────────────────────────────────────────────────────────

def truncated_svd(
    M: NDArray,
    rank: int | None = None,
    energy_threshold: float = 0.99,
) -> tuple[NDArray, NDArray, NDArray, int]:
    """Thin SVD with rank selection.

    Parameters
    ----------
    M : array (m, n)
    rank : explicit rank cutoff.  If None, select automatically via
           cumulative energy threshold.
    energy_threshold : fraction of total singular-value energy to retain
                       when rank is None.

    Returns
    -------
    U_r, sigma_r, Vt_r, chosen_rank
    """
    M = np.asarray(M, dtype=np.float64)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    if rank is None:
        cum_energy = np.cumsum(s ** 2)
        total = cum_energy[-1]
        rank = int(np.searchsorted(cum_energy, energy_threshold * total)) + 1
    rank = min(rank, len(s))

    return U[:, :rank], s[:rank], Vt[:rank, :], rank


def low_rank_approx(M: NDArray, rank: int) -> NDArray:
    """Best rank-r approximation of M in Frobenius norm (Eckart–Young)."""
    U_r, s_r, Vt_r, _ = truncated_svd(M, rank=rank)
    return U_r @ np.diag(s_r) @ Vt_r


# ──────────────────────────────────────────────────────────────
#  Proximal operators
# ──────────────────────────────────────────────────────────────

def prox_l1(X: NDArray, lam: float) -> NDArray:
    """Element-wise soft-thresholding (proximal of λ‖·‖₁).

    prox_{λ‖·‖₁}(X) = sign(X) ⊙ max(|X| - λ, 0)
    """
    X = np.asarray(X, dtype=np.float64)
    return np.sign(X) * np.maximum(np.abs(X) - lam, 0.0)


def prox_nuclear(X: NDArray, lam: float) -> tuple[NDArray, int]:
    """Singular Value Thresholding (proximal of λ‖·‖_*).

    Applies soft-thresholding to the singular values:
        prox_{λ‖·‖_*}(X) = U diag(max(σ - λ, 0)) Vᵀ

    Returns (thresholded matrix, effective rank).
    """
    X = np.asarray(X, dtype=np.float64)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_thresh = np.maximum(s - lam, 0.0)
    rank = int(np.sum(s_thresh > 0))
    return U @ np.diag(s_thresh) @ Vt, rank


# ──────────────────────────────────────────────────────────────
#  Utility: Frobenius / spectral norms
# ──────────────────────────────────────────────────────────────

def frobenius(A: NDArray) -> float:
    """Frobenius norm ‖A‖_F."""
    return float(np.linalg.norm(np.asarray(A), 'fro'))


def spectral_norm(A: NDArray) -> float:
    """Spectral norm ‖A‖₂ = σ_max."""
    return float(np.linalg.norm(np.asarray(A), 2))
