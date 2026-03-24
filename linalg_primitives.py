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


# ──────────────────────────────────────────────────────────────
#  Tensor operations (mode-n unfolding / folding)
# ──────────────────────────────────────────────────────────────

def tensor_unfold(T: NDArray, mode: int) -> NDArray:
    """Mode-n unfolding (matricization) of a 3-D tensor.

    For T ∈ ℝ^{I₁×I₂×I₃}, mode-n unfolding produces a matrix by
    arranging the mode-n fibers as columns.

    Parameters
    ----------
    T    : 3-D array of shape (I1, I2, I3)
    mode : unfolding mode (0, 1, or 2)

    Returns
    -------
    M : 2-D array — mode-n unfolding of T
    """
    T = np.asarray(T)
    ndim = T.ndim
    if ndim != 3:
        raise ValueError(f"tensor_unfold requires a 3-D tensor, got {ndim}-D")
    # Move target mode to axis 0, then reshape
    axes = [mode] + [i for i in range(ndim) if i != mode]
    return np.transpose(T, axes).reshape(T.shape[mode], -1)


def tensor_fold(M: NDArray, mode: int, shape: tuple[int, ...]) -> NDArray:
    """Inverse of tensor_unfold — fold a matrix back into a 3-D tensor.

    Parameters
    ----------
    M     : 2-D matrix (mode-n unfolding)
    mode  : which mode was unfolded
    shape : original tensor shape (I1, I2, I3)

    Returns
    -------
    T : 3-D tensor of given shape
    """
    ndim = len(shape)
    if ndim != 3:
        raise ValueError(f"tensor_fold requires 3-D shape, got {ndim}-D")
    axes_order = [mode] + [i for i in range(ndim) if i != mode]
    new_shape = tuple(shape[a] for a in axes_order)
    T = M.reshape(new_shape)
    # Invert the transposition
    inv_axes = [0] * ndim
    for i, a in enumerate(axes_order):
        inv_axes[a] = i
    return np.transpose(T, inv_axes)


def tensor_n_mode_product(T: NDArray, M: NDArray, mode: int) -> NDArray:
    """n-mode product of a 3-D tensor with a matrix.

    (T ×_n M) is computed by unfolding along mode n, multiplying, and refolding.
    """
    T_unf = tensor_unfold(T, mode)
    result_unf = M @ T_unf
    new_shape = list(T.shape)
    new_shape[mode] = M.shape[0]
    return tensor_fold(result_unf, mode, tuple(new_shape))


def multilinear_rank(T: NDArray) -> tuple[int, ...]:
    """Compute the multilinear rank (Tucker rank) of a 3-D tensor.

    Returns (rank_0, rank_1, rank_2) where rank_n = rank(unfold_n(T)).
    """
    ranks = []
    for mode in range(T.ndim):
        M = tensor_unfold(T, mode)
        s = np.linalg.svd(M, compute_uv=False)
        r = int(np.sum(s > s[0] * 1e-10))
        ranks.append(r)
    return tuple(ranks)


# ──────────────────────────────────────────────────────────────
#  Shift-matrix lag analysis
# ──────────────────────────────────────────────────────────────

def shift_lag_analysis(
    X: NDArray,
    max_lag: int = 10,
) -> tuple[NDArray, NDArray]:
    """Analyse cross-correlation structure via shift-matrix operators.

    For a multi-asset matrix X ∈ ℝ^{n×T}, computes lag-k auto-correlation
    matrices  R(k) = (1/T) · X · S_k^T · X^T  for k = 0, 1, ..., max_lag.

    The shift matrix S_k acts as a discrete lag operator, expressing the
    temporal dependence structure in operator-theoretic form.

    Parameters
    ----------
    X       : (n, T) time-series matrix
    max_lag : number of lags to compute

    Returns
    -------
    lags            : (max_lag+1,) lag indices
    autocorr_norms  : (max_lag+1,) Frobenius norm ‖R(k)‖_F at each lag
    """
    X = np.asarray(X, dtype=np.float64)
    n, T = X.shape
    max_lag = min(max_lag, T - 1)

    lags = np.arange(max_lag + 1)
    norms = np.zeros(max_lag + 1)

    for k in range(max_lag + 1):
        # R(k) = (1/T) * X[:, k:] @ X[:, :T-k].T — efficient computation
        # avoids building the full T×T shift matrix
        valid = T - k
        if valid < 1:
            break
        R_k = X[:, k:] @ X[:, :valid].T / valid
        norms[k] = float(np.linalg.norm(R_k, 'fro'))

    return lags, norms


def nearest_kronecker_product(
    Sigma: NDArray,
    m: int,
    n: int,
) -> tuple[NDArray, NDArray]:
    """Nearest Kronecker product decomposition  Σ ≈ A ⊗ B.

    Uses the Van Loan & Pitsianis rearrangement lemma:
    reshape Σ into a (m²×n²) matrix, then take rank-1 SVD to extract
    the best A ∈ ℝ^{m×m} and B ∈ ℝ^{n×n}.

    Parameters
    ----------
    Sigma : (m*n, m*n) matrix to approximate
    m, n  : block dimensions such that Σ ≈ A_{m×m} ⊗ B_{n×n}

    Returns
    -------
    A : (m, m) inter-block (sector) factor
    B : (n, n) intra-block factor
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    mn = m * n
    assert Sigma.shape == (mn, mn), f"Expected {(mn, mn)}, got {Sigma.shape}"

    # Rearrangement: build the (m², n²) matrix R
    R = np.zeros((m * m, n * n), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            block = Sigma[i * n:(i + 1) * n, j * n:(j + 1) * n]
            R[i * m + j, :] = block.ravel()

    # Rank-1 SVD
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    # A = reshape(√σ₁ · u₁, (m, m)),  B = reshape(√σ₁ · v₁, (n, n))
    sqrt_s = np.sqrt(s[0])
    A = (sqrt_s * U[:, 0]).reshape(m, m)
    B = (sqrt_s * Vt[0, :]).reshape(n, n)

    return A, B
