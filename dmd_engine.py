"""
dmd_engine.py
=============
Dynamic Mode Decomposition (DMD) for operator-theoretic analysis
of reconstructed liquidity dynamics.

Implements exact DMD, multi-step forecasting, stability classification,
and frequency-spectrum extraction from DMD eigenvalues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from linalg_primitives import truncated_svd


@dataclass
class StabilityReport:
    """Classification of DMD modes by spectral radius."""
    eigenvalues: NDArray
    magnitudes: NDArray
    frequencies: NDArray          # in Hz (given dt)
    growth_rates: NDArray         # ln|λ| / dt
    stable_mask: NDArray          # |λ| < 1 - ε
    marginal_mask: NDArray        # |λ| ≈ 1
    unstable_mask: NDArray        # |λ| > 1 + ε
    dominant_frequency: float = 0.0
    spectral_radius: float = 0.0


def exact_dmd(
    X: NDArray,
    rank: int | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Exact Dynamic Mode Decomposition.

    Given snapshot matrix X ∈ ℝ^{n×T}, forms:
        X₁ = X[:, :-1],  X₂ = X[:, 1:]
    and computes the rank-r DMD operator.

    Parameters
    ----------
    X : (n, T) snapshot matrix
    rank : truncation rank for SVD.  None → auto via energy threshold.

    Returns
    -------
    Phi  : (n, r) DMD modes  (columns are spatial modes)
    lam  : (r,)   DMD eigenvalues
    b    : (r,)   mode amplitudes (initial condition coefficients)
    A_tilde : (r, r) reduced dynamics matrix
    """
    X = np.asarray(X, dtype=np.float64)
    X1 = X[:, :-1]
    X2 = X[:, 1:]

    # SVD of X₁
    U_r, s_r, Vt_r, r = truncated_svd(X1, rank=rank, energy_threshold=0.99)

    # Reduced dynamics:  à = U_r^H X₂ V_r Σ_r^{-1}
    A_tilde = U_r.T @ X2 @ Vt_r.T @ np.diag(1.0 / s_r)

    # Eigen-decomposition of Ã
    eigvals, W = np.linalg.eig(A_tilde)

    # DMD modes in original coordinates
    Phi = X2 @ Vt_r.T @ np.diag(1.0 / s_r) @ W

    # Initial amplitudes:  b = Φ⁺ x₁
    b = np.linalg.lstsq(Phi, X[:, 0], rcond=None)[0]

    return Phi, eigvals, b, A_tilde


def predict_dmd(
    Phi: NDArray,
    lam: NDArray,
    b: NDArray,
    t_steps: int,
) -> NDArray:
    """Reconstruct / forecast using DMD modes.

    x(t) = Σ_j  b_j · λ_j^t · φ_j

    Parameters
    ----------
    Phi   : (n, r) DMD modes
    lam   : (r,)  eigenvalues
    b     : (r,)  amplitudes
    t_steps : number of time steps to generate

    Returns
    -------
    X_pred : (n, t_steps) predicted snapshot matrix
    """
    r = lam.size
    n = Phi.shape[0]

    # Vandermonde matrix of eigenvalues
    vander = np.zeros((r, t_steps), dtype=np.complex128)
    for t in range(t_steps):
        vander[:, t] = lam ** t

    # Scale by amplitudes
    dynamics = np.diag(b) @ vander

    X_pred = Phi @ dynamics
    return np.real(X_pred)


def dmd_stability_analysis(
    eigenvalues: NDArray,
    dt: float = 1.0,
    tol: float = 0.02,
) -> StabilityReport:
    """Classify DMD modes as stable, marginal, or unstable.

    Parameters
    ----------
    eigenvalues : complex DMD eigenvalues
    dt : sampling interval (for frequency conversion)
    tol : tolerance band around unit circle for "marginal" classification

    Returns
    -------
    StabilityReport with full classification
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
    mags = np.abs(eigenvalues)
    phases = np.angle(eigenvalues)

    freqs = phases / (2 * np.pi * dt)
    growth = np.log(np.maximum(mags, 1e-15)) / dt

    stable = mags < (1.0 - tol)
    unstable = mags > (1.0 + tol)
    marginal = ~stable & ~unstable

    # Dominant frequency: mode with largest amplitude near unit circle
    marginal_or_near = mags > 0.5
    if marginal_or_near.any():
        dominant_idx = np.argmax(mags[marginal_or_near])
        dominant_freq = float(np.abs(freqs[marginal_or_near][dominant_idx]))
    else:
        dominant_freq = 0.0

    return StabilityReport(
        eigenvalues=eigenvalues,
        magnitudes=mags,
        frequencies=freqs,
        growth_rates=growth,
        stable_mask=stable,
        marginal_mask=marginal,
        unstable_mask=unstable,
        dominant_frequency=dominant_freq,
        spectral_radius=float(np.max(mags)),
    )


def dmd_frequency_spectrum(
    eigenvalues: NDArray,
    dt: float = 1.0,
) -> tuple[NDArray, NDArray]:
    """Extract oscillation frequencies and growth/decay rates.

    Returns
    -------
    frequencies  : (r,)  oscillation frequencies in Hz
    growth_rates : (r,)  exponential growth rates (negative = decay)
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
    freqs = np.angle(eigenvalues) / (2 * np.pi * dt)
    growth = np.log(np.maximum(np.abs(eigenvalues), 1e-15)) / dt
    return freqs, growth
