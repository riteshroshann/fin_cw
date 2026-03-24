"""
admm_solver.py
==============
Augmented-Lagrangian ADMM solvers for:

1. Robust PCA  (low-rank + sparse decomposition)
       min  ‖X‖_*  +  λ ‖S‖₁   s.t.  P_Ω(M) = P_Ω(X + S)

2. Pure matrix completion  (low-rank recovery from partial observations)
       min  ½‖P_Ω(M - X)‖²_F  +  μ ‖X‖_*

Both solvers return full convergence diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from linalg_primitives import prox_l1, prox_nuclear, frobenius


@dataclass
class ADMMHistory:
    """Convergence diagnostics container."""
    primal_residual: list[float] = field(default_factory=list)
    dual_residual: list[float] = field(default_factory=list)
    rank: list[int] = field(default_factory=list)
    objective: list[float] = field(default_factory=list)
    iterations: int = 0


# ──────────────────────────────────────────────────────────────
#  Robust PCA via ADMM
# ──────────────────────────────────────────────────────────────

def admm_rpca(
    M: NDArray,
    Omega: NDArray | None = None,
    lam: float | None = None,
    rho: float = 1.0,
    max_iter: int = 300,
    tol: float = 1e-6,
    verbose: bool = False,
) -> tuple[NDArray, NDArray, ADMMHistory]:
    """Robust PCA: decompose M = X + S where X is low-rank, S is sparse.

    Solves via ADMM:
        X^{k+1} = prox_{(1/ρ)‖·‖_*}(M - S^k - Y^k/ρ)
        S^{k+1} = prox_{(λ/ρ)‖·‖₁}(M - X^{k+1} - Y^k/ρ)
        Y^{k+1} = Y^k + ρ(X^{k+1} + S^{k+1} - M)

    When Omega is provided, the data fidelity is restricted to observed entries.

    Parameters
    ----------
    M       : (m, n) observation matrix (may contain NaN for missing)
    Omega   : (m, n) boolean mask — True = observed.  None → fully observed.
    lam     : sparsity weight.  Default: 1/√max(m,n).
    rho     : ADMM penalty parameter.
    max_iter: hard iteration cap.
    tol     : convergence tolerance on normalised residuals.

    Returns
    -------
    X_hat : low-rank component
    S_hat : sparse anomaly component
    hist  : ADMMHistory with convergence diagnostics
    """
    M = np.asarray(M, dtype=np.float64)
    m, n = M.shape

    if Omega is None:
        Omega = ~np.isnan(M)
        M = np.nan_to_num(M, nan=0.0)
    else:
        Omega = np.asarray(Omega, dtype=bool)
        M = np.where(Omega, M, 0.0)

    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    # Initialisations
    X = np.zeros((m, n), dtype=np.float64)
    S = np.zeros((m, n), dtype=np.float64)
    Y = np.zeros((m, n), dtype=np.float64)  # dual variable
    hist = ADMMHistory()

    M_norm = max(frobenius(M), 1e-10)

    for k in range(max_iter):
        X_old = X.copy()

        # ── X-update: nuclear-norm proximal ──
        V = M - S - Y / rho
        # Only enforce data fidelity on observed entries
        V_proj = np.where(Omega, V, X)  # keep previous estimate for unobserved
        X, rank_k = prox_nuclear(V_proj, 1.0 / rho)

        # ── S-update: ℓ₁ proximal ──
        W = M - X - Y / rho
        S = prox_l1(W, lam / rho)
        # Zero out S at unobserved positions (cannot detect anomalies there)
        S = np.where(Omega, S, 0.0)

        # ── Y-update (dual ascent) ──
        residual = np.where(Omega, X + S - M, 0.0)
        Y += rho * residual

        # ── Convergence tracking ──
        primal_res = frobenius(residual) / M_norm
        dual_res = frobenius(rho * (X - X_old)) / M_norm

        # Objective: ‖X‖_* + λ‖S‖₁
        nuc_norm = float(np.sum(np.linalg.svd(X, compute_uv=False)))
        obj = nuc_norm + lam * float(np.sum(np.abs(S)))

        hist.primal_residual.append(primal_res)
        hist.dual_residual.append(dual_res)
        hist.rank.append(rank_k)
        hist.objective.append(obj)

        # ── Adaptive ρ-scaling (Boyd et al. §3.4.1) ──
        # Balance primal and dual convergence rates by adjusting ρ
        _tau = 2.0
        _mu_adapt = 10.0
        if primal_res > _mu_adapt * dual_res:
            rho *= _tau
            Y *= (1.0 / _tau)  # rescale dual to maintain Y/ρ invariant
        elif dual_res > _mu_adapt * primal_res:
            rho /= _tau
            Y *= _tau

        if verbose and k % 20 == 0:
            print(f"  ADMM iter {k:4d} | primal {primal_res:.2e} | "
                  f"dual {dual_res:.2e} | rank {rank_k} | ρ {rho:.4f}")

        if primal_res < tol and dual_res < tol:
            break

    hist.iterations = k + 1
    return X, S, hist


# ──────────────────────────────────────────────────────────────
#  Pure Matrix Completion via ADMM
# ──────────────────────────────────────────────────────────────

def admm_matrix_completion(
    M: NDArray,
    Omega: NDArray,
    mu: float = 1.0,
    rho: float = 1.0,
    max_iter: int = 300,
    tol: float = 1e-6,
    verbose: bool = False,
) -> tuple[NDArray, ADMMHistory]:
    """Matrix completion via ADMM (no sparse term).

    Solves:  min  μ‖X‖_*  +  ½‖P_Ω(X - M)‖²_F
    using variable splitting X = Z, then ADMM.

    Parameters
    ----------
    M     : (m, n) partially observed matrix (NaN or 0 for missing)
    Omega : (m, n) boolean observation mask
    mu    : nuclear-norm regularisation weight
    rho   : ADMM penalty
    max_iter, tol : stopping criteria

    Returns
    -------
    X_hat : completed matrix
    hist  : convergence diagnostics
    """
    M = np.asarray(M, dtype=np.float64)
    Omega = np.asarray(Omega, dtype=bool)
    m, n = M.shape

    M_clean = np.where(Omega, M, 0.0)

    X = M_clean.copy()
    Z = X.copy()
    U = np.zeros_like(X)  # scaled dual
    hist = ADMMHistory()

    M_norm = max(frobenius(M_clean), 1e-10)

    for k in range(max_iter):
        Z_old = Z.copy()

        # X-update (closed-form with observation constraint)
        # X = (ρZ - U + P_Ω(M)) / (ρ + P_Ω(1))
        numerator = rho * Z - U
        numerator[Omega] += M_clean[Omega]
        denominator = rho * np.ones((m, n))
        denominator[Omega] += 1.0
        X = numerator / denominator

        # Z-update: nuclear norm proximal
        Z, rank_k = prox_nuclear(X + U / rho, mu / rho)

        # U-update (scaled dual)
        U += rho * (X - Z)

        # Convergence
        primal_res = frobenius(X - Z) / M_norm
        dual_res = frobenius(rho * (Z - Z_old)) / M_norm

        hist.primal_residual.append(primal_res)
        hist.dual_residual.append(dual_res)
        hist.rank.append(rank_k)

        if primal_res < tol and dual_res < tol:
            break

    hist.iterations = k + 1
    return Z, hist


# ──────────────────────────────────────────────────────────────
#  Iterative Hankel-ADMM Coupling
# ──────────────────────────────────────────────────────────────

def iterative_hankel_admm(
    M: NDArray,
    Omega: NDArray | None = None,
    L: int | None = None,
    ssa_rank: int | None = None,
    lam: float | None = None,
    rho: float = 1.5,
    admm_max_iter: int = 200,
    admm_tol: float = 1e-5,
    outer_iters: int = 3,
    outer_tol: float = 1e-4,
) -> tuple[NDArray, NDArray, ADMMHistory, list[float]]:
    """Iterative Hankel-SSA ↔ ADMM-RPCA coupling.

    Unlike a single serial pass (SSA → ADMM), this alternates between:
      1) Hankel-SSA reconstruction (impute + denoise via rank constraint)
      2) ADMM-RPCA decomposition   (low-rank + sparse separation)

    feeding the ADMM low-rank output back into SSA for refinement.
    The outer loop converges when the Frobenius-norm change in the
    low-rank component falls below `outer_tol`.

    Parameters
    ----------
    M         : (n_assets, T) raw data (may contain NaN)
    Omega     : observation mask (True = observed)
    L         : Hankel embedding window length
    ssa_rank  : SSA truncation rank per asset
    lam       : ADMM sparse penalty
    rho       : initial ADMM penalty parameter
    admm_max_iter : ADMM iteration cap per outer step
    admm_tol  : ADMM inner convergence tolerance
    outer_iters : maximum outer iterations
    outer_tol : convergence threshold on ‖X_new - X_old‖_F / ‖X_old‖_F

    Returns
    -------
    X_clean      : (n_assets, T) low-rank component
    S_anomaly    : (n_assets, T) sparse anomaly component
    admm_hist    : ADMMHistory from the final inner ADMM run
    outer_deltas : list of per-outer-iteration relative change values
    """
    from hankel_pipeline import multi_asset_hankel

    M = np.asarray(M, dtype=np.float64)
    n, T = M.shape

    if Omega is None:
        Omega = ~np.isnan(M)

    if L is None:
        L = max(2, T // 4)

    # Initialise: first SSA pass
    X_ssa = multi_asset_hankel(M, L=L, rank=ssa_rank)

    outer_deltas: list[float] = []
    X_prev = np.zeros_like(M)
    S_hat = np.zeros_like(M)
    hist = ADMMHistory()

    for outer_k in range(outer_iters):
        # ADMM-RPCA on the SSA-smoothed matrix
        X_hat, S_hat, hist = admm_rpca(
            X_ssa,
            Omega=Omega,
            lam=lam,
            rho=rho,
            max_iter=admm_max_iter,
            tol=admm_tol,
        )

        # Convergence check on the low-rank component
        delta_norm = frobenius(X_hat - X_prev)
        ref_norm = max(frobenius(X_prev), 1e-10)
        relative_delta = delta_norm / ref_norm
        outer_deltas.append(relative_delta)

        if outer_k > 0 and relative_delta < outer_tol:
            break

        X_prev = X_hat.copy()

        # Re-inject ADMM low-rank output into SSA for next round
        # Replace observed-but-anomalous entries with ADMM estimate,
        # keep original observations where no anomaly was detected
        M_refined = M.copy()
        anomaly_mask = np.abs(S_hat) > np.std(S_hat) * 0.5
        M_refined[anomaly_mask & Omega] = X_hat[anomaly_mask & Omega]
        M_refined[~Omega] = X_hat[~Omega]  # fill missing with current estimate

        X_ssa = multi_asset_hankel(M_refined, L=L, rank=ssa_rank)

    return X_hat, S_hat, hist, outer_deltas
