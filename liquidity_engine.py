"""
liquidity_engine.py
===================
Smart-Liquidity Engine — the orchestrator that ties Hankel reconstruction,
ADMM-RPCA decomposition, DMD dynamics analysis, and statistical anomaly
forensics into a single coherent pipeline.

Usage:
    engine = LiquidityEngine(L=128, admm_rho=1.5, dmd_rank=12)
    engine.ingest(M_raw)
    engine.reconstruct()
    engine.analyze_dynamics()
    engine.detect_anomalies()
    weights_new = engine.rebalance_portfolio(current_weights)
    report = engine.generate_report()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hankel_pipeline import multi_asset_hankel, spectral_filter
from admm_solver import admm_rpca, ADMMHistory
from dmd_engine import exact_dmd, predict_dmd, dmd_stability_analysis, StabilityReport
from stat_testing import (
    confidence_interval_reconstruction,
    jarque_bera_test,
    granger_causality_f_test,
    anomaly_significance_test,
    multiple_testing_correction,
    TestResult,
)


@dataclass
class PipelineTiming:
    """Wall-clock timing for each pipeline stage (seconds)."""
    hankel_ssa: float = 0.0
    admm_rpca: float = 0.0
    dmd: float = 0.0
    stat_tests: float = 0.0
    granger: float = 0.0
    rebalance: float = 0.0
    total: float = 0.0


class LiquidityEngine:
    """End-to-end pipeline for autonomous liquidity reconstruction and
    anomaly forensics in fragmented FinTech ecosystems."""

    def __init__(
        self,
        L: int | None = None,
        ssa_rank: int | None = None,
        admm_lam: float | None = None,
        admm_rho: float = 1.5,
        admm_max_iter: int = 300,
        admm_tol: float = 1e-5,
        dmd_rank: int | None = None,
        alpha: float = 0.05,
    ):
        # Hyperparameters
        self.L = L
        self.ssa_rank = ssa_rank
        self.admm_lam = admm_lam
        self.admm_rho = admm_rho
        self.admm_max_iter = admm_max_iter
        self.admm_tol = admm_tol
        self.dmd_rank = dmd_rank
        self.alpha = alpha

        # State
        self.M_raw: NDArray | None = None
        self.X_ssa: NDArray | None = None          # after Hankel-SSA
        self.X_clean: NDArray | None = None         # low-rank from ADMM
        self.S_anomaly: NDArray | None = None       # sparse from ADMM
        self.admm_hist: ADMMHistory | None = None
        self.dmd_modes: NDArray | None = None
        self.dmd_eigenvalues: NDArray | None = None
        self.dmd_amplitudes: NDArray | None = None
        self.stability: StabilityReport | None = None
        self.X_forecast: NDArray | None = None
        self.reconstruction_ci: tuple | None = None
        self.normality_test: TestResult | None = None
        self.anomaly_mask: NDArray | None = None
        self.anomaly_pvalues: NDArray | None = None
        self.n_anomalies: int = 0
        self.granger_adj: NDArray | None = None   # (n, n) adjacency matrix
        self.granger_fstats: NDArray | None = None # (n, n) F-statistics
        self.granger_pvals: NDArray | None = None  # (n, n) FDR-corrected p-vals
        self.portfolio_weights: NDArray | None = None
        self.timing = PipelineTiming()

    # ──────────────────────────────────────────────────────────
    #  Stage 1: Ingest raw data
    # ──────────────────────────────────────────────────────────

    def ingest(self, M_raw: NDArray) -> None:
        """Accept raw fragmented multi-asset price matrix.

        M_raw : (n_assets, T), may contain NaN for missing values.
        """
        self.M_raw = np.asarray(M_raw, dtype=np.float64).copy()

    # ──────────────────────────────────────────────────────────
    #  Stage 2: Reconstruct
    # ──────────────────────────────────────────────────────────

    def reconstruct(self) -> None:
        """Run Hankel-SSA → ADMM-RPCA pipeline.

        1) Multi-asset Hankel-SSA: impute NaN + denoise
        2) ADMM-RPCA: decompose into low-rank X and sparse S
        """
        if self.M_raw is None:
            raise RuntimeError("Call ingest() first")

        # — Hankel SSA —
        t0 = time.perf_counter()
        self.X_ssa = multi_asset_hankel(
            self.M_raw, L=self.L, rank=self.ssa_rank,
        )
        self.timing.hankel_ssa = time.perf_counter() - t0

        # — ADMM RPCA —
        t0 = time.perf_counter()
        Omega = ~np.isnan(self.M_raw)
        self.X_clean, self.S_anomaly, self.admm_hist = admm_rpca(
            self.X_ssa,
            Omega=Omega,
            lam=self.admm_lam,
            rho=self.admm_rho,
            max_iter=self.admm_max_iter,
            tol=self.admm_tol,
        )
        self.timing.admm_rpca = time.perf_counter() - t0

    # ──────────────────────────────────────────────────────────
    #  Stage 3: Analyse dynamics via DMD
    # ──────────────────────────────────────────────────────────

    def analyze_dynamics(self, forecast_steps: int = 64) -> None:
        """Run DMD on cleaned data; compute stability and forecast."""
        if self.X_clean is None:
            raise RuntimeError("Call reconstruct() first")

        t0 = time.perf_counter()
        Phi, lam, b, _ = exact_dmd(self.X_clean, rank=self.dmd_rank)
        self.dmd_modes = Phi
        self.dmd_eigenvalues = lam
        self.dmd_amplitudes = b

        self.stability = dmd_stability_analysis(lam)
        T_existing = self.X_clean.shape[1]
        self.X_forecast = predict_dmd(Phi, lam, b, T_existing + forecast_steps)
        self.timing.dmd = time.perf_counter() - t0

    # ──────────────────────────────────────────────────────────
    #  Stage 4: Statistical anomaly detection
    # ──────────────────────────────────────────────────────────

    def detect_anomalies(self) -> None:
        """Run statistical tests on the sparse component S."""
        if self.S_anomaly is None:
            raise RuntimeError("Call reconstruct() first")

        t0 = time.perf_counter()

        # Anomaly significance
        mask, pvals, n_sig = anomaly_significance_test(
            self.S_anomaly, alpha=self.alpha,
        )
        # Multiple-testing correction (flatten → correct → reshape)
        flat_p = pvals.ravel()
        adj_p, reject = multiple_testing_correction(flat_p, method="bh", alpha=self.alpha)
        self.anomaly_mask = reject.reshape(pvals.shape)
        self.anomaly_pvalues = adj_p.reshape(pvals.shape)
        self.n_anomalies = int(np.sum(self.anomaly_mask))

        # Reconstruction quality (compare SSA output to ADMM-cleaned)
        flat_ssa = self.X_ssa.ravel()
        flat_clean = self.X_clean.ravel()
        self.reconstruction_ci = confidence_interval_reconstruction(
            flat_ssa, flat_clean, alpha=self.alpha,
        )

        # Normality of residuals
        residuals = (self.X_ssa - self.X_clean).ravel()
        self.normality_test = jarque_bera_test(residuals, alpha=self.alpha)

        self.timing.stat_tests = time.perf_counter() - t0

    # ──────────────────────────────────────────────────────────
    #  Stage 4b: Granger causality lead-lag network
    # ──────────────────────────────────────────────────────────

    def analyze_lead_lag(
        self,
        max_lag: int = 5,
        alpha: float | None = None,
    ) -> None:
        """Build pairwise Granger causality adjacency matrix.

        For every ordered asset pair (i, j), tests whether asset i
        Granger-causes asset j.  All p-values are FDR-corrected
        (Benjamini–Hochberg) to control for multiple comparisons.

        Stores:
            granger_adj   : (n, n) boolean — True if link is significant
            granger_fstats: (n, n) best F-statistic across lags
            granger_pvals : (n, n) FDR-corrected p-value (best lag)
        """
        if self.X_clean is None:
            raise RuntimeError("Call reconstruct() first")

        t0 = time.perf_counter()
        alpha_gc = alpha if alpha is not None else self.alpha
        n = self.X_clean.shape[0]

        fstats = np.zeros((n, n), dtype=np.float64)
        raw_p = np.ones((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                results = granger_causality_f_test(
                    self.X_clean[i], self.X_clean[j],
                    max_lag=max_lag, alpha=alpha_gc,
                )
                # Pick lag with smallest p-value (strongest evidence)
                best = min(results, key=lambda r: r.p_value)
                fstats[i, j] = best.statistic
                raw_p[i, j] = best.p_value

        # FDR correction on off-diagonal entries
        off_diag_mask = ~np.eye(n, dtype=bool)
        flat_p = raw_p[off_diag_mask]
        adj_p, reject = multiple_testing_correction(flat_p, method="bh", alpha=alpha_gc)

        corrected_p = np.ones((n, n), dtype=np.float64)
        corrected_p[off_diag_mask] = adj_p
        adj_matrix = np.zeros((n, n), dtype=bool)
        adj_matrix[off_diag_mask] = reject

        self.granger_adj = adj_matrix
        self.granger_fstats = fstats
        self.granger_pvals = corrected_p
        self.timing.granger = time.perf_counter() - t0

    # ──────────────────────────────────────────────────────────
    #  Stage 5: Portfolio rebalancing
    # ──────────────────────────────────────────────────────────

    def rebalance_portfolio(
        self,
        current_weights: NDArray | None = None,
        risk_aversion: float = 2.0,
    ) -> NDArray:
        """Mean-variance rebalance using cleaned covariance.

        If current_weights is None, starts from equal-weight.
        Returns new optimal weights (long-only, sum-to-one).
        """
        if self.X_clean is None:
            raise RuntimeError("Call reconstruct() first")

        t0 = time.perf_counter()
        n = self.X_clean.shape[0]

        # Log-returns from cleaned prices
        prices = self.X_clean
        returns = np.diff(np.log(np.maximum(prices, 1e-8)), axis=1)

        mu = np.mean(returns, axis=1)
        Sigma = np.cov(returns) + 1e-6 * np.eye(n)

        # Mean-variance: w* = (1/γ) Σ⁻¹ μ  (unconstrained)
        Sigma_inv = np.linalg.inv(Sigma)
        w_star = Sigma_inv @ mu / risk_aversion

        # Long-only projection + normalisation
        w_star = np.maximum(w_star, 0.0)
        w_sum = w_star.sum()
        if w_sum > 1e-12:
            w_star /= w_sum
        else:
            w_star = np.ones(n) / n

        self.portfolio_weights = w_star
        self.timing.rebalance = time.perf_counter() - t0
        return w_star

    # ──────────────────────────────────────────────────────────
    #  Report generation
    # ──────────────────────────────────────────────────────────

    def generate_report(self) -> dict[str, Any]:
        """Return structured forensics report."""
        self.timing.total = (
            self.timing.hankel_ssa + self.timing.admm_rpca +
            self.timing.dmd + self.timing.stat_tests +
            self.timing.granger + self.timing.rebalance
        )

        report: dict[str, Any] = {
            "pipeline_timing": {
                "hankel_ssa_s": round(self.timing.hankel_ssa, 4),
                "admm_rpca_s": round(self.timing.admm_rpca, 4),
                "dmd_s": round(self.timing.dmd, 4),
                "stat_tests_s": round(self.timing.stat_tests, 4),
                "granger_s": round(self.timing.granger, 4),
                "rebalance_s": round(self.timing.rebalance, 4),
                "total_s": round(self.timing.total, 4),
            },
        }

        if self.admm_hist is not None:
            report["admm"] = {
                "iterations": self.admm_hist.iterations,
                "final_primal_residual": self.admm_hist.primal_residual[-1] if self.admm_hist.primal_residual else None,
                "final_dual_residual": self.admm_hist.dual_residual[-1] if self.admm_hist.dual_residual else None,
                "final_rank": self.admm_hist.rank[-1] if self.admm_hist.rank else None,
            }

        if self.stability is not None:
            report["dmd_stability"] = {
                "spectral_radius": round(self.stability.spectral_radius, 6),
                "dominant_frequency_hz": round(self.stability.dominant_frequency, 6),
                "n_stable": int(self.stability.stable_mask.sum()),
                "n_marginal": int(self.stability.marginal_mask.sum()),
                "n_unstable": int(self.stability.unstable_mask.sum()),
            }

        if self.reconstruction_ci is not None:
            rmse, ci_lo, ci_hi = self.reconstruction_ci
            report["reconstruction"] = {
                "rmse": round(rmse, 6),
                "ci_lo": round(ci_lo, 6),
                "ci_hi": round(ci_hi, 6),
                "alpha": self.alpha,
            }

        if self.normality_test is not None:
            report["normality_test"] = {
                "test": self.normality_test.test_name,
                "statistic": round(self.normality_test.statistic, 4),
                "p_value": round(self.normality_test.p_value, 6),
                "reject_null": self.normality_test.reject_null,
                "detail": self.normality_test.detail,
            }

        report["anomalies"] = {
            "n_detected": self.n_anomalies,
            "alpha": self.alpha,
            "correction": "Benjamini-Hochberg",
        }

        if self.granger_adj is not None:
            n_edges = int(np.sum(self.granger_adj))
            strongest_f = float(np.max(self.granger_fstats))
            # Find the (i,j) with largest F-stat
            if strongest_f > 0:
                idx = np.unravel_index(
                    np.argmax(self.granger_fstats), self.granger_fstats.shape,
                )
                strongest_pair = (int(idx[0]), int(idx[1]))
            else:
                strongest_pair = (-1, -1)
            report["granger_network"] = {
                "n_edges": n_edges,
                "n_nodes": self.granger_adj.shape[0],
                "strongest_f_stat": round(strongest_f, 4),
                "strongest_pair": strongest_pair,
                "adjacency": self.granger_adj.astype(int).tolist(),
            }

        if self.portfolio_weights is not None:
            report["portfolio"] = {
                "weights": [round(float(w), 6) for w in self.portfolio_weights],
                "sum": round(float(self.portfolio_weights.sum()), 6),
            }

        return report
