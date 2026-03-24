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
from admm_solver import admm_rpca, iterative_hankel_admm, ADMMHistory
from dmd_engine import exact_dmd, predict_dmd, dmd_stability_analysis, StabilityReport
from stat_testing import (
    confidence_interval_reconstruction,
    jarque_bera_test,
    granger_causality_f_test,
    anomaly_significance_test,
    multiple_testing_correction,
    classify_anomalies,
    reconstruction_bias_variance,
    TestResult,
    AnomalyClassification,
)
from linalg_primitives import (
    tensor_unfold,
    multilinear_rank,
    shift_lag_analysis,
    nearest_kronecker_product,
    hankelise,
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
        self.anomaly_classification: AnomalyClassification | None = None
        self.granger_adj: NDArray | None = None   # (n, n) adjacency matrix
        self.granger_fstats: NDArray | None = None # (n, n) F-statistics
        self.granger_pvals: NDArray | None = None  # (n, n) FDR-corrected p-vals
        self.portfolio_weights: NDArray | None = None
        self.kronecker_A: NDArray | None = None
        self.kronecker_B: NDArray | None = None
        self.kronecker_approx_error: float | None = None
        self.forecast_validation: dict | None = None
        self.bias_variance: dict | None = None
        self.outer_deltas: list[float] | None = None
        self.tensor_ranks: tuple | None = None
        self.lag_analysis: tuple | None = None
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
        """Run Iterative Hankel-SSA ↔ ADMM-RPCA pipeline.

        Uses the iterative coupling (outer loop) that alternates between
        Hankel-SSA denoising and ADMM low-rank + sparse decomposition,
        feeding refined estimates back for progressive improvement.
        """
        if self.M_raw is None:
            raise RuntimeError("Call ingest() first")

        t0 = time.perf_counter()
        Omega = ~np.isnan(self.M_raw)

        self.X_clean, self.S_anomaly, self.admm_hist, self.outer_deltas = (
            iterative_hankel_admm(
                self.M_raw,
                Omega=Omega,
                L=self.L,
                ssa_rank=self.ssa_rank,
                lam=self.admm_lam,
                rho=self.admm_rho,
                admm_max_iter=self.admm_max_iter,
                admm_tol=self.admm_tol,
                outer_iters=3,
                outer_tol=1e-4,
            )
        )

        # Also store SSA-only output for diagnostics
        self.X_ssa = multi_asset_hankel(
            self.M_raw, L=self.L, rank=self.ssa_rank,
        )

        elapsed = time.perf_counter() - t0
        self.timing.hankel_ssa = elapsed * 0.4  # approximate split
        self.timing.admm_rpca = elapsed * 0.6

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

        # Anomaly classification
        self.anomaly_classification = classify_anomalies(
            self.S_anomaly, self.anomaly_mask,
        )

        # Reconstruction quality (compare SSA output to ADMM-cleaned)
        flat_ssa = self.X_ssa.ravel()
        flat_clean = self.X_clean.ravel()
        self.reconstruction_ci = confidence_interval_reconstruction(
            flat_ssa, flat_clean, alpha=self.alpha,
        )

        # Bias-variance decomposition
        self.bias_variance = reconstruction_bias_variance(
            flat_ssa, flat_clean,
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
    #  Stage 6: Kronecker covariance estimation
    # ──────────────────────────────────────────────────────────

    def estimate_kronecker_covariance(
        self,
        block_size: int | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Fit Σ ≈ A ⊗ B to cleaned return covariance.

        Uses Van Loan & Pitsianis rearrangement lemma for the
        nearest Kronecker product in Frobenius norm.

        A captures inter-sector correlations, B captures intra-sector.
        """
        if self.X_clean is None:
            raise RuntimeError("Call reconstruct() first")

        n = self.X_clean.shape[0]
        returns = np.diff(np.log(np.maximum(self.X_clean, 1e-8)), axis=1)
        Sigma = np.cov(returns) + 1e-6 * np.eye(n)

        # Determine block structure
        if block_size is None:
            block_size = max(2, n // 4)
        n_blocks = n // block_size
        effective_n = n_blocks * block_size

        if effective_n < n:
            # Trim to fit Kronecker structure
            Sigma_trimmed = Sigma[:effective_n, :effective_n]
        else:
            Sigma_trimmed = Sigma

        A, B = nearest_kronecker_product(Sigma_trimmed, n_blocks, block_size)

        # Compute approximation quality
        from linalg_primitives import kron, frobenius
        Sigma_approx = kron(A, B)
        self.kronecker_approx_error = float(
            frobenius(Sigma_trimmed - Sigma_approx) / frobenius(Sigma_trimmed)
        )
        self.kronecker_A = A
        self.kronecker_B = B

        return A, B

    # ──────────────────────────────────────────────────────────
    #  Stage 7: Forecast validation
    # ──────────────────────────────────────────────────────────

    def validate_forecast(
        self,
        test_fraction: float = 0.2,
    ) -> dict:
        """Split cleaned data into in-sample/out-of-sample, run DMD
        on in-sample, and measure forecast RMSE against actuals."""
        if self.X_clean is None:
            raise RuntimeError("Call reconstruct() first")

        n, T = self.X_clean.shape
        split = int(T * (1.0 - test_fraction))
        X_train = self.X_clean[:, :split]
        X_test = self.X_clean[:, split:]
        T_test = X_test.shape[1]

        if T_test < 2 or split < 10:
            self.forecast_validation = {"error": "Insufficient data for split"}
            return self.forecast_validation

        dmd_rank = min(self.dmd_rank or 10, n, split - 1)
        try:
            Phi, lam, b, _ = exact_dmd(X_train, rank=dmd_rank)
            X_pred = predict_dmd(Phi, lam, b, split + T_test)
            X_forecast = np.real(X_pred[:, split:split + T_test])

            forecast_rmse = float(np.sqrt(np.mean((X_test - X_forecast) ** 2)))
            per_asset_rmse = np.sqrt(np.mean((X_test - X_forecast) ** 2, axis=1))

            self.forecast_validation = {
                "test_steps": T_test,
                "train_steps": split,
                "forecast_rmse": round(forecast_rmse, 6),
                "per_asset_rmse": [round(float(r), 6) for r in per_asset_rmse],
                "X_test": X_test,
                "X_forecast": X_forecast,
            }
        except Exception as e:
            self.forecast_validation = {"error": str(e)}

        return self.forecast_validation

    # ──────────────────────────────────────────────────────────
    #  Stage 8: Tensor and structural analysis
    # ──────────────────────────────────────────────────────────

    def analyze_tensor_structure(self) -> None:
        """Construct a 3-D tensor from the multi-asset Hankel matrices
        and compute multilinear rank for structural analysis."""
        if self.X_clean is None:
            raise RuntimeError("Call reconstruct() first")

        n, T = self.X_clean.shape
        L = min(self.L or T // 4, T // 2)

        # Stack per-asset Hankel matrices into a 3-D tensor (n, L, K)
        K = T - L + 1
        if K < 1:
            return

        tensor = np.zeros((n, L, K))
        for i in range(n):
            tensor[i] = hankelise(self.X_clean[i], L)

        self.tensor_ranks = multilinear_rank(tensor)

    def analyze_lag_structure(self) -> None:
        """Run shift-matrix lag analysis on cleaned data."""
        if self.X_clean is None:
            raise RuntimeError("Call reconstruct() first")

        self.lag_analysis = shift_lag_analysis(self.X_clean, max_lag=20)

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

        # Iterative coupling info
        if self.outer_deltas is not None:
            report["iterative_coupling"] = {
                "outer_iterations": len(self.outer_deltas),
                "deltas": [round(d, 6) for d in self.outer_deltas],
                "converged": len(self.outer_deltas) < 3 or (
                    self.outer_deltas[-1] < 1e-4 if self.outer_deltas else True
                ),
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

        if self.bias_variance is not None:
            report["bias_variance"] = {
                k: round(v, 6) for k, v in self.bias_variance.items()
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

        if self.anomaly_classification is not None:
            ac = self.anomaly_classification
            report["anomaly_classification"] = {
                "flash_crash": ac.flash_crash,
                "wash_trade": ac.wash_trade,
                "latency_gap": ac.latency_gap,
                "unclassified": ac.unclassified,
            }

        if self.granger_adj is not None:
            n_edges = int(np.sum(self.granger_adj))
            strongest_f = float(np.max(self.granger_fstats))
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

        if self.kronecker_A is not None:
            report["kronecker_covariance"] = {
                "A_shape": list(self.kronecker_A.shape),
                "B_shape": list(self.kronecker_B.shape),
                "approx_error": round(self.kronecker_approx_error, 6),
                "A_diag": [round(float(self.kronecker_A[i, i]), 6)
                           for i in range(self.kronecker_A.shape[0])],
            }

        if self.forecast_validation is not None and "error" not in self.forecast_validation:
            fv = self.forecast_validation
            report["forecast_validation"] = {
                "test_steps": fv["test_steps"],
                "train_steps": fv["train_steps"],
                "forecast_rmse": fv["forecast_rmse"],
                "per_asset_rmse": fv["per_asset_rmse"],
            }

        if self.tensor_ranks is not None:
            report["tensor_info"] = {
                "multilinear_rank": list(self.tensor_ranks),
            }

        return report
