"""
backtest.py
===========
Rolling-window cross-validation backtester for the Hankel-ADMM
reconstruction pipeline.

Splits a multi-asset price matrix into temporally ordered
train/test windows and evaluates out-of-sample reconstruction
RMSE on each held-out segment.  This measures generalisation
quality and detects overfitting to in-sample noise patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from hankel_pipeline import multi_asset_hankel
from admm_solver import admm_rpca
from data_engine import fragment_data


@dataclass
class BacktestReport:
    """Aggregate cross-validation results."""
    n_windows: int = 0
    window_rmses: NDArray = field(default_factory=lambda: np.array([]))
    train_rmses: NDArray = field(default_factory=lambda: np.array([]))
    mean_rmse: float = 0.0
    std_rmse: float = 0.0
    worst_rmse: float = 0.0
    best_rmse: float = 0.0
    overfit_ratio: float = 0.0   # mean(test_rmse) / mean(train_rmse)


class RollingBacktester:
    """Expanding / sliding-window backtester for Hankel-ADMM.

    Parameters
    ----------
    n_windows       : number of cross-validation splits
    test_fraction   : fraction of each window used for held-out evaluation
    missing_ratio   : simulated data fragmentation ratio per window
    L               : Hankel embedding length
    ssa_rank        : SSA truncation rank
    admm_rho        : ADMM penalty parameter
    admm_max_iter   : ADMM iteration cap
    admm_tol        : ADMM convergence tolerance
    """

    def __init__(
        self,
        n_windows: int = 5,
        test_fraction: float = 0.2,
        missing_ratio: float = 0.15,
        L: int | None = None,
        ssa_rank: int | None = None,
        admm_rho: float = 1.5,
        admm_max_iter: int = 200,
        admm_tol: float = 1e-5,
    ):
        self.n_windows = n_windows
        self.test_fraction = test_fraction
        self.missing_ratio = missing_ratio
        self.L = L
        self.ssa_rank = ssa_rank
        self.admm_rho = admm_rho
        self.admm_max_iter = admm_max_iter
        self.admm_tol = admm_tol

    def run(self, X_clean: NDArray, seed: int = 42) -> BacktestReport:
        """Execute rolling-window cross-validation.

        Parameters
        ----------
        X_clean : (n_assets, T) ground-truth (or best-estimate) matrix
        seed    : base RNG seed for fragmentation

        Returns
        -------
        BacktestReport with per-window diagnostics
        """
        n_assets, T = X_clean.shape
        test_len = max(16, int(T * self.test_fraction))
        stride = max(1, (T - test_len) // max(self.n_windows, 1))

        window_rmses = []
        train_rmses = []
        rng = np.random.default_rng(seed)

        for w in range(self.n_windows):
            # Define temporal boundaries
            test_start = min(w * stride + (T - test_len - stride * (self.n_windows - 1)),
                             T - test_len)
            test_start = max(0, test_start)
            test_end = test_start + test_len
            train_end = test_start
            if train_end < 32:
                train_end = max(32, T // 4)
                test_start = train_end
                test_end = min(test_start + test_len, T)

            # Train segment
            X_train = X_clean[:, :train_end].copy()

            # Fragment training data
            X_frag, Omega = fragment_data(
                X_train, missing_ratio=self.missing_ratio,
                seed=seed + w,
            )

            # Reconstruct on train
            L = self.L if self.L else max(2, train_end // 4)
            L = min(L, train_end - 1)
            X_ssa = multi_asset_hankel(X_frag, L=L, rank=self.ssa_rank)
            X_hat, _, _ = admm_rpca(
                X_ssa, Omega=Omega, rho=self.admm_rho,
                max_iter=self.admm_max_iter, tol=self.admm_tol,
            )

            # Train RMSE
            train_rmse = float(np.sqrt(np.mean((X_clean[:, :train_end] - X_hat) ** 2)))
            train_rmses.append(train_rmse)

            # Test: use the trained pipeline's last-state to "predict" the test window
            # In practice, we reconstruct the full window and measure test error
            if test_end > train_end:
                full_seg = X_clean[:, :test_end].copy()
                # Fragment the test portion only
                full_frag = full_seg.copy()
                test_mask = rng.random((n_assets, test_end - train_end)) > self.missing_ratio
                full_frag[:, train_end:test_end] = np.where(
                    test_mask, full_seg[:, train_end:test_end], np.nan,
                )

                L_full = min(L, test_end - 1)
                X_ssa_full = multi_asset_hankel(full_frag, L=L_full, rank=self.ssa_rank)
                Omega_full = ~np.isnan(full_frag)
                X_hat_full, _, _ = admm_rpca(
                    X_ssa_full, Omega=Omega_full, rho=self.admm_rho,
                    max_iter=self.admm_max_iter, tol=self.admm_tol,
                )
                test_rmse = float(np.sqrt(np.mean(
                    (X_clean[:, train_end:test_end] - X_hat_full[:, train_end:test_end]) ** 2
                )))
            else:
                test_rmse = train_rmse

            window_rmses.append(test_rmse)

        window_rmses = np.array(window_rmses)
        train_rmses = np.array(train_rmses)

        mean_train = float(np.mean(train_rmses)) if len(train_rmses) > 0 else 1e-8
        overfit_ratio = float(np.mean(window_rmses)) / max(mean_train, 1e-8)

        return BacktestReport(
            n_windows=self.n_windows,
            window_rmses=window_rmses,
            train_rmses=train_rmses,
            mean_rmse=float(np.mean(window_rmses)),
            std_rmse=float(np.std(window_rmses)),
            worst_rmse=float(np.max(window_rmses)),
            best_rmse=float(np.min(window_rmses)),
            overfit_ratio=overfit_ratio,
        )
