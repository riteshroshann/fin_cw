"""
stat_testing.py
===============
Statistical estimation and hypothesis-testing module for validating
reconstruction quality and assessing anomaly significance.

Implements bootstrap confidence intervals, normality tests,
Granger-causality F-tests, anomaly significance tests, and
multiple-testing corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class TestResult:
    """Container for a single hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    reject_null: bool
    alpha: float
    detail: str = ""


# ──────────────────────────────────────────────────────────────
#  Reconstruction Confidence Intervals (bootstrap)
# ──────────────────────────────────────────────────────────────

def confidence_interval_reconstruction(
    x_true: NDArray,
    x_hat: NDArray,
    alpha: float = 0.05,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for reconstruction RMSE.

    Returns (rmse, ci_lo, ci_hi) at the (1 - alpha) confidence level.
    """
    x_true = np.asarray(x_true).ravel()
    x_hat = np.asarray(x_hat).ravel()
    errors = (x_true - x_hat) ** 2
    n = errors.size

    rng = np.random.default_rng(seed)
    boot_rmse = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        boot_rmse[b] = np.sqrt(np.mean(errors[idx]))

    lo = float(np.percentile(boot_rmse, 100 * alpha / 2))
    hi = float(np.percentile(boot_rmse, 100 * (1 - alpha / 2)))
    rmse = float(np.sqrt(np.mean(errors)))
    return rmse, lo, hi


# ──────────────────────────────────────────────────────────────
#  Jarque–Bera Normality Test
# ──────────────────────────────────────────────────────────────

def jarque_bera_test(
    residuals: NDArray,
    alpha: float = 0.05,
) -> TestResult:
    """Jarque–Bera test for normality of residuals.

    Under H₀ (normality), J-B ~ χ²(2).
    """
    r = np.asarray(residuals).ravel()
    n = r.size
    mu = np.mean(r)
    sigma = np.std(r, ddof=0)
    if sigma < 1e-15:
        return TestResult("Jarque-Bera", 0.0, 1.0, False, alpha,
                          "Zero variance — undefined")

    z = (r - mu) / sigma
    S = float(np.mean(z ** 3))          # skewness
    K = float(np.mean(z ** 4)) - 3.0    # excess kurtosis

    jb = n / 6.0 * (S ** 2 + K ** 2 / 4.0)

    # χ²(2) survival function approximation via incomplete gamma
    p_value = float(np.exp(-jb / 2.0) * (1 + jb / 2.0))
    p_value = min(max(p_value, 0.0), 1.0)

    return TestResult(
        test_name="Jarque-Bera",
        statistic=jb,
        p_value=p_value,
        reject_null=(p_value < alpha),
        alpha=alpha,
        detail=f"skew={S:.4f}, kurtosis={K:.4f}",
    )


# ──────────────────────────────────────────────────────────────
#  Granger Causality (F-test)
# ──────────────────────────────────────────────────────────────

def granger_causality_f_test(
    x: NDArray,
    y: NDArray,
    max_lag: int = 5,
    alpha: float = 0.05,
) -> list[TestResult]:
    """F-test for Granger causality: does x Granger-cause y?

    For each lag L in [1, max_lag]:
        Restricted:   y_t = c + Σ a_i y_{t-i}
        Unrestricted: y_t = c + Σ a_i y_{t-i} + Σ b_j x_{t-j}
    under H₀: all b_j = 0.

    Returns a list of TestResult, one per lag.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    T = min(x.size, y.size)
    results = []

    for lag in range(1, max_lag + 1):
        # Build matrices
        n_obs = T - lag
        Y = y[lag:T]

        # Restricted: lags of y only
        Z_r = np.column_stack([
            np.ones(n_obs),
            *[y[lag - i - 1:T - i - 1] for i in range(lag)]
        ])

        # Unrestricted: lags of y + lags of x
        Z_u = np.column_stack([
            Z_r,
            *[x[lag - i - 1:T - i - 1] for i in range(lag)]
        ])

        # OLS residuals
        beta_r = np.linalg.lstsq(Z_r, Y, rcond=None)[0]
        beta_u = np.linalg.lstsq(Z_u, Y, rcond=None)[0]

        rss_r = float(np.sum((Y - Z_r @ beta_r) ** 2))
        rss_u = float(np.sum((Y - Z_u @ beta_u) ** 2))

        p_r = Z_r.shape[1]
        p_u = Z_u.shape[1]
        df1 = p_u - p_r  # = lag
        df2 = n_obs - p_u

        if rss_u < 1e-15 or df2 <= 0:
            f_stat = 0.0
            p_val = 1.0
        else:
            f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
            # F-distribution survival approximation (Satterthwaite)
            p_val = _f_survival(f_stat, df1, df2)

        results.append(TestResult(
            test_name=f"Granger-F (lag={lag})",
            statistic=f_stat,
            p_value=p_val,
            reject_null=(p_val < alpha),
            alpha=alpha,
            detail=f"df1={df1}, df2={df2}",
        ))

    return results


def _f_survival(f: float, d1: int, d2: int) -> float:
    """Approximate upper-tail probability of F(d1, d2).

    Uses the regularised incomplete beta function relation:
        P(F > f) = I_{d2/(d2+d1*f)}(d2/2, d1/2)

    Falls back to a simple Gaussian approximation for large df.
    """
    if f <= 0:
        return 1.0
    x = d2 / (d2 + d1 * f)
    # Log-beta via Stirling for moderate df
    try:
        from math import lgamma, exp
        a, b = d2 / 2.0, d1 / 2.0
        # Approximate I_x(a,b) via continued fraction is complex;
        # use a rough normal approximation instead.
        # Abramowitz & Stegun 26.9.2: F → z transform
        z = (f ** (1.0 / 3.0) * (1 - 2.0 / (9 * d2))
             - (1 - 2.0 / (9 * d1))) / (
            (2.0 / (9 * d1) + f ** (2.0 / 3.0) * 2.0 / (9 * d2)) ** 0.5 + 1e-15
        )
        # Φ(-z) via error function
        from math import erfc
        p = 0.5 * erfc(z / 2 ** 0.5)
        return max(min(p, 1.0), 0.0)
    except Exception:
        return 0.5  # fallback


# ──────────────────────────────────────────────────────────────
#  Anomaly Significance Test
# ──────────────────────────────────────────────────────────────

def anomaly_significance_test(
    S: NDArray,
    null_sigma: float | None = None,
    alpha: float = 0.05,
) -> tuple[NDArray, NDArray, int]:
    """Test whether detected sparse anomalies exceed the noise floor.

    Each entry S[i,j] is tested against H₀: |S[i,j]| ~ N(0, σ²).
    The null standard deviation σ is estimated from the data if not provided
    (MAD estimator).

    Returns
    -------
    significant_mask : boolean matrix of significant anomalies
    p_values         : matrix of per-entry p-values
    n_significant    : total count of significant anomalies
    """
    S = np.asarray(S, dtype=np.float64)
    abs_S = np.abs(S)

    if null_sigma is None:
        # Robust scale estimate via MAD
        nonzero = abs_S[abs_S > 1e-12]
        if nonzero.size > 0:
            null_sigma = float(np.median(nonzero)) / 0.6745
        else:
            null_sigma = 1.0

    null_sigma = max(null_sigma, 1e-15)

    # Two-sided p-values under Gaussian null
    z_scores = abs_S / null_sigma
    from math import erfc as _erfc
    p_values = np.vectorize(lambda z: min(_erfc(z / np.sqrt(2)), 1.0))(z_scores)

    significant = p_values < alpha
    return significant, p_values, int(np.sum(significant))


# ──────────────────────────────────────────────────────────────
#  Multiple Testing Correction
# ──────────────────────────────────────────────────────────────

def multiple_testing_correction(
    p_values: NDArray,
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> tuple[NDArray, NDArray]:
    """Apply multiple-testing correction.

    Parameters
    ----------
    p_values : 1-D array of raw p-values
    method   : 'bonferroni' or 'bh' (Benjamini–Hochberg FDR)
    alpha    : significance level

    Returns
    -------
    adjusted_p : corrected p-values
    reject     : boolean mask for significant tests
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    m = p.size

    if method == "bonferroni":
        adjusted = np.minimum(p * m, 1.0)
        reject = adjusted < alpha

    elif method == "bh":
        # Benjamini–Hochberg procedure
        order = np.argsort(p)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, m + 1)

        adjusted = np.minimum(p * m / ranks, 1.0)
        # Enforce monotonicity (step-up)
        adj_sorted = adjusted[order]
        for i in range(m - 2, -1, -1):
            adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
        adjusted[order] = adj_sorted
        reject = adjusted < alpha

    else:
        raise ValueError(f"Unknown correction method: {method}")

    return adjusted, reject
