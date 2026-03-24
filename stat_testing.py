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

    # Proper χ²(2) survival function
    try:
        from scipy.stats import chi2
        p_value = float(chi2.sf(jb, df=2))
    except ImportError:
        # Fallback: χ²(2) sf = exp(-x/2), exact for 2 degrees of freedom
        p_value = float(np.exp(-jb / 2.0))
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
    """Upper-tail probability of F(d1, d2).

    Uses scipy.stats.f.sf for exact computation. Falls back to
    Abramowitz & Stegun §26.9.2 normal approximation if scipy
    is unavailable.
    """
    if f <= 0:
        return 1.0
    try:
        from scipy.stats import f as f_dist
        return float(f_dist.sf(f, d1, d2))
    except ImportError:
        pass
    # Fallback: Abramowitz & Stegun normal approximation
    try:
        from math import erfc
        z = (f ** (1.0 / 3.0) * (1 - 2.0 / (9 * d2))
             - (1 - 2.0 / (9 * d1))) / (
            (2.0 / (9 * d1) + f ** (2.0 / 3.0) * 2.0 / (9 * d2)) ** 0.5 + 1e-15
        )
        p = 0.5 * erfc(z / 2 ** 0.5)
        return max(min(p, 1.0), 0.0)
    except Exception:
        return 0.5


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


# ──────────────────────────────────────────────────────────────
#  Anomaly Classification
# ──────────────────────────────────────────────────────────────

@dataclass
class AnomalyClassification:
    """Classification of detected anomalies by temporal signature."""
    flash_crash: int = 0           # isolated spikes (single-entry)
    wash_trade: int = 0            # flat plateaus (consecutive entries)
    latency_gap: int = 0           # clustered missing (NaN blocks)
    unclassified: int = 0
    per_asset: dict = None         # type: ignore

    def __post_init__(self):
        if self.per_asset is None:
            self.per_asset = {}


def classify_anomalies(
    S: NDArray,
    significant_mask: NDArray,
    run_length_threshold: int = 3,
) -> AnomalyClassification:
    """Classify significant anomalies by their spatio-temporal pattern.

    Classification rules:
        - flash_crash: isolated significant entries (≤2 consecutive)
        - wash_trade: runs of ≥ `run_length_threshold` consecutive
                      significant entries along the time axis
        - latency_gap: runs where the magnitude is near-constant
                       (std/mean < 0.1), suggesting missing-data artefacts

    Parameters
    ----------
    S                     : (n, T) sparse anomaly matrix
    significant_mask      : (n, T) boolean mask from BH correction
    run_length_threshold  : minimum consecutive entries for wash-trade

    Returns
    -------
    AnomalyClassification with counts and per-asset breakdown
    """
    S = np.asarray(S)
    significant_mask = np.asarray(significant_mask, dtype=bool)
    n, T = S.shape

    result = AnomalyClassification()

    for i in range(n):
        row_mask = significant_mask[i]
        row_S = np.abs(S[i])
        asset_counts = {"flash_crash": 0, "wash_trade": 0, "latency_gap": 0}

        # Find runs of consecutive significant entries
        j = 0
        while j < T:
            if not row_mask[j]:
                j += 1
                continue
            # Start of a run
            run_start = j
            while j < T and row_mask[j]:
                j += 1
            run_len = j - run_start
            run_vals = row_S[run_start:j]

            if run_len >= run_length_threshold:
                # Check if flat (wash-trade) or variable
                run_mean = np.mean(run_vals)
                run_std = np.std(run_vals)
                if run_mean > 1e-12 and run_std / run_mean < 0.1:
                    asset_counts["latency_gap"] += run_len
                    result.latency_gap += run_len
                else:
                    asset_counts["wash_trade"] += run_len
                    result.wash_trade += run_len
            else:
                asset_counts["flash_crash"] += run_len
                result.flash_crash += run_len

        result.per_asset[i] = asset_counts

    result.unclassified = max(
        0, int(np.sum(significant_mask)) - result.flash_crash
        - result.wash_trade - result.latency_gap
    )
    return result


# ──────────────────────────────────────────────────────────────
#  Reconstruction Bias-Variance Decomposition
# ──────────────────────────────────────────────────────────────

def reconstruction_bias_variance(
    x_true: NDArray,
    x_hat: NDArray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap decomposition of MSE into bias² + variance.

    Uses the identity: MSE = Bias² + Var(x̂) + irreducible noise.

    Returns
    -------
    dict with keys: 'mse', 'bias_squared', 'variance', 'rmse'
    """
    x_true = np.asarray(x_true).ravel()
    x_hat = np.asarray(x_hat).ravel()
    n = x_true.size

    rng = np.random.default_rng(seed)
    boot_means = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        boot_means[b] = np.mean(x_hat[idx])

    bias = float(np.mean(x_hat) - np.mean(x_true))
    variance = float(np.var(boot_means))
    mse = float(np.mean((x_true - x_hat) ** 2))

    return {
        "mse": mse,
        "bias_squared": bias ** 2,
        "variance": variance,
        "rmse": float(np.sqrt(mse)),
    }
