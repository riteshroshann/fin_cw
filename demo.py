"""
demo.py
=======
End-to-end demonstration and benchmark of the Robust Tensor-Completion
via Hankel-Iterative Proximal ADMM pipeline.

Uses REAL Kaggle LOB data (BTC/ETH/ADA) when available, falls back to
synthetic data generation for testing.

Usage:
    python demo.py              # auto-detect real data or use synthetic
    python demo.py --synthetic  # force synthetic mode
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Local modules ──
from data_engine import generate_lob, inject_anomalies, fragment_data
from kaggle_loader import (
    check_data_available,
    download_dataset,
    extract_midprices,
    build_liquidity_depth_matrix,
    compute_order_flow_imbalance,
    extract_lob_features,
    dataset_summary,
    build_augmented_matrix,
)
from liquidity_engine import LiquidityEngine
from backtest import RollingBacktester
from viz_forensics import (
    plot_reconstruction_comparison,
    plot_admm_convergence,
    plot_dmd_spectrum,
    plot_anomaly_heatmap,
    plot_portfolio_weights,
    plot_spectral_density,
    plot_multi_asset_reconstruction,
    plot_granger_network,
    plot_backtest_rmse,
)


# ═══════════════════════════════════════════════════════════════
#  Terminal formatting helpers
# ═══════════════════════════════════════════════════════════════

_B = "\033[1m"
_D = "\033[2m"
_CY = "\033[96m"
_GR = "\033[92m"
_YL = "\033[93m"
_RD = "\033[91m"
_MG = "\033[95m"
_BL = "\033[94m"
_RS = "\033[0m"
_LN = "─" * 72


def _header(title: str) -> None:
    print(f"\n{_CY}{_B}{'═' * 72}{_RS}")
    print(f"{_CY}{_B}  {title}{_RS}")
    print(f"{_CY}{_B}{'═' * 72}{_RS}\n")


def _section(title: str) -> None:
    print(f"\n  {_MG}{_B}▶ {title}{_RS}")
    print(f"  {_D}{_LN}{_RS}")


def _kv(key: str, value, color: str = _GR) -> None:
    print(f"    {_D}{key:.<42s}{_RS} {color}{value}{_RS}")


def _status(msg: str) -> None:
    print(f"  {_YL}⏳ {msg}{_RS}", end="", flush=True)


def _done(elapsed: float) -> None:
    print(f" {_GR}✓{_RS} {_D}({elapsed:.3f}s){_RS}")


# ═══════════════════════════════════════════════════════════════
#  Data loading: real Kaggle LOB or synthetic
# ═══════════════════════════════════════════════════════════════

def _load_real_data(max_rows: int = 4096) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load real Kaggle LOB mid-prices and inject controlled anomalies.

    Returns (X_true, X_frag, asset_names).
    """
    _section("Data Source — Kaggle High-Frequency Crypto LOB")
    print(dataset_summary())

    _status("Extracting mid-prices (1-min frequency)")
    t0 = time.perf_counter()
    midprices, names = extract_midprices(freq="1min", max_rows=max_rows)
    _done(time.perf_counter() - t0)

    n_assets, T = midprices.shape
    _kv("Assets", f"{names}")
    _kv("Time steps", T)
    for i, name in enumerate(names):
        _kv(f"  {name} range", f"[{midprices[i].min():.2f}, {midprices[i].max():.2f}]")

    # Inject anomalies for forensics testing
    _status("Injecting flash-crash anomalies (3%)")
    t0 = time.perf_counter()
    X_anomalous, S_ground = inject_anomalies(
        midprices, ratio=0.03, mode="flash_crash", magnitude=5.0, seed=123,
    )
    _done(time.perf_counter() - t0)
    n_injected = int(np.sum(np.abs(S_ground) > 1e-8))
    _kv("Injected anomalies", n_injected)

    # Fragment to simulate cross-venue gaps
    _status("Fragmenting data (15% missing)")
    t0 = time.perf_counter()
    X_frag, Omega = fragment_data(X_anomalous, missing_ratio=0.15, seed=77)
    _done(time.perf_counter() - t0)
    n_missing = int(np.sum(~Omega))
    _kv("Missing entries", f"{n_missing} ({100 * n_missing / X_frag.size:.1f}%)")

    return midprices, X_frag, names


def _load_synthetic_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic LOB data as fallback."""
    N_ASSETS = 8
    T = 512
    names = [f"SYN{i}" for i in range(N_ASSETS)]

    _section("Data Source — Synthetic LOB Generator (no Kaggle data found)")

    _status("Generating synthetic LOB mid-prices")
    t0 = time.perf_counter()
    X_true = generate_lob(n_assets=N_ASSETS, T=T, noise_sigma=0.08, seed=42)
    _done(time.perf_counter() - t0)
    _kv("Assets", N_ASSETS)
    _kv("Time steps", T)

    _status("Injecting flash-crash anomalies")
    t0 = time.perf_counter()
    X_anomalous, S_ground = inject_anomalies(
        X_true, ratio=0.03, mode="flash_crash", magnitude=6.0, seed=123,
    )
    _done(time.perf_counter() - t0)
    n_injected = int(np.sum(np.abs(S_ground) > 1e-8))
    _kv("Injected anomalies", n_injected)

    _status("Fragmenting data (15% missing)")
    t0 = time.perf_counter()
    X_frag, Omega = fragment_data(X_anomalous, missing_ratio=0.15, seed=77)
    _done(time.perf_counter() - t0)
    n_missing = int(np.sum(~Omega))
    _kv("Missing entries", f"{n_missing} ({100 * n_missing / X_frag.size:.1f}%)")

    return X_true, X_frag, names


# ═══════════════════════════════════════════════════════════════
#  LOB microstructure analysis (real data bonus)
# ═══════════════════════════════════════════════════════════════

def _analyze_microstructure() -> None:
    """Additional microstructure analysis when real data is available."""
    if not check_data_available():
        return

    _section("Bonus — LOB Microstructure Analysis")

    for asset_key, ticker in [("btc_usdt", "BTC"), ("eth_usdt", "ETH"), ("ada_usdt", "ADA")]:
        try:
            _status(f"Computing order-flow imbalance for {ticker}")
            t0 = time.perf_counter()
            ofi = compute_order_flow_imbalance(
                asset=asset_key, freq="1min", levels=5, max_rows=2000,
            )
            _done(time.perf_counter() - t0)
            _kv(f"  {ticker} OFI mean", f"{np.nanmean(ofi):.6f}")
            _kv(f"  {ticker} OFI std", f"{np.nanstd(ofi):.6f}")
            _kv(f"  {ticker} OFI skew", f"{_skewness(ofi):.4f}")
        except Exception:
            pass

    # Liquidity depth analysis
    try:
        _status("Building liquidity depth matrix")
        t0 = time.perf_counter()
        depth, depth_names = build_liquidity_depth_matrix(
            freq="1min", levels=5, max_rows=2000,
        )
        _done(time.perf_counter() - t0)
        for i, name in enumerate(depth_names):
            _kv(f"  {name} avg depth", f"{np.nanmean(depth[i]):.2f}")
    except Exception:
        pass


def _skewness(x: np.ndarray) -> float:
    """Compute skewness of finite values."""
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-15:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 3))


# ═══════════════════════════════════════════════════════════════
#  Main demonstration
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hankel-ADMM Liquidity Reconstruction Pipeline"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Force synthetic data mode even if real data is available",
    )
    parser.add_argument(
        "--max-rows", type=int, default=4096,
        help="Maximum rows to load from each asset CSV (default: 4096)",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Attempt to download the Kaggle dataset before running",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run rolling-window cross-validation after the main pipeline",
    )
    args = parser.parse_args()

    _header("Robust Tensor-Completion via Hankel-Iterative Proximal ADMM")
    print(f"  {_D}Low-Latency Liquidity Reconstruction in Fragmented FinTech Ecosystems{_RS}")
    print(f"  {_D}{_LN}{_RS}\n")

    # ── Attempt download if requested ──
    if args.download:
        try:
            download_dataset()
        except FileNotFoundError:
            pass

    # ── Load data ──
    use_real = check_data_available() and not args.synthetic
    if use_real:
        X_true, X_frag, asset_names = _load_real_data(max_rows=args.max_rows)
        data_source = "Kaggle LOB (BTC/ETH/ADA)"
    else:
        X_true, X_frag, asset_names = _load_synthetic_data()
        data_source = "Synthetic"

    n_assets, T = X_frag.shape
    n_injected = int(np.sum(~np.isnan(X_frag) & (X_frag != X_true)))

    # ── Pipeline Execution ──
    _section("Liquidity Reconstruction Pipeline")

    L = max(T // 4, 32)
    ssa_rank = min(6, n_assets)
    dmd_rank = min(10, n_assets)

    engine = LiquidityEngine(
        L=L,
        ssa_rank=ssa_rank,
        admm_rho=1.5,
        admm_max_iter=300,
        admm_tol=1e-5,
        dmd_rank=dmd_rank,
        alpha=0.05,
    )

    _status("Ingesting data")
    t0 = time.perf_counter()
    engine.ingest(X_frag)
    _done(time.perf_counter() - t0)

    _status("Running Hankel-SSA + ADMM-RPCA")
    t0 = time.perf_counter()
    engine.reconstruct()
    _done(time.perf_counter() - t0)

    _status("Running Dynamic Mode Decomposition")
    t0 = time.perf_counter()
    engine.analyze_dynamics(forecast_steps=64)
    _done(time.perf_counter() - t0)

    _status("Running statistical anomaly forensics")
    t0 = time.perf_counter()
    engine.detect_anomalies()
    _done(time.perf_counter() - t0)

    # Portfolio rebalance
    w_before = np.ones(n_assets) / n_assets
    _status("Computing optimal portfolio rebalance")
    t0 = time.perf_counter()
    w_after = engine.rebalance_portfolio(w_before, risk_aversion=2.0)
    _done(time.perf_counter() - t0)

    report = engine.generate_report()

    # ── Granger Causality Network ──
    _section("Granger Causality Lead-Lag Analysis")

    _status("Running pairwise Granger F-tests (BH-corrected)")
    t0 = time.perf_counter()
    engine.analyze_lead_lag(max_lag=5)
    _done(time.perf_counter() - t0)

    gc = engine.granger_adj
    n_gc_edges = int(np.sum(gc)) if gc is not None else 0
    _kv("Significant links (BH-corrected)", n_gc_edges)
    if gc is not None and n_gc_edges > 0:
        best_idx = np.unravel_index(np.argmax(engine.granger_fstats), gc.shape)
        src = asset_names[best_idx[0]] if best_idx[0] < len(asset_names) else f"A{best_idx[0]}"
        tgt = asset_names[best_idx[1]] if best_idx[1] < len(asset_names) else f"A{best_idx[1]}"
        _kv("Strongest link", f"{src} → {tgt} (F={engine.granger_fstats[best_idx]:.2f})")
        # Count bidirectional pairs
        bidir = gc & gc.T
        n_bidir = int(np.sum(bidir)) // 2
        _kv("Bidirectional pairs", n_bidir)

    # Refresh report to include Granger data
    report = engine.generate_report()

    # ── Results Report ──
    _section("Results Report")
    _kv("Data source", data_source, _CY)
    _kv("Assets", f"{asset_names}")
    _kv("Dimensions", f"{n_assets} assets × {T} time steps")

    # Reconstruction quality
    rmse_per_asset = np.sqrt(np.mean((X_true - engine.X_clean) ** 2, axis=1))
    overall_rmse = float(np.mean(rmse_per_asset))
    _kv("Overall reconstruction RMSE", f"{overall_rmse:.6f}")
    for i, name in enumerate(asset_names):
        _kv(f"  {name} RMSE", f"{rmse_per_asset[i]:.6f}", _D)

    if report.get("reconstruction"):
        ci = report["reconstruction"]
        _kv("Bootstrap RMSE CI (95%)", f"[{ci['ci_lo']:.6f}, {ci['ci_hi']:.6f}]")

    # ADMM convergence
    admm = report.get("admm", {})
    _kv("ADMM iterations", admm.get("iterations", "N/A"))
    _kv("Final primal residual", f"{admm.get('final_primal_residual', 0):.2e}")
    _kv("Final dual residual", f"{admm.get('final_dual_residual', 0):.2e}")
    _kv("Final rank", admm.get("final_rank", "N/A"))

    # DMD stability
    dmd = report.get("dmd_stability", {})
    _kv("Spectral radius", f"{dmd.get('spectral_radius', 0):.6f}")
    _kv("Dominant frequency (Hz)", f"{dmd.get('dominant_frequency_hz', 0):.6f}")
    _kv("Stable modes", dmd.get("n_stable", 0))
    _kv("Marginal modes", dmd.get("n_marginal", 0))
    _kv("Unstable modes", dmd.get("n_unstable", 0),
        _RD if dmd.get("n_unstable", 0) > 0 else _GR)

    # Anomalies
    anomaly = report.get("anomalies", {})
    _kv("Anomalies detected (BH-corrected)", anomaly.get("n_detected", 0))

    # Normality
    norm = report.get("normality_test", {})
    _kv("Normality test", norm.get("test", "N/A"))
    _kv("  JB statistic", f"{norm.get('statistic', 0):.4f}")
    _kv("  p-value", f"{norm.get('p_value', 0):.6f}")
    _kv("  Reject H₀?", "Yes" if norm.get("reject_null") else "No",
        _RD if norm.get("reject_null") else _GR)

    # Portfolio
    port = report.get("portfolio", {})
    _kv("Portfolio weights sum", f"{port.get('sum', 0):.6f}")
    if port.get("weights"):
        for i, w in enumerate(port["weights"]):
            name = asset_names[i] if i < len(asset_names) else f"A{i}"
            colour = _GR if w > 0.01 else _D
            _kv(f"  {name} weight", f"{w:.6f}", colour)

    # Timing
    timing = report.get("pipeline_timing", {})
    _section("Pipeline Timing")
    _kv("Hankel-SSA", f"{timing.get('hankel_ssa_s', 0):.4f}s")
    _kv("ADMM-RPCA", f"{timing.get('admm_rpca_s', 0):.4f}s")
    _kv("DMD", f"{timing.get('dmd_s', 0):.4f}s")
    _kv("Statistical tests", f"{timing.get('stat_tests_s', 0):.4f}s")
    _kv("Granger causality", f"{timing.get('granger_s', 0):.4f}s")
    _kv("Rebalance", f"{timing.get('rebalance_s', 0):.4f}s")
    _kv("Total pipeline", f"{timing.get('total_s', 0):.4f}s", _CY)

    # ── Microstructure analysis (real data only) ──
    if use_real:
        _analyze_microstructure()

    # ── Visualisations ──
    _section("Generating Visualisations")

    X_noisy_plot = np.nan_to_num(X_frag, nan=0.0)

    _status(f"Reconstruction comparison ({asset_names[0]})")
    path = plot_reconstruction_comparison(
        X_true[0], X_noisy_plot[0], engine.X_clean[0], asset_idx=0,
    )
    _done(0)
    _kv("Saved", path, _D)

    n_show = min(4, n_assets)
    _status(f"Multi-asset reconstruction grid ({n_show} assets)")
    path = plot_multi_asset_reconstruction(
        X_true, X_noisy_plot, engine.X_clean, n_show=n_show,
    )
    _done(0)
    _kv("Saved", path, _D)

    _status("ADMM convergence plot")
    path = plot_admm_convergence(
        engine.admm_hist.primal_residual,
        engine.admm_hist.dual_residual,
        engine.admm_hist.rank,
    )
    _done(0)
    _kv("Saved", path, _D)

    _status("DMD spectrum (unit circle)")
    path = plot_dmd_spectrum(engine.dmd_eigenvalues)
    _done(0)
    _kv("Saved", path, _D)

    _status("Anomaly heatmap")
    path = plot_anomaly_heatmap(engine.S_anomaly)
    _done(0)
    _kv("Saved", path, _D)

    _status("Portfolio weights chart")
    path = plot_portfolio_weights(w_before, w_after)
    _done(0)
    _kv("Saved", path, _D)

    _status(f"Spectral density ({asset_names[0]})")
    path = plot_spectral_density(engine.X_clean[0])
    _done(0)
    _kv("Saved", path, _D)

    # Granger network
    if engine.granger_adj is not None:
        _status("Granger causality network")
        path = plot_granger_network(
            engine.granger_adj, engine.granger_fstats,
            asset_names=asset_names,
        )
        _done(0)
        _kv("Saved", path, _D)

    # ── Rolling-Window Backtest ──
    if args.backtest:
        _section("Rolling-Window Cross-Validation Backtest")
        _status("Running 5-fold rolling backtest")
        t0 = time.perf_counter()
        backtester = RollingBacktester(
            n_windows=5,
            test_fraction=0.2,
            missing_ratio=0.15,
            L=L,
            ssa_rank=ssa_rank,
            admm_rho=1.5,
            admm_max_iter=200,
            admm_tol=1e-5,
        )
        bt_report = backtester.run(X_true)
        _done(time.perf_counter() - t0)

        _kv("Windows", bt_report.n_windows)
        _kv("Mean OOS RMSE", f"{bt_report.mean_rmse:.6f}")
        _kv("Std OOS RMSE", f"{bt_report.std_rmse:.6f}")
        _kv("Best window", f"{bt_report.best_rmse:.6f}")
        _kv("Worst window", f"{bt_report.worst_rmse:.6f}")
        overfit_colour = _RD if bt_report.overfit_ratio > 2.0 else _GR
        _kv("Overfit ratio (test/train)", f"{bt_report.overfit_ratio:.4f}", overfit_colour)

        for i, (tr, te) in enumerate(zip(bt_report.train_rmses, bt_report.window_rmses)):
            _kv(f"  Window {i}", f"train={tr:.4f}  test={te:.4f}", _D)

        _status("Generating backtest RMSE plot")
        path = plot_backtest_rmse(bt_report.window_rmses)
        _done(0)
        _kv("Saved", path, _D)

        report["backtest"] = {
            "n_windows": bt_report.n_windows,
            "mean_rmse": round(bt_report.mean_rmse, 6),
            "std_rmse": round(bt_report.std_rmse, 6),
            "worst_rmse": round(bt_report.worst_rmse, 6),
            "best_rmse": round(bt_report.best_rmse, 6),
            "overfit_ratio": round(bt_report.overfit_ratio, 4),
            "per_window": [round(float(r), 6) for r in bt_report.window_rmses],
        }

    # ── Export JSON ──
    _section("Exporting Results")

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results_summary.json"

    report["reconstruction_per_asset_rmse"] = {
        name: round(float(r), 6)
        for name, r in zip(asset_names, rmse_per_asset)
    }
    report["overall_rmse"] = round(overall_rmse, 6)
    report["data_config"] = {
        "source": data_source,
        "n_assets": n_assets,
        "T": T,
        "asset_names": asset_names,
    }

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    _kv("JSON report", str(json_path), _D)

    # ── Done ──
    _header("Pipeline Complete ✓")
    print(f"  {_GR}All stages executed successfully.{_RS}")
    print(f"  {_D}Data source: {data_source}{_RS}")
    print(f"  {_D}Outputs saved to: {out_dir}{_RS}\n")


if __name__ == "__main__":
    main()
