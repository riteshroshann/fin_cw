"""
viz_forensics.py
================
Publication-quality matplotlib visualizations for every stage of the
Hankel-ADMM liquidity reconstruction pipeline.

All figures are auto-saved to the ``outputs/`` directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── Global style ──
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "monospace",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

OUTPUT_DIR = Path(__file__).parent / "outputs"


def _ensure_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# Curated colour palette (cyberpunk-fintech aesthetic)
PALETTE = [
    "#58a6ff",  # electric blue
    "#f0883e",  # warm orange
    "#3fb950",  # green
    "#bc8cff",  # lavender
    "#ff7b72",  # coral
    "#79c0ff",  # light blue
    "#d2a8ff",  # soft purple
    "#ffa657",  # amber
]


# ──────────────────────────────────────────────────────────────
#  1. Reconstruction comparison
# ──────────────────────────────────────────────────────────────

def plot_reconstruction_comparison(
    x_true: NDArray,
    x_noisy: NDArray,
    x_hat: NDArray,
    asset_idx: int = 0,
    filename: str = "reconstruction_comparison.png",
) -> str:
    """Overlay plot: true signal vs noisy vs reconstructed."""
    out = _ensure_dir()
    fig, ax = plt.subplots(figsize=(14, 5))

    t = np.arange(len(x_true))
    ax.plot(t, x_true, color=PALETTE[0], lw=1.8, label="Ground Truth", alpha=0.85)
    ax.plot(t, x_noisy, color=PALETTE[4], lw=0.7, label="Observed (noisy/missing)", alpha=0.5)
    ax.plot(t, x_hat, color=PALETTE[2], lw=1.4, label="Reconstructed", alpha=0.9, linestyle="--")

    ax.set_xlabel("Time step")
    ax.set_ylabel("Mid-price")
    ax.set_title(f"Asset {asset_idx} — Liquidity Reconstruction", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.3)
    ax.grid(True, ls=":")
    fig.tight_layout()

    path = str(out / filename)
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
#  2. ADMM convergence
# ──────────────────────────────────────────────────────────────

def plot_admm_convergence(
    primal_res: list[float],
    dual_res: list[float],
    ranks: list[int] | None = None,
    filename: str = "admm_convergence.png",
) -> str:
    """Primal/dual residual log-scale curves + rank evolution."""
    out = _ensure_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    iters = np.arange(1, len(primal_res) + 1)

    # Residuals
    ax = axes[0]
    ax.semilogy(iters, primal_res, color=PALETTE[0], lw=1.5, label="Primal residual")
    ax.semilogy(iters, dual_res, color=PALETTE[1], lw=1.5, label="Dual residual")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalised residual")
    ax.set_title("ADMM Convergence", fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.3)
    ax.grid(True, ls=":", alpha=0.5)

    # Rank evolution
    ax = axes[1]
    if ranks:
        ax.plot(iters[:len(ranks)], ranks, color=PALETTE[2], lw=1.5, marker="o", ms=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Effective rank")
    ax.set_title("Low-rank Component Rank", fontsize=12, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.5)

    fig.tight_layout()
    path = str(out / filename)
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
#  3. DMD spectrum (unit circle)
# ──────────────────────────────────────────────────────────────

def plot_dmd_spectrum(
    eigenvalues: NDArray,
    filename: str = "dmd_spectrum.png",
) -> str:
    """DMD eigenvalues on the unit circle with stability annotations."""
    out = _ensure_dir()
    fig, ax = plt.subplots(figsize=(7, 7))

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#30363d", lw=1.5, ls="--", alpha=0.8)

    mags = np.abs(eigenvalues)
    for i, ev in enumerate(eigenvalues):
        colour = PALETTE[2] if mags[i] < 1.02 else PALETTE[4]  # stable vs unstable
        ax.plot(ev.real, ev.imag, "o", color=colour, ms=8, alpha=0.85,
                markeredgecolor="#0d1117", markeredgewidth=0.8)

    ax.axhline(0, color="#30363d", lw=0.5)
    ax.axvline(0, color="#30363d", lw=0.5)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title("DMD Eigenvalue Spectrum", fontsize=13, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[2],
               ms=10, label="Stable / Marginal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[4],
               ms=10, label="Unstable"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.3)

    fig.tight_layout()
    path = str(out / filename)
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
#  4. Anomaly heatmap
# ──────────────────────────────────────────────────────────────

def plot_anomaly_heatmap(
    S: NDArray,
    filename: str = "anomaly_heatmap.png",
) -> str:
    """Heatmap of the sparse anomaly matrix S."""
    out = _ensure_dir()
    fig, ax = plt.subplots(figsize=(14, 5))

    im = ax.imshow(
        np.abs(S),
        aspect="auto",
        cmap="inferno",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("|S| — Anomaly Magnitude")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Asset")
    ax.set_title("Sparse Anomaly Matrix (|S|)", fontsize=13, fontweight="bold")

    fig.tight_layout()
    path = str(out / filename)
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
#  5. Portfolio weights
# ──────────────────────────────────────────────────────────────

def plot_portfolio_weights(
    weights_before: NDArray,
    weights_after: NDArray,
    filename: str = "portfolio_weights.png",
) -> str:
    """Side-by-side bar chart of portfolio weights before/after rebalance."""
    out = _ensure_dir()
    n = len(weights_before)
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(n)
    width = 0.35
    ax.bar(x - width / 2, weights_before, width, color=PALETTE[0], alpha=0.85, label="Before")
    ax.bar(x + width / 2, weights_after, width, color=PALETTE[2], alpha=0.85, label="After")

    ax.set_xlabel("Asset")
    ax.set_ylabel("Weight")
    ax.set_title("Portfolio Rebalancing", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"A{i}" for i in range(n)])
    ax.legend(framealpha=0.3)
    ax.grid(True, ls=":", axis="y", alpha=0.4)

    fig.tight_layout()
    path = str(out / filename)
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
#  6. Power Spectral Density
# ──────────────────────────────────────────────────────────────

def plot_spectral_density(
    x: NDArray,
    dt: float = 1.0,
    filename: str = "spectral_density.png",
) -> str:
    """Power spectral density of a time-series."""
    out = _ensure_dir()
    x = np.asarray(x).ravel()
    n = x.size
    freqs = np.fft.rfftfreq(n, d=dt)
    psd = np.abs(np.fft.rfft(x - np.mean(x))) ** 2 / n

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(freqs, psd, color=PALETTE[3], lw=1.3, alpha=0.9)
    ax.fill_between(freqs, psd, alpha=0.15, color=PALETTE[3])
    ax.set_xlabel("Frequency (1/Δt)")
    ax.set_ylabel("Power")
    ax.set_title("Power Spectral Density", fontsize=13, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    path = str(out / filename)
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
#  7. Multi-asset reconstruction grid
# ──────────────────────────────────────────────────────────────

def plot_multi_asset_reconstruction(
    X_true: NDArray,
    X_noisy: NDArray,
    X_hat: NDArray,
    n_show: int = 4,
    filename: str = "multi_asset_reconstruction.png",
) -> str:
    """Grid of reconstruction plots for multiple assets."""
    out = _ensure_dir()
    n = min(n_show, X_true.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    t = np.arange(X_true.shape[1])
    for i, ax in enumerate(axes):
        ax.plot(t, X_true[i], color=PALETTE[0], lw=1.5, alpha=0.8, label="True")
        ax.plot(t, X_noisy[i], color=PALETTE[4], lw=0.6, alpha=0.4, label="Observed")
        ax.plot(t, X_hat[i], color=PALETTE[2], lw=1.2, ls="--", alpha=0.9, label="Reconstructed")
        ax.set_ylabel(f"Asset {i}")
        ax.grid(True, ls=":", alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.3, fontsize=8)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("Multi-Asset Liquidity Reconstruction", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    path = str(out / filename)
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
#  8. Granger causality network
# ──────────────────────────────────────────────────────────────

def plot_granger_network(
    adj: np.ndarray,
    fstats: np.ndarray,
    asset_names: list[str] | None = None,
    filename: str = "granger_network.png",
) -> str:
    """Directed graph of significant Granger-causal links.

    Nodes are placed on a circle. Directed edges are drawn for each
    significant (i → j) link, with width proportional to F-statistic.
    Bidirectional pairs are coloured distinctly.
    """
    out = _ensure_dir()
    n = adj.shape[0]
    if asset_names is None:
        asset_names = [f"A{i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Circular layout
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    cx = np.cos(angles)
    cy = np.sin(angles)
    radius = 0.85

    # Determine bidirectional pairs
    bidir = adj & adj.T

    # Normalise edge widths
    max_f = max(float(np.max(fstats)), 1e-6)

    # Draw edges
    for i in range(n):
        for j in range(n):
            if not adj[i, j]:
                continue
            f_norm = fstats[i, j] / max_f
            lw = 0.8 + 3.5 * f_norm
            alpha_e = 0.4 + 0.5 * f_norm

            # Slight offset for bidirectional to avoid overlap
            if bidir[i, j]:
                colour = PALETTE[5]  # light blue for bidirectional
                offset = 0.03
            else:
                colour = PALETTE[1]  # orange for unidirectional
                offset = 0.0

            x0, y0 = radius * cx[i], radius * cy[i]
            x1, y1 = radius * cx[j], radius * cy[j]

            # Perpendicular offset for bidirectional
            dx, dy = x1 - x0, y1 - y0
            perp_x, perp_y = -dy, dx
            norm = max(np.sqrt(perp_x**2 + perp_y**2), 1e-8)
            perp_x, perp_y = perp_x / norm * offset, perp_y / norm * offset

            ax.annotate(
                "",
                xy=(x1 + perp_x, y1 + perp_y),
                xytext=(x0 + perp_x, y0 + perp_y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=colour,
                    lw=lw,
                    alpha=alpha_e,
                    connectionstyle="arc3,rad=0.12",
                    mutation_scale=14,
                ),
            )

    # Draw nodes
    for i in range(n):
        x, y = radius * cx[i], radius * cy[i]
        circle = plt.Circle(
            (x, y), 0.08, facecolor="#161b22", edgecolor=PALETTE[0],
            linewidth=2.0, zorder=10,
        )
        ax.add_patch(circle)
        ax.text(
            x, y, asset_names[i], ha="center", va="center",
            fontsize=9, fontweight="bold", color=PALETTE[0], zorder=11,
        )

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_title("Granger Causality Network", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=PALETTE[1], lw=2.5, label="Unidirectional"),
        Line2D([0], [0], color=PALETTE[5], lw=2.5, label="Bidirectional"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.3, fontsize=9)

    n_edges = int(np.sum(adj))
    ax.text(
        0, -1.18, f"{n_edges} significant link{'s' if n_edges != 1 else ''} (BH-corrected)",
        ha="center", va="center", fontsize=9, color="#8b949e",
    )

    fig.tight_layout()
    path = str(out / filename)
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
#  9. Backtest RMSE bar chart
# ──────────────────────────────────────────────────────────────

def plot_backtest_rmse(
    window_rmses: np.ndarray,
    filename: str = "backtest_rmse.png",
) -> str:
    """Bar chart of per-window rolling-backtest RMSE with mean ± 1σ."""
    out = _ensure_dir()
    n_windows = len(window_rmses)
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(n_windows)
    bars = ax.bar(x, window_rmses, color=PALETTE[0], alpha=0.85, edgecolor="#30363d", lw=0.5)

    mu = float(np.mean(window_rmses))
    sigma = float(np.std(window_rmses))
    ax.axhline(mu, color=PALETTE[2], lw=1.8, ls="--", label=f"Mean = {mu:.4f}")
    ax.axhspan(mu - sigma, mu + sigma, alpha=0.12, color=PALETTE[2], label=f"±1σ = {sigma:.4f}")

    ax.set_xlabel("Window Index")
    ax.set_ylabel("Out-of-Sample RMSE")
    ax.set_title("Rolling-Window Backtest", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"W{i}" for i in range(n_windows)], fontsize=8)
    ax.legend(framealpha=0.3, fontsize=9)
    ax.grid(True, ls=":", axis="y", alpha=0.4)

    fig.tight_layout()
    path = str(out / filename)
    fig.savefig(path)
    plt.close(fig)
    return path
