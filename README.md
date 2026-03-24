# Robust Tensor-Completion via Hankel-Iterative Proximal ADMM

**Low-Latency Liquidity Reconstruction in Fragmented FinTech Ecosystems**

---

## Abstract

This project implements a research-grade framework for **autonomous liquidity reconstruction** and **real-time anomaly forensics** in decentralized finance (DeFi) ecosystems. By combining **Hankelization-based Singular Spectrum Analysis (SSA)**, **Proximal-ADMM optimisation** for Robust PCA, and **Dynamic Mode Decomposition (DMD)** for operator-theoretic stability analysis, the system provides a mathematically rigorous "Smart-Liquidity" engine capable of:

1. **Reconstructing fragmented order-book data** from asynchronous, incomplete market feeds
2. **Decomposing market microstructure** into low-rank dynamics and sparse anomalies  
3. **Detecting predatory trading patterns** (flash crashes, wash-trades) with statistical significance testing
4. **Discovering lead-lag relationships** between assets via Granger causality networks
5. **Forecasting short-term liquidity dynamics** via DMD eigenvalue analysis
6. **Autonomous portfolio rebalancing** using cleaned covariance estimates
7. **Cross-validation** via rolling-window backtesting for out-of-sample robustness

## Mathematical Foundations

### Unit 1 — Structured Linear Algebra
- **Kronecker Products** (A ⊗ B) for cross-asset covariance modelling
- **Shift Matrices** as discrete lag operators in state-space representation
- **Hankelization** transforms irregular transaction streams into trajectory matrices
- **Discrete Fourier Transform** for isolating periodic liquidity cycles

### Unit 2 — Convex Optimisation (ADMM)
- **Augmented Lagrangian** formulation: min ‖X‖_* + λ‖S‖₁  s.t. P_Ω(M) = P_Ω(X + S)
- **Proximal operators**: Singular Value Thresholding (nuclear norm) and soft-thresholding (ℓ₁)
- **Primal-dual convergence** monitoring with automatic stopping

### Unit 3 — Statistical Inference
- **Bootstrap confidence intervals** for reconstruction RMSE
- **Jarque–Bera normality testing** on residuals
- **Granger causality F-tests** for lead-lag relationships (BH FDR-corrected)
- **Benjamini–Hochberg FDR control** for multiple anomaly testing

## Data Sources

### Synthetic Data (default)
The pipeline generates realistic multi-asset LOB mid-prices with controlled anomalies for testing. No external data required.

### Kaggle LOB Dataset (real-world)
Supports the [High Frequency Crypto Limit Order Book Data](https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data) by Martin Søgaard Nielsen (BTC, ETH, ADA limit-order-book snapshots at 1-sec/1-min/5-min granularity).

```bash
# Option 1: Automatic download via Kaggle API
pip install kaggle
# Place ~/.kaggle/kaggle.json with your credentials
python demo.py --download

# Option 2: Manual download
# Download from Kaggle and unzip into data/ directory
```

## Project Structure

```
fintech_admm/
├── linalg_primitives.py     # Kronecker, Hankel, DFT, SVD, proximal operators
├── data_engine.py           # Synthetic LOB generator, anomaly injection
├── kaggle_loader.py         # Kaggle LOB dataset loader & augmented features
├── hankel_pipeline.py       # SSA reconstruction pipeline
├── admm_solver.py           # ADMM-RPCA and matrix completion solvers
├── dmd_engine.py            # Dynamic Mode Decomposition engine
├── stat_testing.py          # Hypothesis testing and confidence intervals
├── liquidity_engine.py      # Smart-Liquidity orchestrator + Granger network
├── backtest.py              # Rolling-window cross-validation backtester
├── viz_forensics.py         # Publication-quality visualisations (9 plot types)
├── demo.py                  # End-to-end demonstration & benchmarks
├── requirements.txt         # Dependencies
├── README.md                # This file
└── outputs/                 # Generated plots and JSON report
```

## Quick Start

```bash
pip install -r requirements.txt
python demo.py                         # synthetic data, full pipeline
python demo.py --synthetic --backtest  # include rolling-window backtest
python demo.py --download              # download Kaggle data and run
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--synthetic` | Force synthetic data mode |
| `--download` | Attempt Kaggle API download before running |
| `--backtest` | Run 5-fold rolling-window cross-validation |
| `--max-rows N` | Limit CSV rows per asset (default: 4096) |

## Outputs

After running `demo.py`, the `outputs/` directory will contain:

| File | Description |
|------|-------------|
| `reconstruction_comparison.png` | True vs observed vs reconstructed (single asset) |
| `multi_asset_reconstruction.png` | Grid of 4 assets |
| `admm_convergence.png` | Primal/dual residual curves + rank evolution |
| `dmd_spectrum.png` | DMD eigenvalues on the unit circle |
| `anomaly_heatmap.png` | Sparse anomaly matrix visualisation |
| `portfolio_weights.png` | Before/after rebalancing bar chart |
| `spectral_density.png` | Power spectral density of cleaned signal |
| `granger_network.png` | Directed Granger causality graph |
| `backtest_rmse.png` | Rolling-window out-of-sample RMSE (with `--backtest`) |
| `results_summary.json` | Structured metrics and diagnostics |

## Research Applications

- **ACM ICAIF** — AI in Finance conference paper on robust liquidity provision
- **Journal of Risk and Financial Management** — Operator-theoretic market stability
- **Patent potential**: Method for real-time liquidity reconstruction using sparse proximal proxies

## Key Algorithmic Contributions

1. **Hankel → ADMM pipeline**: First application of cascaded SSA + Proximal-ADMM to limit-order-book reconstruction
2. **DMD stability forensics**: Operator-theoretic classification of market micro-dynamics as stable/marginal/unstable
3. **Granger causality networks**: FDR-corrected lead-lag discovery across reconstructed asset dynamics
4. **Rolling cross-validation**: Temporal backtesting to validate out-of-sample reconstruction robustness
5. **Statistical rigour**: All anomaly detections carry p-values with FDR-corrected significance thresholds

---

*Built with NumPy, Matplotlib, and first-principles linear algebra.*
