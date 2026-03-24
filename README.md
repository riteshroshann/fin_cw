# Robust Tensor-Completion via Hankel-Iterative Proximal ADMM

**Low-Latency Liquidity Reconstruction in Fragmented FinTech Ecosystems**

---

## Abstract

This project implements a research-grade framework for **autonomous liquidity reconstruction** and **real-time anomaly forensics** in decentralized finance (DeFi) ecosystems. By combining **Hankelization-based Singular Spectrum Analysis (SSA)**, **Proximal-ADMM optimisation** for Robust PCA, and **Dynamic Mode Decomposition (DMD)** for operator-theoretic stability analysis, the system provides a mathematically rigorous "Smart-Liquidity" engine capable of:

1. **Reconstructing fragmented order-book data** from asynchronous, incomplete market feeds via iterative Hankel-ADMM coupling
2. **Decomposing market microstructure** into low-rank dynamics and sparse anomalies  
3. **Detecting and classifying predatory trading patterns** (flash crashes, wash-trades, latency gaps)
4. **Discovering lead-lag relationships** between assets via Granger causality networks (BH FDR-corrected)
5. **Forecasting short-term liquidity dynamics** via DMD eigenvalue analysis with out-of-sample validation
6. **Autonomous portfolio rebalancing** using cleaned covariance estimates
7. **Kronecker-structured covariance estimation** (Σ ≈ A ⊗ B) via Van Loan & Pitsianis rearrangement
8. **Tensor-mode structural analysis** of multi-asset Hankel trajectory tensors
9. **Cross-validation** via rolling-window backtesting for out-of-sample robustness

## Mathematical Foundations

### Unit 1 — Structured Linear Algebra
- **Kronecker Products** (A ⊗ B) for cross-asset covariance modelling and nearest-Kronecker decomposition
- **Shift Matrices** as discrete lag operators with quantitative autocorrelation analysis
- **Hankelization** transforms irregular transaction streams into trajectory matrices
- **Discrete Fourier Transform** for isolating periodic liquidity cycles
- **Tensor mode-n unfolding/folding** for multilinear rank analysis of Hankel tensors

### Unit 2 — Convex Optimisation (ADMM)
- **Augmented Lagrangian** formulation: min ‖X‖_* + λ‖S‖₁  s.t. P_Ω(M) = P_Ω(X + S)
- **Proximal operators**: Singular Value Thresholding (nuclear norm) and soft-thresholding (ℓ₁)
- **Adaptive ρ-scaling** (Boyd et al. §3.4.1) for balanced primal-dual convergence
- **Iterative Hankel-ADMM coupling**: alternating SSA ↔ ADMM for progressive refinement

### Unit 3 — Statistical Inference
- **Bootstrap confidence intervals** for reconstruction RMSE
- **Bias-variance decomposition** of reconstruction error via bootstrap resampling
- **Jarque–Bera normality testing** on residuals (scipy χ²(2))
- **Granger causality F-tests** for lead-lag relationships (scipy F-distribution, BH FDR-corrected)
- **Anomaly classification** into flash-crash / wash-trade / latency-gap categories
- **Q-Q residual diagnostics** for normality assessment

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
├── linalg_primitives.py     # Kronecker, Hankel, DFT, SVD, tensor ops, proximal operators
├── data_engine.py           # Synthetic LOB generator, anomaly injection
├── kaggle_loader.py         # Kaggle LOB dataset loader & augmented features
├── hankel_pipeline.py       # SSA reconstruction pipeline
├── admm_solver.py           # ADMM-RPCA, matrix completion, iterative Hankel-ADMM
├── dmd_engine.py            # Dynamic Mode Decomposition engine
├── stat_testing.py          # Hypothesis testing, anomaly classification, CI, bias-variance
├── liquidity_engine.py      # Smart-Liquidity orchestrator (8 pipeline stages)
├── backtest.py              # Rolling-window cross-validation backtester
├── viz_forensics.py         # Publication-quality visualisations (13 plot types)
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

## Pipeline Stages

| Stage | Method | Output |
|-------|--------|--------|
| 1. Ingest | Raw data intake | Fragmented matrix M |
| 2. Reconstruct | Iterative Hankel-SSA ↔ ADMM-RPCA | Low-rank X, sparse S |
| 3. Dynamics | Exact DMD + stability analysis | Eigenvalues, modes, forecast |
| 4a. Anomalies | BH-corrected significance + classification | Typed anomaly report |
| 4b. Granger | Pairwise F-tests + FDR correction | Directed causality graph |
| 5. Rebalance | Mean-variance optimisation on cleaned Σ | Optimal portfolio weights |
| 6. Kronecker | Van Loan & Pitsianis decomposition | Σ ≈ A ⊗ B |
| 7. Forecast | DMD 80/20 split validation | Out-of-sample RMSE |
| 8. Tensor | Multilinear rank of Hankel tensor | Tucker rank analysis |

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
| `qq_residuals.png` | Normal Q-Q plot for residual diagnostics |
| `forecast_comparison.png` | DMD forecast vs actuals with error bands |
| `kronecker_spectrum.png` | Singular value spectrum of Kronecker factors |
| `cumulative_pnl.png` | Portfolio equity curve vs equal-weight baseline |
| `backtest_rmse.png` | Rolling-window out-of-sample RMSE (with `--backtest`) |
| `results_summary.json` | Structured metrics and diagnostics |

## Research Applications

- **ACM ICAIF** — AI in Finance conference paper on robust liquidity provision
- **Journal of Risk and Financial Management** — Operator-theoretic market stability
- **Patent potential**: Method for real-time liquidity reconstruction using sparse proximal proxies

## Key Algorithmic Contributions

1. **Iterative Hankel-ADMM coupling**: Alternating SSA ↔ Proximal-ADMM with adaptive ρ-scaling for provably convergent reconstruction
2. **Tensor-mode structural analysis**: Multilinear rank characterisation of the multi-asset Hankel trajectory tensor
3. **Kronecker covariance estimation**: Nearest Σ ≈ A ⊗ B decomposition for sector-structured correlation modelling
4. **Anomaly forensics**: Classification of detected sparse anomalies into flash-crash, wash-trade, and latency-gap categories
5. **DMD forecast validation**: Out-of-sample DMD prediction with per-asset RMSE benchmarks
6. **Statistical rigour**: scipy-backed p-values, BH FDR correction, bootstrap bias-variance decomposition
7. **Rolling cross-validation**: Temporal backtesting with overfit ratio monitoring

---

*Built with NumPy, SciPy, Matplotlib, and first-principles linear algebra.*
