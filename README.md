# Aircraft Engine Failure Forecasting
### NASA CMAPSS FD004 · IT 402 Applied Forecasting Methods · Group 13

*Predicting Remaining Useful Life on NASA CMAPSS FD004 using Classical Time-Series, Deep Learning, and Probabilistic Models*

| | |
|---|---|
| **Course** | IT 402 — Applied Forecasting Methods |
| **Institution** | Dhirubhai Ambani University, Gandhinagar |
| **Instructor** | Prof. Pritam Anand |
| **Academic Year** | 2025–26 |
| **Group** | 13 |

| Name | Student ID |
|------|-----------|
| Ummesalma Diwan | 202518017 |
| Dhruv Parmar | 202518030 |
| Aditya Jana | 202518035 |
| Shrey Pandya | 202518045 |

---

## Abstract

Unplanned turbofan engine failures impose severe economic and safety costs on the aviation industry. This project develops a complete, data-driven pipeline for **Remaining Useful Life (RUL) prediction** on the NASA CMAPSS FD004 dataset — the most challenging standard benchmark, combining six operating conditions and two simultaneous fault modes (High-Pressure Compressor and Fan degradation).

We design and evaluate **18 models** spanning three families: classical time-series models (AR, ARMA, ARIMA), five deep learning architectures (MLP, RNN, LSTM, GRU, Transformer), and five quantile regression models augmented with split conformal calibration for uncertainty-aware prediction. A PCA-based Health Index compresses 9 sensor signals into a scalar degradation trajectory used by classical models; deep learning models consume 30-cycle sliding windows of 48 per-cluster-normalised rolling features directly.

The **Transformer achieves RMSE = 14.42 cycles** and R² = 0.887 on 248 test engines. Deep learning collectively improves over classical ARIMA by **52.4% in RMSE**. The calibrated Q-GRU model provides 80.6% prediction interval coverage with a mean interval width of 25.1 cycles. Every design choice — RUL cap, cluster count, differencing order, window length, failure threshold, safety factor — is derived from data through statistical testing or validation-set optimisation.

**Keywords:** Remaining Useful Life, Predictive Maintenance, NASA CMAPSS, Transformer, Quantile Regression, Conformal Calibration, Health Index, PCA.

---

## 1. Introduction

### 1.1 Background and Motivation

A single unplanned in-flight engine failure can cost airlines in excess of $500,000 per incident. Traditional fixed-interval maintenance schedules service healthy engines unnecessarily and may still miss engines degrading faster than anticipated.

**Condition-Based Maintenance (CBM)** replaces the fixed-calendar trigger with a data-driven signal: estimate how much useful life remains in each engine and schedule maintenance only when that estimate falls below a defined safety threshold. A CBM programme backed by accurate predictions reduces fleet-wide maintenance burden by approximately 15–30% relative to fixed schedules.

### 1.2 Problem Statement

Given the operational history **X**_t^(i) ∈ ℝ^(t×d) of aircraft engine *i* through flight cycle *t*, predict the Remaining Useful Life:

```
ŷ_t^(i) = f(X_t^(i); θ)

where y_t^(i) = min(T^(i) − t, 125) cycles
```

Dataset: NASA CMAPSS FD004 — 249 training engines (61,249 rows), 248 test engines (41,214 rows), 21 sensors, 3 operating settings per cycle.

Three FD004-specific challenges: (i) six operating conditions mask degradation trends; (ii) two independent fault modes produce different trajectories; (iii) test RUL spans 6–195 cycles (mean = 86.6, std = 54.5).

### 1.3 Objectives

1. **Benchmark-quality point prediction** — competitive with published state of the art on FD004.
2. **Calibrated uncertainty quantification** — prediction intervals with guaranteed coverage, e.g. *"RUL = 45 cycles, 80% CI: [38, 53]"*.
3. **Evidence-based design** — every hyperparameter derived from data via statistical tests or validation-set optimisation.
4. **Classical vs. deep learning comparison** — systematically quantify the performance gap and identify structural reasons.

---

## 2. Data Source — NASA CMAPSS FD004

The **NASA C-MAPSS** dataset models turbofan engine degradation under controlled simulation, providing run-to-failure sensor trajectories. We select **FD004** for three data-backed reasons:

1. **Maximum complexity** — the only subset combining 6 operating conditions *and* 2 fault modes simultaneously.
2. **Widest RUL spread** — test RUL std = 54.5 cycles, requiring accurate prediction across the full dynamic range.
3. **Literature consensus** — all five published state-of-the-art papers use FD004 as their primary benchmark.

### 2.1 Dataset Characteristics

| Subset | Op. Conditions | Fault Modes | Train Engines | Test Engines | Mean RUL | Std RUL |
|--------|---------------|-------------|--------------|-------------|---------|--------|
| FD001 | 1 | 1 | 100 | 100 | — | — |
| FD002 | 6 | 1 | 260 | 259 | — | — |
| FD003 | 1 | 2 | 100 | 100 | — | — |
| **FD004** | **6** | **2** | **249** | **248** | **86.6** | **54.5** |

- **Structure:** engine ID · cycle number · 3 operating settings (throttle, altitude, Mach) · 21 sensor measurements
- **Size:** 61,249 training rows · 41,214 test rows · mean engine lifetime ≈ 246 cycles
- **Missing values:** none — all rows complete

### 2.2 Data Pre-processing

#### RUL Computation and Capping

```
y_t^(i) = min(T^(i) − t, RUL_CAP)   where RUL_CAP = 125 cycles
```

Cap value selected by validation-set grid search over {100, 110, 115, 120, 125, 130, 140, 150} — RMSE minimised at 125. Training RUL range: [0, 125]. Test RUL range: [6, 125].

#### Sensor Selection

Seven sensors dropped (near-zero variance or negligible RUL correlation): **s1, s5, s6, s10, s16, s18, s19**.

**14 retained sensors:** s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21.

Raw sensor–RUL correlations are universally weak (|r| < 0.11) because operating-condition variance dominates. Per-cluster normalisation exposes the underlying degradation signal.

#### Operating Condition Clustering

K-Means applied to (op1, op2, op3) space. Silhouette score peaks at **k = 6** (score = 1.00); cross-tabulation against NASA-defined regimes shows **100% agreement** (ARI ≥ 0.95). Cluster sizes: 15,395 / 9,224 / 9,139 / 9,091 / 9,238 / 9,162 training cycles.

Sensor s9's overall std ≈ 335 collapses to ≈ 15 within each condition — a **22× reduction** — confirming that per-cluster `StandardScaler` normalisation is essential.

#### Rolling Statistical Features

For each of 16 retained sensors and each window width w ∈ {5, 10, 20}:

```
μ_w(t) = (1/w) Σ x(τ)       rolling mean
σ_w(t) = std deviation        rolling std
```

Yields **96 rolling features** → **112 total features per cycle**. Rolling features never cross engine boundaries. Ablation: removing rolling features increases Transformer RMSE by +16.5% and GRU RMSE by +19.0%.

---

## 3. Methodology

### 3.1 PCA Health Index (for Classical Models)

Classical models require a univariate input. The **PCA Health Index** compresses 9 sensors (those with |r(s, −RUL)| ≥ 0.50) into a scalar degradation trajectory:

1. **Correlation filter** — retain 9 sensors: s2, s3, s4, s8, s9, s11, s13, s14, s17
2. **Per-cluster detrending** — subtract cluster mean to remove operating-condition baseline
3. **PCA** — PC1 alone explains **76.3% of within-condition variance** (PC2: 8.5%)
4. **Sign-flip + rolling median (window=10)** to produce a monotone-increasing degradation scalar

**Failure threshold:** θ = 1.4117 (5th percentile of near-failure health-index values, RUL ≤ 5)

**Predicted RUL** = 0.88 × min{h ≥ 1 : ẑ_{t+h} ≥ θ} — safety factor α = 0.88 chosen by validation-set grid search.

### 3.2 Classical Models

| Model | Key Property |
|-------|-------------|
| **AR(2)** | d=2 (ADF modal, 189/248 engines need d≥2); PACF cuts off at lag 2; AIC selects p=2 on 15 engines |
| **ARMA(2,2)** | AR(2) + MA terms to reduce residual ACF; MA reduces Ljung-Box violation |
| **ARIMA(1,2,2)** | AIC grid search over (p,q) ∈ {1,2,3,4}²; (1,2) wins modal on 15 engines |

All classical models predict RUL via threshold-crossing on the multi-step ARIMA health-index forecast.

### 3.3 Deep Learning Models

All 5 DL models share a unified framework. Training uses **NASA Asymmetric Loss** (not MSE):

```
d = ŷ − y

Early (d < 0):  penalty = exp(−d/13) − 1   ← grows slowly
Late  (d > 0):  penalty = exp( d/10) − 1   ← grows fast (safety-critical)
```

| Model | Architecture | Key Result |
|-------|-------------|-----------|
| **Transformer** | d_model=64, 4 heads, d_ff=256, 2 layers, mean-pooling, early stop ep.24 | RMSE=**14.42**, R²=0.887, Bias=−1.27 |
| **GRU** | 2-layer, hidden=64, dropout=0.1, early stop ep.37 | RMSE=15.29, R²=0.873, Bias=+0.71 |
| **RNN** | 2-layer, hidden=64, LayerNorm | RMSE=15.67, R²=0.87, Bias=−3.50 |
| **LSTM** | 2-layer StableLSTMBlock, hidden=64, LayerNorm | RMSE=16.29, R²=0.86, Bias=−4.99 |
| **MLP** | 2×FC(128), BatchNorm, dropout=0.35, flattened input 1440-dim | RMSE=18.41, R²=0.817, Bias=−7.47 |

### 3.4 Probabilistic Models

**Quantile regression** (Q-MLP, Q-RNN, Q-LSTM, Q-GRU, Q-Transformer): 3-output head predicts Q10/Q50/Q90 jointly using pinball loss. The [Q10, Q90] band forms an 80% prediction interval.

**Split conformal calibration** (distribution-free, finite-sample guarantee):
1. Compute non-conformity scores: s_i = max(Q10_i − y_i, y_i − Q90_i, 0)
2. Find conformal margin δ at the ⌈(n+1)(1−α)⌉/n quantile
3. Expand: calibrated bounds = [Q10 − δ, Q90 + δ] clipped to [0, 125]

![Calibration Comparison](results/calibration_comparison.png)
*Coverage and interval width before/after conformal calibration at 80% target. All models reach ≥80.6% after calibration.*

![Quantile Calibration Summary](results/summary_quantile_calibration.png)
*Summary of calibrated quantile model performance across all configurations.*

### 3.5 Tuned Hyperparameter Values

| Component | Parameter | Value (method) |
|-----------|-----------|---------------|
| Preprocessing | RUL cap | 125 cycles (val-set grid) |
| | KMeans clusters k | 6 (silhouette, ARI) |
| | Rolling window widths | 5, 10, 20 cycles |
| | PCA components | 1 (PC1 = 76.3% variance) |
| | Failure threshold θ | 1.4117 (q=0.05, val-set) |
| | Safety factor α | 0.88 (val-set NASA score) |
| AR / ARMA | Differencing order d | 2 (ADF modal) |
| | AR order p | 2 (AIC, 15 engines) |
| | MA order q | 2 (AIC, 15 engines) |
| ARIMA | Order (p, d, q) | (1, 2, 2) (AIC modal) |
| | Smooth window | 10 cycles (rolling median) |
| | Fit window | 50 cycles (recency) |
| Deep learning | Window size W | 30 cycles (val-set RMSE) |
| | Batch size | 128 |
| | Max epochs | 50 |
| | Learning rate | 1e-3 |
| | LR patience / factor | 5 / 0.5 |
| | Early stopping patience | 10 |
| | Hidden size | 64 |
| | Layers | 2 |
| | Dropout | 0.2 (GRU/LSTM/RNN); 0.35 (MLP) |
| | Transformer d_model | 64 |
| | Transformer n_heads | 4 |
| | Transformer d_ff | 256 |
| | Quantiles | {0.10, 0.50, 0.90} |
| | Conformal target | 80%, 90% |

---

## 4. Results

### 4.1 Classical Models

| Model | RMSE ↓ | NASA Score ↓ | R² ↑ | Bias |
|-------|--------|-------------|------|------|
| AR(2) | 30.90 | 24,719 | 0.483 | −3.93 (early) |
| ARMA(2,2) | 30.47 | 20,983 | 0.497 | −3.99 (early) |
| **ARIMA(1,2,2)** | **30.33** | **20,823** | **0.502** | **−3.44 (early)** |

All classical models explain ~50% of RUL variance (RMSE ≈ 30 cycles). The fundamental bottleneck is the **two-stage pipeline**: ARIMA forecasts the health-index trajectory, then threshold-crossing converts to RUL — each stage compounding error over 50–150 extrapolation steps.

### 4.2 Deep Learning Models

| Model | RMSE ↓ | NASA Score ↓ | R² ↑ | Bias |
|-------|--------|-------------|------|------|
| **Transformer** | **14.42** | **1,094** | **0.887** | −1.27 |
| GRU | 15.29 | 1,389 | 0.873 | +0.71 |
| RNN | 15.67 | 978 | 0.87 | −3.50 |
| LSTM | 16.29 | 956 | 0.86 | −4.99 |
| MLP | 18.41 | 3,201 | 0.817 | −7.47 |

**Transformer achieves RMSE = 14.42 cycles — new state of the art on FD004**, surpassing Song et al. (2022) at 14.86 cycles.

### 4.3 Probabilistic Models

| Model | Coverage (raw) | Coverage (+cal80) | Width (cycles) |
|-------|--------------|-----------------|---------------|
| Q-Transformer | 68.5% | **80.6%** | 20.9 |
| Q-GRU | 71.8% | **80.6%** | 25.1 |
| Q-LSTM | — | **80.6%** | 39.3 |
| Q-MLP | — | **80.6%** | 33.0 |
| Q-RNN | 86.3% (wide) | **80.6%** | 89.6 |

Q-Transformer achieves the best uncertainty/accuracy trade-off: **80.6% coverage at 20.9 cycles mean width**.

### 4.4 Literature Benchmark

| Reference | Architecture | FD004 RMSE | Year |
|-----------|-------------|-----------|------|
| Li et al. | DCNN | 22.36 | 2018 |
| Zhang et al. | Bidirectional LSTM | 23.99 | 2018 |
| Zhao et al. | BiLSTM variant | 18.42 | 2020 |
| Chen et al. | IBTSA | 16.14 | 2020 |
| Song et al. | TF-LSTM | 14.86 | 2022 |
| **This work** | **Transformer** | **14.42** | **2026** |

### 4.5 DL vs. Classical: 52.4% RMSE Improvement

Best DL (Transformer, 14.42) vs best classical (ARIMA, 30.33): **15.91 cycles — 52.4% improvement**. Two structural reasons:

1. **End-to-end RUL regression** — DL directly minimises RMSE on RUL, eliminating two-stage pipeline error.
2. **Multivariate modelling** — DL processes all 48 features × 30 cycles; ARIMA operates on a single scalar.

![PCA vs Median Ablation](results/ablation_pca_vs_median.png)
*Ablation: PCA Health Index (R² = −5.19) vs. median-of-top-5-sensors baseline (R² = −13.34), showing +8.15 improvement from PCA.*

---

## 5. Converting Forecasts to Actions

### Decision Framework

The maintenance engineer uses the **lower bound of the calibrated interval** as the hard deadline:

> *"Engine 12: RUL = 45 cycles — 90% CI: [38, 53] — model: Q-GRU+cal90"*
> → Hard deadline = 38 cycles

**Three action zones (by CI lower bound):**

| Zone | Lower Bound | Action |
|------|------------|--------|
| 🔴 IMMEDIATE | ≤ 15 cycles | Schedule at next available slot |
| 🟡 PLAN AHEAD | 16–40 cycles | Schedule within planning window; order parts now |
| 🟢 MONITOR | > 40 cycles | No action; flag for accelerated monitoring |

### Fleet Prioritisation Example (Q-GRU+cal90)

| Engine | Pred. RUL | CI Lower | CI Upper | True RUL | Action |
|--------|----------|---------|---------|---------|--------|
| Engine 7 | 12 | 7 | 19 | 11 | 🔴 IMMEDIATE |
| Engine 23 | 28 | 18 | 38 | 31 | 🟡 PLAN AHEAD |
| Engine 41 | 44 | 32 | 56 | 48 | 🟢 MONITOR |
| Engine 58 | 65 | 49 | 81 | 70 | 🟢 MONITOR |
| Engine 85 | 97 | 76 | 118 | 102 | 🟢 MONITOR |
| Engine 112 | 19 | 10 | 31 | 15 | 🔴 IMMEDIATE |

---

## 6. Future Work

1. **Multi-head conformal prediction sets** — capture bimodal RUL distributions from two independent fault modes.
2. **Online adaptive prediction** — real-time RUL tracking via incremental GRU/Transformer state updates.
3. **Attention-weight interpretability** — SHAP values to identify which cycles and sensors drive each prediction.
4. **Real sensor data validation** — confirm 52.4% DL advantage holds under real-world noise and missing readings.

---

## Pipeline

```
Raw CMAPSS data
      │
      ▼
01 · Data Loading          → globally-unique engine IDs
02 · RUL Computation       → compute & cap RUL at 125 cycles
03 · EDA                   → sensor variance, operating condition analysis
03b · Dataset Selection    → evidence-based FD004 justification
      │
      ▼
04 · Feature Engineering
      ├─ Drop 7 near-constant sensors (s1, s5, s6, s10, s16, s18, s19)
      ├─ KMeans clustering (k=6)    → recover 6 operating conditions
      ├─ Per-cluster StandardScaler → remove altitude/speed baseline
      └─ Rolling features           → 16 sensors × 3 windows × 2 stats = 96 features
      │
      ▼
Classical Models   → AR(2) · ARMA(2,2) · ARIMA(1,2,2) on PCA Health Index
Deep Learning      → MLP · RNN · LSTM · GRU · Transformer
Quantile Models    → Q-MLP · Q-RNN · Q-LSTM · Q-GRU · Q-Transformer
      │
      ▼
05 · Robustness & Ablation → isotonic ablation, PCA diagnostics
06 · Summary               → unified comparison across all 27 configurations
07 · Calibration           → split conformal at 80% / 90% targets
```

---

## Project Structure

```
├── data/
│   ├── raw/                         # original CMAPSS txt files (not committed)
│   └── processed/
│       ├── train_features.csv       # 16 sensors + 96 rolling features
│       └── test_features.csv
│
├── artifacts/
│   ├── kmeans_op_clusters.pkl       # fitted KMeans (k=6)
│   └── scalers.pkl                  # per-cluster StandardScalers
│
├── experiments/
│   ├── 01_data_pipeline/
│   │   ├── T01_data_loading.ipynb
│   │   ├── T02_rul_computation.ipynb
│   │   ├── T03_eda.ipynb
│   │   ├── T03b_dataset_selection.ipynb
│   │   └── T04_feature_engineering.ipynb
│   ├── 02_classical_models/
│   │   ├── T08_AR_model_book.ipynb
│   │   ├── T09_ARMA_model_book.ipynb
│   │   └── T10_ARIMA_model_book.ipynb
│   ├── 03_DL_Models/
│   │   ├── MLP.ipynb
│   │   ├── RNN.ipynb
│   │   ├── LSTM.ipynb
│   │   ├── GRU.ipynb
│   │   └── Transformer.ipynb
│   ├── 04_quantile_models/
│   │   ├── Q_MLP.ipynb
│   │   ├── Q_RNN.ipynb
│   │   ├── Q_LSTM.ipynb
│   │   ├── Q_GRU.ipynb
│   │   └── Q_Transformer.ipynb
│   ├── 05_robustness/
│   │   └── T13_ablation_robustness.ipynb
│   ├── 06_summary/
│   │   └── T14_final_summary.ipynb
│   └── 07_calibration/
│       └── T15_calibration.ipynb
│
├── src/
│   ├── preprocessing/
│   │   ├── load_data.py             # raw file loading + engine ID offsetting
│   │   ├── rul.py                   # RUL computation & capping
│   │   ├── cleaning.py              # variance-based sensor dropping
│   │   └── scaling.py              # KMeans + per-cluster StandardScaler
│   ├── features/
│   │   ├── rolling.py               # rolling mean/std per engine
│   │   └── windowing.py            # sliding window sequences for DL models
│   ├── models/
│   │   ├── classical.py             # Health Index, ARIMA, threshold, safety factor
│   │   ├── deep_learning.py         # DL architectures + quantile training utilities
│   │   └── dl_architectures.py      # MLP, RNN, LSTM, GRU, Transformer definitions
│   └── evaluation/
│       └── metrics.py               # RMSE, NASA score, R², save_model_results()
│
├── results/
│   ├── all_model_results.csv        # unified results written by every model notebook
│   ├── calibration_comparison.png
│   ├── summary_quantile_calibration.png
│   └── ablation_pca_vs_median.png
│
├── document/
│   └── report/
│       └── report.tex               # full LaTeX project report
│
└── Group 13.pdf                     # submitted project report
```

---

## Setup and Run Order

```bash
# 1. Create and activate environment
conda activate dl    # requires torch, statsmodels, scikit-learn, pandas, numpy

# 2. Place raw CMAPSS files in data/raw/
#    train_FD004.txt · test_FD004.txt · RUL_FD004.txt

# 3. Run notebooks in order
experiments/01_data_pipeline/    → T01 → T02 → T03 → T03b → T04
experiments/02_classical_models/ → T08 → T09 → T10
experiments/03_DL_Models/        → MLP → RNN → LSTM → GRU → Transformer
experiments/04_quantile_models/  → Q_MLP → Q_RNN → Q_LSTM → Q_GRU → Q_Transformer
experiments/05_robustness/       → T13
experiments/06_summary/          → T14
experiments/07_calibration/      → T15
```

Each model notebook writes to `results/all_model_results.csv`. T14 loads this file for the final comparison.

---

## Evaluation Metrics

**RMSE** — prediction error in flight cycles. Lower is better.

**NASA Asymmetric Score** — penalises late predictions exponentially more than early ones:
```
d = predicted_RUL − true_RUL

Early (d < 0):  exp(−d/13) − 1   ← grows slowly
Late  (d > 0):  exp( d/10) − 1   ← grows fast (safety-critical)

Total score = Σ penalties across all test engines
```

**R²** — proportion of RUL variance explained. Closer to 1.0 is better.

**PICP** — Prediction Interval Coverage Probability (target: 80%).

**MPIW** — Mean Prediction Interval Width in cycles. Lower = more precise.

---

*Group 13 · IT 402 Applied Forecasting Methods · Dhirubhai Ambani University · 2025–26*
