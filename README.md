# Aircraft Engine Failure Forecasting
### NASA CMAPSS · IT 402 Applied Forecasting Methods · Group 13

Predicting the **Remaining Useful Life (RUL)** of turbofan aircraft engines using classical time-series models, deep learning, and probabilistic quantile regression — evaluated on NASA's CMAPSS FD004 benchmark dataset.

---

## The Problem

Unplanned engine failures cost airlines $500K+ per incident and ground entire fleets. Traditional fixed-interval maintenance over-services healthy engines and under-protects degrading ones. This project builds a data-driven pipeline that forecasts **how many flight cycles an engine has left** before failure — enabling maintenance to be scheduled precisely when needed.

---

## Dataset — NASA CMAPSS FD004

We use **FD004**, the most challenging subset of the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset.

| Subset | Operating Conditions | Fault Modes | Train Engines | Test Engines |
|--------|---------------------|-------------|--------------|--------------|
| FD001  | 1                   | 1           | 100          | 100          |
| FD002  | 6                   | 1           | 260          | 259          |
| FD003  | 1                   | 2           | 100          | 100          |
| **FD004** | **6**            | **2**       | **248**      | **248**      |

FD004 is the only subset with **both** 6 operating conditions and 2 fault modes simultaneously — making it the hardest and the standard benchmark for multi-condition PHM research.

Each row is one flight cycle for one engine: 3 operating settings + 21 sensor measurements. Engines are run to failure in training; the test set is truncated and RUL must be predicted.

---

## Results

Ranked by NASA Score (lower = better). All results on FD004 test set (248 engines).

| # | Model | Type | NASA Score ↓ | RMSE ↓ | R² ↑ | Coverage |
|---|-------|------|-------------|--------|------|----------|
| 1 | **Q-GRU** | Quantile | **956** | 17.07 | 0.842 | 43.2% |
| 2 | **Q-MLP** | Quantile | **1,086** | 16.07 | 0.860 | 87.9% |
| 3 | **Q-Transformer** | Quantile | **1,222** | **13.90** | **0.895** | 80.7% |
| 4 | **Q-RNN** | Quantile | **1,455** | 15.29 | 0.873 | 71.8% |
| 5 | ARIMA(1,2,1) | Classical | 12,336 | 24.86 | 0.666 | — |
| 6 | Q-LSTM | Quantile | 11,235 | 40.31 | 0.120 | 21.0% |

**Best RMSE: Q-Transformer** — 13.90 cycles, R² 0.895, 80.7% coverage  
**Best NASA Score: Q-GRU** — 956, driven by tightest conservative predictions

> NASA Score penalises late predictions (pred > true) exponentially more than early ones — the metric that matters for safety-critical maintenance scheduling. Coverage = % of true RUL values falling within the Q10–Q90 prediction interval (target: 80%).

---

## Pipeline Overview

```
Raw CMAPSS data
      │
      ▼
01 · Data Loading          → assign globally-unique engine IDs across subsets
02 · RUL Computation       → compute & cap RUL at 125 cycles
03 · EDA                   → sensor variance, operating condition analysis
03b· Dataset Selection     → evidence-based justification for FD004
      │
      ▼
04 · Feature Engineering
      ├─ Drop 5 near-constant sensors  (s1, s5, s16, s18, s19)
      ├─ KMeans clustering (k=6)       → recover 6 operating conditions
      ├─ Per-cluster StandardScaler    → remove altitude/speed baseline
      └─ Rolling features              → 16 sensors × 3 windows × 2 stats = 96 features
      │
      ▼
Classical Models           → AR · ARMA · ARIMA on PCA Health Index
Deep Learning Models       → MLP · RNN · LSTM · GRU · Transformer
Quantile Models            → Q-MLP · Q-RNN · Q-LSTM · Q-GRU · Q-Transformer
      │
      ▼
05 · Robustness & Ablation → isotonic ablation, PCA diagnostics, FD001 generalisation
06 · Summary               → unified comparison across all models
```

---

## Key Design Choices (All Evidence-Based)

| Choice | Value | How Derived |
|--------|-------|-------------|
| RUL cap | 125 cycles | Sensitivity analysis — minimises val RMSE |
| KMeans k | 6 | Silhouette score + matches 6 known NASA conditions |
| Differencing order d | 2 | ADF test on all 248 training engines — modal d=2 |
| ARIMA order | (1, 2, 1) | AIC grid search over 15 representative engines |
| Failure threshold | 1.685 | 5th percentile of near-failure (RUL ≤ 5) health index |
| Safety factor | 0.88 | Grid search on validation data — minimises NASA score |
| Sequence window | 30 cycles | Window-size sensitivity on validation RMSE |
| Rolling windows | 5, 10, 20 | Multi-scale: noise filter, trend, long-range drift |

---

## Project Structure

```
├── data/
│   ├── raw/                        # original CMAPSS txt files (not committed)
│   └── processed/
│       ├── train_features.csv      # 16 sensors + 96 rolling features
│       └── test_features.csv
│
├── artifacts/
│   ├── kmeans_op_clusters.pkl      # fitted KMeans (k=6)
│   └── scalers.pkl                 # per-cluster StandardScalers
│
├── experiments/
│   ├── 00_pipeline.ipynb           # end-to-end demo: data → results
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
│   └── 06_summary/
│       └── T14_final_summary.ipynb
│
├── src/
│   ├── preprocessing/
│   │   ├── load_data.py            # raw file loading + engine ID offsetting
│   │   ├── rul.py                  # RUL computation & capping
│   │   ├── cleaning.py             # variance-based sensor dropping
│   │   └── scaling.py             # KMeans + per-cluster StandardScaler
│   ├── features/
│   │   ├── rolling.py              # rolling mean/std per engine
│   │   └── windowing.py           # sliding window sequences for DL models
│   ├── models/
│   │   ├── classical.py            # Health Index, ARIMA, threshold, safety factor
│   │   └── deep_learning.py        # DL architectures + quantile training utilities
│   └── evaluation/
│       └── metrics.py              # RMSE, NASA score, R², save_model_results()
│
├── results/
│   └── all_model_results.csv       # unified results written by every model notebook
│
├── AFM_ppt.pptx                    # project presentation
└── PROJECT_REPORT.md               # full report with all outputs and figures
```

---

## Run Order

```
# 1. Install dependencies
pip install -r requirements.txt   # or: numpy pandas scikit-learn torch statsmodels

# 2. Place raw CMAPSS files in data/raw/
#    train_FD001.txt … train_FD004.txt
#    test_FD001.txt  … test_FD004.txt
#    RUL_FD001.txt   … RUL_FD004.txt

# 3. Run notebooks in order
experiments/01_data_pipeline/   → T01 → T02 → T03 → T03b → T04
experiments/02_classical_models/ → T08 → T09 → T10
experiments/03_DL_Models/        → MLP → RNN → LSTM → GRU → Transformer
experiments/04_quantile_models/  → Q_MLP → Q_RNN → Q_LSTM → Q_GRU → Q_Transformer
experiments/05_robustness/       → T13
experiments/06_summary/          → T14

# Or run the full pipeline in one notebook:
experiments/00_pipeline.ipynb
```

Each model notebook writes its results to `results/all_model_results.csv`. T14 loads this file to produce the final comparison table and charts.

---

## Evaluation Metrics

**RMSE** — standard prediction error in flight cycles. Lower is better.

**NASA Asymmetric Score** — penalises late predictions (engine fails before maintenance) more than early ones:
```
d = predicted_RUL − true_RUL

Early (d < 0):  penalty = exp(−d / 13) − 1   ← grows slowly
Late  (d > 0):  penalty = exp( d / 10) − 1   ← grows fast
```
Total score = sum of penalties across all test engines. Lower is better.

**R²** — proportion of RUL variance explained. Closer to 1.0 is better.

---

## Contributors

| Name | ID |
|------|----|
| Dhruv Parmar | 202518030 |
| Ummesalma Diwan | 202518017 |
| Aditya Jana | 202518035 |
| Shrey Pandya | 202518045 |

*Group 13 · IT 402 Applied Forecasting Methods · Dhirubhai Ambani University*
