# Aircraft Engine Failure Forecasting — NASA CMAPSS

## Project Structure

```
cmapss/
├── data/
│   ├── raw/                         # T01 and T02 outputs
│   │   ├── train_FD001.txt          # original CMAPSS files (not committed)
│   │   ├── train_loaded.csv         # T01 output
│   │   ├── test_loaded.csv
│   │   ├── train_rul.csv            # T02 output
│   │   └── test_rul.csv
│   └── processed/                   # T03 output
│       ├── train_features.csv
│       └── test_features.csv
│
├── artifacts/                       # persisted sklearn objects for inference
│   ├── kmeans_op_clusters.pkl
│   └── scalers.pkl
│
├── experiments/
│   └── 01_data_pipeline/
│       ├── T01_data_loading.ipynb
│       ├── T02_rul_computation.ipynb
│       └── T03_feature_engineering.ipynb
│
├── src/
│   ├── preprocessing/
│   │   ├── load_data.py    # raw file loading + globally-unique engine ID offsets
│   │   ├── rul.py          # RUL computation (train + test), verification
│   │   ├── cleaning.py     # per-subset variance-based sensor selection
│   │   └── scaling.py      # per-(dataset_id, op_cluster) StandardScaler + artifact I/O
│   ├── features/
│   │   ├── rolling.py      # rolling mean/std per engine
│   │   └── windowing.py    # sliding window sequences for LSTM/GRU/TCN/TFT
│   └── evaluation/
│       └── metrics.py      # RMSE + NASA asymmetric score (shared by all models)
│
└── README.md
```

## Pipeline

```
T01  →  T02  →  T03  →  modeling experiments
```

| Notebook | Input | Output | Does |
|----------|-------|--------|------|
| T01 | `data/*.txt` | `train_loaded`, `test_loaded` | Load + combine all 4 FD subsets |
| T02 | `*_loaded` | `train_rul`, `test_rul` | Compute + cap RUL, drop rul_last |
| T03 | `*_rul` | `train_features`, `test_features` + `artifacts/` | Drop sensors, normalize, rolling features |

## Bugs Fixed

| Bug | Location | Fix |
|-----|----------|-----|
| MinMaxScaler: test values can exceed [0,1] bounds | `scaling.py` | Replaced with StandardScaler |
| `verify_train_rul` used positional `last()` not cycle-sorted last row | `rul.py` | Sort by cycle before groupby |
| Rolling windows could compute across engine boundaries if DataFrame unsorted | `rolling.py` | Sort by (engine_id, cycle) at top of function |
| Lambda closure bug: all windows captured same value in loop | `rolling.py` | Fixed with `lambda x, w=window` |
| `rul_last` leakage: column persisted into processed output | `rul.py` + T03 | Dropped in `compute_test_rul`, asserted absent in T03 |

## Improvements Added

| Improvement | Location |
|-------------|----------|
| Adaptive KMeans k: FD001/FD003 clamp to actual distinct op conditions | `scaling.py` |
| Artifact persistence: KMeans + scalers saved to `artifacts/` for inference | `scaling.py` + T03 |
| Shared windowing utility: one source of truth for LSTM/GRU/TCN/TFT sequences | `src/features/windowing.py` |
| Shared evaluation: RMSE + NASA score functions for all models | `src/evaluation/metrics.py` |

## Shared Conventions

| Setting | Value |
|---------|-------|
| RUL cap | 125 cycles |
| Sequence window | 30 cycles |
| Train/val split | 80/20 by engine_id (use `windowing.split_by_engine`) |
| Random seed | 42 |
| Dropped sensors | s1, s5, s6, s10, s16, s18, s19 |
| Scaler | StandardScaler per (dataset_id, op_cluster) |
| Rolling windows | 5, 10 cycles (mean + std each) |