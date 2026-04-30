#!/usr/bin/env bash
# run_all_notebooks.sh — execute all experiment notebooks in order and store outputs.
# Run with: bash run_all_notebooks.sh > logs/run_all.log 2>&1

set -euo pipefail

PROJ="/Users/dhruvparmar/DAU/sem_2/IT_402_Applied_Forecasting_Methods/Project/Aircraft Engine Failure Forecasting"
JUPYTER="/opt/anaconda3/envs/dl/bin/jupyter"
PYTHON="/opt/anaconda3/envs/dl/bin/python"
LOG_DIR="$PROJ/logs"
RESULTS_DIR="$PROJ/results"

mkdir -p "$LOG_DIR" "$RESULTS_DIR/predictions"

NB_EXECUTE="$JUPYTER nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=1800"

run_nb() {
    local nb="$1"
    local nb_dir
    nb_dir="$(dirname "$nb")"
    local nb_name
    nb_name="$(basename "$nb")"
    local log="$LOG_DIR/${nb_name%.ipynb}.log"

    echo "━━━ $(date '+%H:%M:%S')  START  $nb_name ━━━"
    if $NB_EXECUTE \
        --ExecutePreprocessor.cwd="$nb_dir" \
        "$nb" > "$log" 2>&1; then
        echo "    $(date '+%H:%M:%S')  OK     $nb_name"
    else
        echo "    $(date '+%H:%M:%S')  FAIL   $nb_name  (see $log)"
    fi
}

echo "═══════════════════════════════════════════════════"
echo "  Notebook execution run — $(date)"
echo "═══════════════════════════════════════════════════"

# ── 01 Data pipeline ─────────────────────────────────────────────────────────
run_nb "$PROJ/experiments/01_data_pipeline/T01_data_loading.ipynb"
run_nb "$PROJ/experiments/01_data_pipeline/T02_rul_computation.ipynb"
run_nb "$PROJ/experiments/01_data_pipeline/T03_eda.ipynb"
run_nb "$PROJ/experiments/01_data_pipeline/T04_feature_engineering.ipynb"

# ── 02 Classical models ──────────────────────────────────────────────────────
run_nb "$PROJ/experiments/02_classical_models/T08_AR_model_book.ipynb"
run_nb "$PROJ/experiments/02_classical_models/T09_ARMA_model_book.ipynb"
run_nb "$PROJ/experiments/02_classical_models/T10_ARIMA_model_book.ipynb"

# ── 03 DL point models ───────────────────────────────────────────────────────
run_nb "$PROJ/experiments/03_DL_Models/GRU.ipynb"
run_nb "$PROJ/experiments/03_DL_Models/LSTM.ipynb"
run_nb "$PROJ/experiments/03_DL_Models/RNN.ipynb"
run_nb "$PROJ/experiments/03_DL_Models/MLP.ipynb"
run_nb "$PROJ/experiments/03_DL_Models/Transformer.ipynb"

# ── 04 Quantile models ───────────────────────────────────────────────────────
run_nb "$PROJ/experiments/04_quantile_models/Q_GRU.ipynb"
run_nb "$PROJ/experiments/04_quantile_models/Q_LSTM.ipynb"
run_nb "$PROJ/experiments/04_quantile_models/Q_RNN.ipynb"
run_nb "$PROJ/experiments/04_quantile_models/Q_MLP.ipynb"
run_nb "$PROJ/experiments/04_quantile_models/Q_Transformer.ipynb"

# ── 05 Robustness ────────────────────────────────────────────────────────────
run_nb "$PROJ/experiments/05_robustness/T13_ablation_robustness.ipynb"

# ── 06 Summary ───────────────────────────────────────────────────────────────
run_nb "$PROJ/experiments/06_summary/T14_final_summary.ipynb"

# ── 07 Calibration ───────────────────────────────────────────────────────────
run_nb "$PROJ/experiments/07_calibration/T15_calibration.ipynb"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  All done — $(date)"
echo "  Results: $RESULTS_DIR/all_model_results.csv"
echo "  Logs:    $LOG_DIR/"
echo "═══════════════════════════════════════════════════"

# Print results table if available
if [ -f "$RESULTS_DIR/all_model_results.csv" ]; then
    echo ""
    echo "── Model Results ──────────────────────────────────"
    "$PYTHON" -c "
import pandas as pd
df = pd.read_csv('$RESULTS_DIR/all_model_results.csv')
# keep last run per model
df = df.drop_duplicates('model_name', keep='last')
df = df.sort_values('rmse')
cols = ['model_name','model_type','rmse','r2_score','bias','coverage_pct','interval_width']
cols = [c for c in cols if c in df.columns]
print(df[cols].to_string(index=False))
"
fi
