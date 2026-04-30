"""
config.py — single source of truth for all project-wide constants.

Every notebook and module should import from here instead of defining
constants locally.  Centralising here means one change propagates everywhere.

Usage in notebooks
------------------
    from src.utils.config import ROOT, PROC_DIR, RUL_CAP, SENSOR_COLS, DL_CONFIG
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ── Project root resolution ────────────────────────────────────────────────────
def _find_root() -> Path:
    """
    Walk up from this file until we find the project root (contains experiments/).
    Works regardless of working directory — notebooks, scripts, or pytest invocations
    all resolve to the same root.
    """
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "experiments").exists():
            return parent
    return p.parents[3]  # fallback: three levels above src/utils/


ROOT     = _find_root()
DATA_DIR = ROOT / "data"
RAW_DIR  = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RESULTS_DIR  = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"


def ensure_src_on_path() -> None:
    """
    Add project root to sys.path if not already present.
    Call once at the top of every notebook instead of the 5-line boilerplate.

    Usage:
        from src.utils.config import ensure_src_on_path
        ensure_src_on_path()
    """
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# ── Sensor columns ─────────────────────────────────────────────────────────────
# 16 sensors retained after variance filtering (dropped s1, s5, s16, s18, s19)
SENSOR_COLS: list[str] = [
    f"s{i}" for i in [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 21]
]

# 14 sensors used by DL models (subset: drops s6, s10 which add noise for window models)
DL_SENSOR_COLS: list[str] = [
    f"s{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
]


# ── RUL / degradation ──────────────────────────────────────────────────────────
RUL_CAP:         int   = 125     # standard CMAPSS cap — focuses model on critical window
END_OF_LIFE_RUL: int   = 5       # cycles where engine is considered "at EOL"
SAFETY_FACTOR:   float = 0.88    # conservative multiplier for classical model predictions
                                  # val-set grid search on NASA loss → 0.88


# ── Classical model defaults ───────────────────────────────────────────────────
CLASSICAL_CONFIG = {
    "smooth_window":   10,    # rolling-median window before SARIMAX fit
    "rolling_window":  10,    # rolling-mean window for PCA features
    "n_pca_components": 2,    # FD004 has 2 fault modes → 2 PCA components
    "corr_threshold":  0.5,   # |Pearson r| threshold for sensor selection
    "recent_window_frac": 0.30,  # adaptive recency window fraction
    "recent_window_min":  20,
    "recent_window_max":  60,
    "max_horizon":     150,   # forecast steps for threshold crossing
    "failure_quantile": 0.05, # percentile of EOL health_index used as threshold
    "ar_p_candidates":  [1, 2, 3, 4, 5],
    "arma_p_candidates": [1, 2, 3],
    "arma_q_candidates": [1, 2, 3],
    "default_ar_p":  2,
    "default_arima_p": 1,
    "default_arima_d": 2,
    "default_arima_q": 2,
    "ci_alpha": 0.20,         # 1 - 0.20 = 80% CI for SARIMAX forecast
}


# ── Deep learning defaults ─────────────────────────────────────────────────────
DL_CONFIG = {
    "window_size":   30,      # sliding window length (val-set optimised)
    "batch_size":    128,
    "epochs":        50,
    "lr":            1e-3,
    "patience":      10,      # early-stopping patience
    "random_seed":   42,
    "hidden_size":   64,
    "n_layers":      2,
    "dropout":       0.2,
    "d_model":       64,      # Transformer model dimension
    "n_heads":       4,
    "dim_feedforward_mult": 4,  # dim_ff = d_model × this
    "mc_dropout_samples": 30,   # MC Dropout forward passes
    "quantiles":     [0.10, 0.50, 0.90],
}


# ── Evaluation targets ─────────────────────────────────────────────────────────
EVAL_CONFIG = {
    "coverage_target": 0.80,   # 80% prediction interval target
    "conformal_target": 0.80,  # conformal calibration target coverage
}
