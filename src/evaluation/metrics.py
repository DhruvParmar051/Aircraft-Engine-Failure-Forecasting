"""
metrics.py — evaluation metrics for RUL prediction
all models must use these functions — results are only comparable if computed identically

RMSE       — symmetric; standard regression baseline
NASA score — asymmetric; late predictions penalised more heavily than early ones
             d < 0 (early): exp(-d/13) - 1    → slow penalty
             d >= 0 (late):  exp(d/10)  - 1    → fast penalty
             total = sum over samples (lower is better, 0 is perfect)
Bias       — mean signed error (pred - true); positive = predicting too late
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error — symmetric, scale-dependent."""
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Official NASA CMAPSS asymmetric scoring function.
    d = predicted - actual (positive = late = more dangerous → steeper penalty)
    """
    d = np.clip(y_pred - y_true, -100, 100)
    scores = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1,   # early prediction — slow penalty
        np.exp(d / 10.0)  - 1,   # late prediction  — fast penalty
    )
    return float(np.sum(scores))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean signed error: mean(pred - true).
    positive → model predicts too late  (over-estimates RUL)  → dangerous
    negative → model predicts too early (under-estimates RUL) → conservative
    """
    return float(np.mean(y_pred - y_true))


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
    verbose: bool = True,
) -> dict[str, float]:
    """
    Compute RMSE, NASA score, R2, and bias for a single set of predictions.
    Returns dict with keys: rmse, nasa_score, nasa_score_mean, r2_score, bias.

    Usage:
        results = evaluate(y_true, y_pred, model_name="AR(10)")
    """
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float32).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("NaN or infinite values detected")

    y_pred = np.clip(y_pred, 0, 125)

    ns = nasa_score(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    b  = bias(y_true, y_pred)

    results = {
        "rmse":            rmse(y_true, y_pred),
        "nasa_score":      ns,
        "nasa_score_mean": ns / len(y_true),
        "r2_score":        r2,
        "bias":            b,
    }

    if verbose:
        direction = "late ↑" if b > 0 else "early ↓"
        print(
            f"  [{model_name}] "
            f"RMSE: {results['rmse']:.4f}  |  "
            f"NASA Score: {results['nasa_score']:.2f} (mean: {results['nasa_score_mean']:.2f})  |  "
            f"R2: {results['r2_score']:.4f}  |  "
            f"Bias: {b:+.2f} ({direction})"
        )

    return results


def summarise_all_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    Build final comparison table ranked by RMSE ascending.

    Usage:
        all_results = {
            "AR(10)":       evaluate(y_true, y_pred_ar,    verbose=False),
            "ARMA(5,3)":    evaluate(y_true, y_pred_arma,  verbose=False),
            "ARIMA(5,2,3)": evaluate(y_true, y_pred_arima, verbose=False),
        }
        display(summarise_all_models(all_results))
    """
    rows = [{"model": name, **scores} for name, scores in results.items()]
    df   = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    df.index     += 1
    df.index.name = "rank"
    return df