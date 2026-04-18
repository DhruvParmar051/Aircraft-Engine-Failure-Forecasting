"""
metrics.py — evaluation metrics for RUL prediction
all models must use these functions — results are only comparable if computed identically

RMSE      — symmetric; standard regression baseline
NASA score — asymmetric; late predictions penalised more heavily than early ones
             d < 0 (early): exp(-d/13) - 1    → slow penalty
             d >= 0 (late):  exp(d/10)  - 1    → fast penalty
             total = sum over samples (lower is better, 0 is perfect)
"""

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """root mean squared error — symmetric, scale-dependent"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    official NASA CMAPSS asymmetric scoring function
    d = predicted - actual (positive = late prediction = more dangerous)
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(scores))


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
    verbose: bool = True,
) -> dict[str, float]:
    """
    compute RMSE and NASA score; optionally print summary
    returns dict with 'rmse' and 'nasa_score' keys
    """
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float32).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    results = {
        "rmse":       rmse(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
    }

    if verbose:
        print(f"  [{model_name}] RMSE: {results['rmse']:.4f}  |  NASA Score: {results['nasa_score']:.2f}")

    return results


def evaluate_per_subset(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_ids: np.ndarray,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    compute RMSE and NASA score per FD subset (1–4) plus overall
    dataset_ids comes directly from create_windows() or create_last_window_per_engine()
    returns DataFrame: dataset_id | rmse | nasa_score
    """
    rows = []
    for did in sorted(np.unique(dataset_ids)):
        mask = dataset_ids == did
        r = evaluate(y_true[mask], y_pred[mask], model_name=f"{model_name}_FD00{did}", verbose=True)
        r["dataset_id"] = int(did)
        rows.append(r)

    overall = evaluate(y_true, y_pred, model_name=f"{model_name}_OVERALL", verbose=True)
    overall["dataset_id"] = "ALL"
    rows.append(overall)

    return pd.DataFrame(rows)[["dataset_id", "rmse", "nasa_score"]]


def summarise_all_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    build the final comparison table
    input: {model_name: {'rmse': ..., 'nasa_score': ...}}
    returns DataFrame ranked by RMSE ascending
    """
    rows = [{"model": name, **scores} for name, scores in results.items()]
    df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    return df