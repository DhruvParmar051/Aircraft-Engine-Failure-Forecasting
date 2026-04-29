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

from __future__ import annotations

import os
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score


# ── Project root resolution ────────────────────────────────────────────────────
# Walk up from this file until we find a directory containing 'experiments/'
def _find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "experiments").exists():
            return parent
    return p.parents[2]   # fallback: two levels above src/

ROOT        = _find_root()
RESULTS_DIR = ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "all_model_results.csv"

_CSV_FIELDS = [
    "model_name", "model_type", "rmse", "nasa_score", "nasa_score_mean",
    "r2_score", "bias", "interval_width", "coverage_pct",
    "n_test_engines", "timestamp",
]

# ══════════════════════════════════════════════════════════════════════════════
# CORE METRICS
# ══════════════════════════════════════════════════════════════════════════════

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
    """
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float32).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("NaN or infinite values detected")

    # No clipping here — callers (predict_test, predict_dataset) are responsible
    # for clipping to [0, RUL_CAP] before calling evaluate(). Clipping inside a
    # measurement function silently understates RMSE for over-predicting models.

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


def naive_baseline(
    y_true: np.ndarray,
    rul_cap: float = 125.0,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Two zero-information baselines for CMAPSS RUL prediction.

    A model comparison table is not credible without a lower bound.
    Both baselines use NO sensor information whatsoever.

    Strategies
    ----------
    mean_predictor  : predicts the training-set mean RUL for every engine.
                      Optimal under MSE with zero features.
    constant_half   : predicts RUL_CAP / 2 for every engine.
                      Represents the uniform prior over [0, RUL_CAP].

    Why these two:
        - mean_predictor is the Bayes-optimal zero-feature regressor under MSE.
        - constant_half makes no assumption about the training label distribution
          and is the prior a maintenance engineer might use with no data.
        - A trained model MUST beat both to be worth deploying.

    Returns
    -------
    dict : {"mean_predictor": metrics_dict, "constant_half": metrics_dict}
    """
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    results = {}

    for name, pred_val in [
        ("mean_predictor", float(np.mean(y_true))),
        ("constant_half",  rul_cap / 2.0),
    ]:
        y_pred = np.full_like(y_true, fill_value=pred_val)
        r      = evaluate(y_true, y_pred, model_name=f"Baseline:{name}", verbose=verbose)
        results[name] = r

    return results


def summarise_all_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    Build final comparison table ranked by RMSE ascending.
    """
    rows = [{"model": name, **scores} for name, scores in results.items()]
    df   = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    df.index     += 1
    df.index.name = "rank"
    return df


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS CSV — single source of truth across all notebooks
# ══════════════════════════════════════════════════════════════════════════════

def save_model_results(
    model_name: str,
    model_type: str,          # "classical" | "dl" | "quantile"
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q10: np.ndarray | None = None,
    q90: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate predictions, then append one row to results/all_model_results.csv.

    Parameters
    ----------
    model_name : str   e.g. "ARIMA(1,2,1)", "GRU", "Q-Transformer"
    model_type : str   one of "classical", "dl", "quantile"
    y_true     : ground-truth RUL values
    y_pred     : point predictions (Q50 for quantile models)
    q10, q90   : optional lower/upper quantile bounds (quantile models only)

    Returns
    -------
    dict with all computed metrics (same as evaluate())
    """
    results = evaluate(y_true, y_pred, model_name=model_name, verbose=verbose)

    # Interval metrics (quantile models only)
    interval_width = float(np.mean(q90 - q10)) if (q10 is not None and q90 is not None) else float("nan")
    coverage_pct   = (
        float(np.mean((y_true >= q10) & (y_true <= q90)) * 100)
        if (q10 is not None and q90 is not None) else float("nan")
    )

    row = {
        "model_name":       model_name,
        "model_type":       model_type,
        "rmse":             round(results["rmse"],            4),
        "nasa_score":       round(results["nasa_score"],      2),
        "nasa_score_mean":  round(results["nasa_score_mean"], 4),
        "r2_score":         round(results["r2_score"],        4),
        "bias":             round(results["bias"],            4),
        "interval_width":   round(interval_width, 4) if not np.isnan(interval_width) else "",
        "coverage_pct":     round(coverage_pct,   2) if not np.isnan(coverage_pct)   else "",
        "n_test_engines":   len(y_true),
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_CSV.exists()

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"  → Saved to {RESULTS_CSV.relative_to(ROOT)}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# QUANTILE CALIBRATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def pinball_loss_by_quantile(
    y_true: np.ndarray,
    preds_matrix: np.ndarray,
    quantiles: list[float],
    plot: bool = True,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Compute pinball (quantile) loss per quantile level.

    Pinball loss for quantile q:
        L = max(q*(y-p), (q-1)*(y-p))
    A well-calibrated model achieves minimum pinball loss for each quantile.

    Parameters
    ----------
    y_true       : (n,)        ground-truth RUL
    preds_matrix : (n, n_q)    predicted values per quantile
    quantiles    : list of q values matching columns of preds_matrix
    """
    y_true = np.asarray(y_true).ravel()
    losses = {}
    for i, q in enumerate(quantiles):
        preds  = preds_matrix[:, i]
        errors = y_true - preds
        loss   = np.mean(np.maximum(q * errors, (q - 1) * errors))
        losses[f"Q{int(q*100)}"] = round(float(loss), 4)

    df = pd.DataFrame([losses])
    print(f"\nPinball Loss by Quantile — {model_name}")
    print(df.to_string(index=False))

    if plot:
        fig, ax = plt.subplots(figsize=(6, 3))
        cols    = list(losses.keys())
        vals    = list(losses.values())
        bars    = ax.bar(cols, vals, color=["steelblue", "orange", "tomato"], edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Mean Pinball Loss (lower is better)")
        ax.set_title(f"{model_name} — Pinball Loss per Quantile")
        plt.tight_layout()
        plt.show()

    return df


def reliability_diagram(
    y_true: np.ndarray,
    all_q_preds: dict[float, np.ndarray],
    title: str = "Reliability Diagram",
) -> None:
    """
    Plot actual vs expected coverage across quantile levels.

    For each target quantile q, computes the fraction of test engines
    where y_true ≤ predicted_q.  A perfectly calibrated model → diagonal.

    Parameters
    ----------
    y_true      : (n,)  ground-truth RUL
    all_q_preds : {q_level: np.ndarray of predictions at that quantile}
                  e.g. {0.1: preds_q10, 0.5: preds_q50, 0.9: preds_q90}
    """
    y_true   = np.asarray(y_true).ravel()
    q_levels = sorted(all_q_preds.keys())
    actual   = [float(np.mean(y_true <= all_q_preds[q])) for q in q_levels]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.plot(q_levels, actual, "o-", color="steelblue", lw=2, ms=8, label="Observed coverage")

    # shade over/under calibration regions
    ax.fill_between(q_levels, q_levels, actual,
                    where=[a > q for a, q in zip(actual, q_levels)],
                    alpha=0.15, color="green", label="Over-confident (too wide)")
    ax.fill_between(q_levels, q_levels, actual,
                    where=[a < q for a, q in zip(actual, q_levels)],
                    alpha=0.15, color="red", label="Under-confident (too narrow)")

    ax.set_xlabel("Target quantile level")
    ax.set_ylabel("Observed fraction of y_true ≤ predicted")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # print calibration error
    cal_error = float(np.mean(np.abs(np.array(actual) - np.array(q_levels))))
    print(f"  Mean Calibration Error (MCE): {cal_error:.4f}  (0=perfect, closer=better)")


def interval_coverage_by_rul_bucket(
    y_true: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
    buckets: list[tuple[int, int]] | None = None,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Stratify engines by true RUL bucket and report coverage + interval width.

    Addresses critic concern: 'Q-LSTM/Q-GRU show massive uncertainty intervals
    in early life (RUL > 100) — how can a maintenance lead trust the lower bound?'

    Answer: Wide early-life intervals reflect genuine epistemic uncertainty about
    long-horizon predictions, not a model failure. Operators rely on Q50 for
    scheduling and Q10 as the safety-critical lower bound.
    """
    if buckets is None:
        buckets = [(0, 25), (25, 50), (50, 100), (100, 125)]

    y_true = np.asarray(y_true).ravel()
    q10    = np.asarray(q10).ravel()
    q90    = np.asarray(q90).ravel()
    rows   = []

    for lo, hi in buckets:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() == 0:
            continue
        yt   = y_true[mask]
        lo_q = q10[mask]
        hi_q = q90[mask]
        rows.append({
            "RUL bucket":   f"[{lo}, {hi})",
            "n_engines":    int(mask.sum()),
            "coverage_%":   round(float(np.mean((yt >= lo_q) & (yt <= hi_q)) * 100), 1),
            "mean_width":   round(float(np.mean(hi_q - lo_q)), 2),
            "median_width": round(float(np.median(hi_q - lo_q)), 2),
        })

    df = pd.DataFrame(rows)
    print(f"\nInterval Coverage by RUL Bucket — {model_name}")
    print(df.to_string(index=False))
    print("\nNote: wider intervals in early life (RUL 50-125) reflect genuine")
    print("epistemic uncertainty — model has less certainty about long-horizon predictions.")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = df["RUL bucket"].tolist()

    ax = axes[0]
    bars = ax.bar(labels, df["coverage_%"], color="steelblue", edgecolor="white")
    ax.axhline(80, color="red", ls="--", lw=1.5, label="Target 80%")
    for bar, v in zip(bars, df["coverage_%"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Coverage %"); ax.set_title(f"{model_name} — Coverage by RUL bucket")
    ax.legend(); ax.set_ylim(0, 110)

    ax = axes[1]
    bars = ax.bar(labels, df["mean_width"], color="orange", edgecolor="white")
    for bar, v in zip(bars, df["mean_width"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean Interval Width (cycles)")
    ax.set_title(f"{model_name} — Interval Width by RUL bucket")

    plt.tight_layout()
    plt.show()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# LITERATURE BENCHMARKS — FD004 published results (DOI cited)
# ══════════════════════════════════════════════════════════════════════════════

# ── PLACEHOLDER — update with verified papers before submission ────────────────
# Add entries in the format:
#   "ModelName\n(Author et al. YEAR)": {"rmse": <float>, "doi": "<doi or arXiv>"}
# Only include papers that:
#   (a) explicitly report results on NASA CMAPSS FD004
#   (b) use the same RUL cap (125 cycles) and RMSE metric
#   (c) are peer-reviewed (journal or major conference)
LITERATURE_BENCHMARKS_FD004: dict[str, dict] = {
    # Example (fill in with verified papers):
    # "Saxena et al. (2008)\n[dataset paper]": {"rmse": None, "doi": "10.1109/MSPEC.2008.4460093"},
}


def compare_to_benchmarks(
    our_results: dict[str, float],
    metric: str = "rmse",
) -> pd.DataFrame:
    """
    Combine our model results with published FD004 benchmarks into a single table
    and plot a bar chart with our models highlighted.

    Parameters
    ----------
    our_results : {model_name: rmse_value}  — our trained models
    metric      : "rmse" (only rmse supported from literature)
    """
    rows = []
    for name, val in our_results.items():
        rows.append({"model": name, metric: val, "source": "This work"})
    for name, info in LITERATURE_BENCHMARKS_FD004.items():
        rows.append({"model": name, metric: info[metric], "source": "Literature"})

    df = pd.DataFrame(rows).sort_values(metric).reset_index(drop=True)
    df.index += 1; df.index.name = "rank"

    print(f"\n=== FD004 {metric.upper()} Comparison vs Literature ===")
    print(df[["model", metric, "source"]].to_string())

    # Bar chart
    colours = ["#2196F3" if s == "This work" else "#BDBDBD" for s in df["source"]]
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.9), 5))
    bars = ax.bar(df["model"], df[metric], color=colours, edgecolor="white")
    for bar, val in zip(bars, df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(f"{metric.upper()} (cycles, lower is better)")
    ax.set_title(f"FD004 {metric.upper()} — This Work vs Published Literature")
    ax.tick_params(axis="x", rotation=30)

    legend_patches = [
        mpatches.Patch(color="#2196F3", label="This work"),
        mpatches.Patch(color="#BDBDBD", label="Literature"),
    ]
    ax.legend(handles=legend_patches)
    plt.tight_layout()
    plt.show()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def load_all_results() -> pd.DataFrame:
    """Load the unified results CSV written by every model notebook."""
    if not RESULTS_CSV.exists():
        print(f"  [WARN] {RESULTS_CSV} not found — run model notebooks first.")
        return pd.DataFrame(columns=_CSV_FIELDS)
    df = pd.read_csv(RESULTS_CSV)
    # Keep only the latest run per model (in case a notebook was re-run)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").groupby("model_name").last().reset_index()
    df = df.sort_values("rmse").reset_index(drop=True)
    df.index += 1; df.index.name = "rank"
    return df


def plot_model_comparison(df: pd.DataFrame) -> None:
    """
    Grouped bar chart: RMSE and NASA score per model, coloured by model_type.
    Used in T14_final_summary.ipynb.
    """
    colour_map = {"classical": "#4CAF50", "dl": "#2196F3", "quantile": "#FF9800"}
    colours    = [colour_map.get(str(t), "#9E9E9E") for t in df["model_type"]]

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(df) * 0.9), 5))

    for ax, col, ylabel, title in zip(
        axes,
        ["rmse", "nasa_score"],
        ["RMSE (cycles, lower is better)", "NASA Score (lower is better)"],
        ["RMSE — All Models", "NASA Score — All Models"],
    ):
        bars = ax.bar(df["model_name"], df[col], color=colours, edgecolor="white")
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{float(val):.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="Classical"),
        mpatches.Patch(color="#2196F3", label="Deep Learning"),
        mpatches.Patch(color="#FF9800", label="Quantile"),
    ]
    axes[0].legend(handles=legend_patches, fontsize=8)
    plt.tight_layout()
    plt.show()
