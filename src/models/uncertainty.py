"""
uncertainty.py — centralized uncertainty estimation for all model types.

Functions
---------
Classical models (SARIMAX conf_int):
    sarimax_ci_to_rul_bounds  — convert SARIMAX forecast CI to RUL (lo, point, hi)

Calibration (conformal):
    conformal_calibrate       — compute additive margin delta from calibration set
    apply_conformal_margin    — expand intervals by delta, re-clip to [0, RUL_CAP]

All functions operate on numpy arrays and are framework-agnostic at the
calling level.
"""

from __future__ import annotations

import warnings
import numpy as np

from src.utils.config import RUL_CAP, EVAL_CONFIG


# ══════════════════════════════════════════════════════════════════════════════
# CLASSICAL: SARIMAX FORECAST CI → RUL BOUNDS
# ══════════════════════════════════════════════════════════════════════════════


def sarimax_ci_to_rul_bounds(
    forecast_mean: np.ndarray,
    ci: np.ndarray,
    observed_series: np.ndarray,
    threshold: float,
    alpha: float = 0.20,
) -> tuple[float, float, float]:
    """
    Convert SARIMAX forecast + confidence interval into (lower, point, upper) RUL.

    The health_index rises toward failure → higher = more degraded.
    CI direction for safety:
        UPPER CI band reaches threshold sooner → smallest RUL → lower_bound.
        LOWER CI band reaches threshold later  → largest RUL → upper_bound.

    Parameters
    ----------
    forecast_mean : (n_steps,) mean forecast of health_index
    ci            : (n_steps, 2) confidence interval array [lower_col, upper_col]
                    from statsmodels get_forecast().conf_int(alpha=alpha)
    observed_series : full smoothed health_index history for this engine
    threshold     : health_index failure threshold
    alpha         : CI level (0.20 = 80% CI, matching EVAL_CONFIG)

    Returns
    -------
    (rul_lower, rul_point, rul_upper)
        rul_lower ≤ rul_point ≤ rul_upper (guaranteed)
    """
    ci_arr = np.asarray(ci)
    ci_lo  = ci_arr[:, 0]   # lower band (optimistic → crosses threshold later)
    ci_hi  = ci_arr[:, 1]   # upper band (pessimistic → crosses threshold sooner)

    rul_point = _threshold_crossing(forecast_mean, observed_series, threshold)
    rul_lower = _threshold_crossing(ci_hi,         observed_series, threshold)
    rul_upper = _threshold_crossing(ci_lo,          observed_series, threshold)

    # Guarantee ordering
    rul_lower = min(rul_lower, rul_point)
    rul_upper = max(rul_upper, rul_point)

    return float(rul_lower), float(rul_point), float(rul_upper)


def _threshold_crossing(
    forecast: np.ndarray,
    observed: np.ndarray,
    threshold: float,
) -> float:
    """
    Step count until forecast first crosses threshold.
    Falls back to linear extrapolation if no crossing within forecast horizon.
    """
    forecast = np.asarray(forecast, dtype=float)

    if forecast.size == 0 or not np.all(np.isfinite(forecast)):
        return _linear_extrapolation(observed, threshold)

    # Already past threshold → EOL now
    if float(observed[-1]) >= threshold:
        return _linear_extrapolation(observed, threshold, tail_frac=0.2)

    crossings = np.where(forecast >= threshold)[0]
    if crossings.size > 0:
        return float(max(crossings[0], 3))

    # No crossing: extrapolate from early-window slope
    early_n = min(30, len(forecast))
    slope, _ = np.polyfit(np.arange(early_n), forecast[:early_n], 1)

    if slope > 1e-4:
        gap   = threshold - float(forecast[early_n - 1])
        steps = early_n + gap / slope
        if steps <= 200:
            return float(np.clip(steps, 0.0, RUL_CAP))

    return _linear_extrapolation(observed, threshold)


def _linear_extrapolation(
    series: np.ndarray,
    threshold: float,
    tail_frac: float = 0.20,
) -> float:
    """Extrapolate from recent slope of observed series."""
    tail  = max(5, min(30, int(tail_frac * len(series))))
    y     = series[-tail:] if len(series) >= tail else series
    if len(y) < 3:
        return float(RUL_CAP)
    x           = np.arange(len(y), dtype=float)
    slope, last = np.polyfit(x, y, 1)[0], float(y[-1])
    if slope <= 1e-4:
        return float(RUL_CAP)
    return float(np.clip((threshold - last) / slope, 0.0, RUL_CAP))


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION: CONFORMAL PREDICTION
# ══════════════════════════════════════════════════════════════════════════════


def conformal_calibrate(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    target_coverage: float = EVAL_CONFIG["conformal_target"],
) -> float:
    """
    Split conformal calibration: compute additive margin delta.

    Given a calibration set (engines not used in training), finds the
    smallest delta such that the expanded intervals [lower-delta, upper+delta]
    achieve the target marginal coverage.

    This provides a distribution-free, finite-sample coverage guarantee
    on exchangeable data — no assumptions about the error distribution.

    Non-conformity score: max(lower - true, true - upper, 0)
    = 0 when true is inside the interval, else the gap.

    Parameters
    ----------
    y_true, y_lower, y_upper : (n_cal,) arrays from calibration engines
    target_coverage          : desired marginal coverage (default 0.80)

    Returns
    -------
    delta : additive margin to expand both bounds.
    """
    y_true  = np.asarray(y_true,  dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)

    scores = np.maximum(
        np.maximum(y_lower - y_true, y_true - y_upper), 0.0
    )
    n     = len(scores)
    level = min(np.ceil((n + 1) * target_coverage) / n, 1.0)
    delta = float(np.quantile(scores, level))

    coverage_before = float(np.mean((y_true >= y_lower) & (y_true <= y_upper)) * 100)
    print(
        f"  [Conformal] n_cal={n}  target={target_coverage*100:.0f}%  "
        f"delta={delta:.3f}  coverage_before={coverage_before:.1f}%"
    )
    return delta


def apply_conformal_margin(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    delta: float,
    rul_cap: float = RUL_CAP,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand intervals by delta (additive symmetric), then re-clip to [0, rul_cap].

    Returns (calibrated_lower, calibrated_upper).
    """
    cal_lower = np.clip(np.asarray(y_lower, dtype=float) - delta, 0.0, rul_cap)
    cal_upper = np.clip(np.asarray(y_upper, dtype=float) + delta, 0.0, rul_cap)
    return cal_lower.astype(np.float32), cal_upper.astype(np.float32)


def calibrate_and_report(
    model_name: str,
    y_cal_true: np.ndarray,
    y_cal_lower: np.ndarray,
    y_cal_upper: np.ndarray,
    y_test_lower: np.ndarray,
    y_test_upper: np.ndarray,
    y_test_true: np.ndarray | None = None,
    target: float = EVAL_CONFIG["conformal_target"],
) -> dict:
    """
    End-to-end conformal calibration: compute delta on cal set,
    apply to test set, report coverage improvement.

    Returns dict with delta, cal_lower, cal_upper (test set),
    and coverage statistics.
    """
    delta = conformal_calibrate(
        y_cal_true, y_cal_lower, y_cal_upper, target_coverage=target
    )
    cal_lower, cal_upper = apply_conformal_margin(y_test_lower, y_test_upper, delta)

    result = {
        "model_name":  model_name,
        "delta":       delta,
        "cal_lower":   cal_lower,
        "cal_upper":   cal_upper,
        "interval_width_after": float(np.mean(cal_upper - cal_lower)),
    }

    if y_test_true is not None:
        y_test_true = np.asarray(y_test_true)
        before = float(
            np.mean(
                (y_test_true >= np.asarray(y_test_lower))
                & (y_test_true <= np.asarray(y_test_upper))
            ) * 100
        )
        after = float(
            np.mean((y_test_true >= cal_lower) & (y_test_true <= cal_upper)) * 100
        )
        result.update({"coverage_before": before, "coverage_after": after})
        print(
            f"  [{model_name}] Test coverage: "
            f"{before:.1f}% → {after:.1f}%  (target {target*100:.0f}%)"
        )

    return result
