"""
uncertainty.py — centralized uncertainty estimation for all model types.

Consolidates functionality previously scattered across classical.py and
deep_learning.py into a single module that any model can import.

Functions
---------
Classical models (SARIMAX conf_int):
    sarimax_ci_to_rul_bounds  — convert SARIMAX forecast CI to RUL (lo, point, hi)

DL point models (MC Dropout):
    MCDropout                 — wrapper keeping dropout active at eval()
    mc_dropout_predict        — run N stochastic forward passes → (Q10, Q50, Q90)

Calibration (conformal):
    conformal_calibrate       — compute additive margin delta from calibration set
    apply_conformal_margin    — expand intervals by delta, re-clip to [0, RUL_CAP]

All functions operate on numpy arrays and are framework-agnostic at the
calling level — they import torch only when needed.
"""

from __future__ import annotations

import warnings
import numpy as np

from src.utils.config import RUL_CAP, DL_CONFIG, EVAL_CONFIG


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
# DL: MONTE CARLO DROPOUT
# ══════════════════════════════════════════════════════════════════════════════


class MCDropout:
    """
    Monte Carlo Dropout wrapper for any PyTorch point-prediction model.

    Keeps dropout active at eval() time, turning each forward pass into
    a sample from the approximate posterior (Gal & Ghahramani, 2016).

    Usage
    -----
        mc = MCDropout(model, p_drop=0.1)
        mc.enable()            # activate dropout at eval time
        q10, q50, q90, std = mc.predict(X_test)
    """

    def __init__(self, model, p_drop: float = 0.1):
        import torch.nn as nn
        self._model  = model
        self._p_drop = p_drop
        self._extra_dropout = nn.Dropout(p=p_drop)

    def enable(self):
        """Force all Dropout layers in the base model to stay active."""
        import torch.nn as nn
        def _activate(m):
            if isinstance(m, nn.Dropout):
                m.train()
        self._model.apply(_activate)

    def predict(
        self,
        X_test: np.ndarray,
        n_samples: int = DL_CONFIG["mc_dropout_samples"],
        quantiles: list[float] = None,
        batch_size: int = DL_CONFIG["batch_size"],
        device=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run MC Dropout inference and return empirical quantile bands.

        Parameters
        ----------
        X_test    : (n_engines, window_size, n_features) float32
        n_samples : stochastic forward passes (default 30)
        quantiles : (q_low, q_mid, q_high), default (0.10, 0.50, 0.90)

        Returns
        -------
        q_low, q_mid, q_high, std_pred : (n_engines,) each
        """
        import torch

        if quantiles is None:
            quantiles = DL_CONFIG["quantiles"]

        if device is None:
            device = next(self._model.parameters()).device

        self._model.eval()
        self.enable()

        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        dataset  = torch.utils.data.TensorDataset(X_tensor)
        loader   = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                batch_preds = []
                for (X_b,) in loader:
                    out = self._extra_dropout(self._model(X_b.to(device)))
                    out = torch.clamp(out, 0, RUL_CAP)
                    batch_preds.append(out.cpu().numpy().ravel())
                samples.append(np.concatenate(batch_preds))

        arr      = np.stack(samples, axis=0)         # (n_samples, n_engines)
        q_vals   = np.quantile(arr, quantiles, axis=0)
        std_pred = arr.std(axis=0)

        q_low, q_mid, q_high = q_vals[0], q_vals[1], q_vals[2]

        # Bug-detection: ordering guarantee
        assert np.all(q_low  <= q_mid  + 1e-4), "MC Dropout: q_low > q_mid"
        assert np.all(q_mid  <= q_high + 1e-4), "MC Dropout: q_mid > q_high"

        return q_low, q_mid, q_high, std_pred


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
