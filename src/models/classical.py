"""
classical.py — AR / ARMA / ARIMA RUL prediction via health-index forecasting.

Pipeline (shared by all three models):

    1. Build a 1-D health_index from sensor rolling-means via PCA (PC1).
       Sign is flipped so that higher values correspond to more degraded engines.
    2. Compute a failure threshold from training cycles with RUL <= END_OF_LIFE_RUL
       (engines that are about to fail).
    3. For each test engine:
        a. Smooth its observed health trajectory (rolling median).
        b. Fit AR / ARMA / ARIMA on the smoothed history.
        c. Forecast MAX_HORIZON steps ahead.
        d. RUL = first forecast step that crosses the failure threshold.
        e. Fallback: linear extrapolation of the recent slope if the forecast
           never reaches the threshold within MAX_HORIZON.
    4. Clip predictions to [0, RUL_CAP].

This is a genuine forecasting approach — predicted RUL depends on the shape of
the forecasted curve, so AR, ARMA, and ARIMA produce different answers.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

RUL_CAP         = 125   # training-time RUL cap; predictions are also clipped to this
MAX_HORIZON     = 150   # forecast this many steps ahead looking for threshold crossing
SMOOTH_WINDOW   = 5     # rolling-median window applied to health_index before forecasting
END_OF_LIFE_RUL = 5     # training rows with RUL <= this define "failure region"

DEFAULT_AR_LAGS  = 10
DEFAULT_ARMA_P   = 2
DEFAULT_ARMA_Q   = 2
DEFAULT_ARIMA_P  = 2
DEFAULT_ARIMA_D  = 1
DEFAULT_ARIMA_Q  = 2


# ─────────────────────────────────────────────
# 1. HEALTH INDEX (PCA on rolling-mean sensors)
# ─────────────────────────────────────────────

def build_pca_health_index(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sensor_cols: list[str],
    rolling_window: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert sensor readings into a scalar health_index via PCA (PC1).

    If rolling-mean columns (e.g. ``s2_rmean_10``) are present in the DataFrame,
    they are used instead of raw sensors — this produces a much smoother curve,
    which is essential for time-series forecasting.

    Sign of PC1 is chosen so that ``health_index`` INCREASES as the engine
    degrades (i.e. the value is high near failure).
    """
    rolling_cols = [f"{s}_rmean_{rolling_window}" for s in sensor_cols]
    if all(c in train.columns for c in rolling_cols):
        input_cols = rolling_cols
    else:
        input_cols = sensor_cols

    pca = PCA(n_components=1, random_state=42)
    pca.fit(train[input_cols])

    train = train.copy()
    test  = test.copy()

    pc1_train = pca.transform(train[input_cols])[:, 0]
    pc1_test  = pca.transform(test[input_cols])[:, 0]

    # Force "higher = more degraded" using RUL (more robust than first/last cycle).
    near_fail = pc1_train[train["RUL"].values <= END_OF_LIFE_RUL].mean()
    healthy   = pc1_train[train["RUL"].values >= RUL_CAP - 5].mean()
    sign = 1.0 if near_fail > healthy else -1.0

    train["health_index"] = sign * pc1_train
    test["health_index"]  = sign * pc1_test

    return train, test


# ─────────────────────────────────────────────
# 2. FAILURE THRESHOLD
# ─────────────────────────────────────────────

def compute_failure_threshold(
    train_df: pd.DataFrame,
    end_of_life_rul: int = END_OF_LIFE_RUL,
    quantile: float = 0.5,
) -> float:
    """
    Failure threshold = typical health_index level when an engine is about to fail.

    Uses training rows with ``RUL <= end_of_life_rul`` and returns their
    ``quantile``-th health_index value. Default is the median — robust to outliers
    and represents a "typical" near-failure state.
    """
    mask = train_df["RUL"] <= end_of_life_rul
    if mask.sum() < 10:
        raise ValueError(
            f"Only {mask.sum()} training rows have RUL <= {end_of_life_rul}. "
            "Cannot compute threshold reliably."
        )
    return float(np.quantile(train_df.loc[mask, "health_index"], quantile))


# ─────────────────────────────────────────────
# 3. SMOOTHING
# ─────────────────────────────────────────────

def smooth_series(series: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Rolling-median smoothing (robust to sensor spikes)."""
    series = np.asarray(series, dtype=float)
    if window <= 1 or len(series) < 2:
        return series
    return pd.Series(series).rolling(window, min_periods=1, center=False).median().values


# ─────────────────────────────────────────────
# 4. FORECAST → RUL
# ─────────────────────────────────────────────

def _linear_extrapolation_rul(series: np.ndarray, threshold: float, tail: int = 30) -> float:
    """
    Fallback when the model forecast never crosses the threshold.
    Fit a linear trend to the last ``tail`` smoothed points and extrapolate.
    """
    y = series[-tail:] if len(series) >= tail else series
    if len(y) < 3:
        return float(RUL_CAP)

    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    last = float(y[-1])

    if slope <= 1e-6:          # series is flat or improving — engine far from failure
        return float(RUL_CAP)

    steps = (threshold - last) / slope
    if steps <= 0:             # already past threshold
        return 0.0
    return float(min(steps, RUL_CAP))


def _estimate_rul_from_forecast(
    preds: np.ndarray,
    observed: np.ndarray,
    threshold: float,
) -> float:
    """First forecast step that crosses the threshold, or linear fallback."""
    preds = np.asarray(preds, dtype=float)
    if preds.size == 0 or not np.all(np.isfinite(preds)):
        return _linear_extrapolation_rul(observed, threshold)

    # Already above threshold at the very first forecast step => RUL = 0.
    crossings = np.where(preds >= threshold)[0]
    if crossings.size > 0:
        return float(crossings[0])

    return _linear_extrapolation_rul(observed, threshold)


# ─────────────────────────────────────────────
# 5. AR MODEL
# ─────────────────────────────────────────────

def predict_rul_ar(
    series: np.ndarray,
    threshold: float,
    lags: int = DEFAULT_AR_LAGS,
    smooth_window: int = SMOOTH_WINDOW,
) -> float:
    """
    AR(p) forecast of the smoothed health_index; RUL = first step above threshold.
    Uses ``trend='c'`` so a constant mean is estimated — combined with the
    autoregressive structure this captures the drift of the degradation signal.
    """
    smoothed = smooth_series(series, smooth_window)

    if len(smoothed) <= lags + 2:
        return _linear_extrapolation_rul(smoothed, threshold)

    try:
        model = AutoReg(smoothed, lags=lags, trend="c", old_names=False).fit()
        preds = model.predict(start=len(smoothed), end=len(smoothed) + MAX_HORIZON - 1)
        return _estimate_rul_from_forecast(preds, smoothed, threshold)
    except Exception:
        return _linear_extrapolation_rul(smoothed, threshold)


# ─────────────────────────────────────────────
# 6. ARMA MODEL (ARIMA with d=0)
# ─────────────────────────────────────────────

def predict_rul_arma(
    series: np.ndarray,
    threshold: float,
    p: int = DEFAULT_ARMA_P,
    q: int = DEFAULT_ARMA_Q,
    smooth_window: int = SMOOTH_WINDOW,
) -> float:
    """
    ARMA(p,q) forecast. ``trend='t'`` adds a deterministic linear time trend,
    which is important because the health index drifts upward over time.
    """
    smoothed = smooth_series(series, smooth_window)

    if len(smoothed) <= p + q + 3:
        return _linear_extrapolation_rul(smoothed, threshold)

    try:
        model = StatsARIMA(smoothed, order=(p, 0, q), trend="t").fit()
        preds = model.forecast(steps=MAX_HORIZON)
        return _estimate_rul_from_forecast(preds, smoothed, threshold)
    except Exception:
        return _linear_extrapolation_rul(smoothed, threshold)


# ─────────────────────────────────────────────
# 7. ARIMA MODEL
# ─────────────────────────────────────────────

def predict_rul_arima(
    series: np.ndarray,
    threshold: float,
    p: int = DEFAULT_ARIMA_P,
    d: int = DEFAULT_ARIMA_D,
    q: int = DEFAULT_ARIMA_Q,
    smooth_window: int = SMOOTH_WINDOW,
) -> float:
    """
    ARIMA(p,d,q) forecast. For d>=1 we use ``trend='c'`` so a drift term is
    estimated on the differenced series — without it the forecast level would
    be flat (or worse, reverts to the mean of the differences = 0), and would
    never cross the failure threshold.
    """
    smoothed = smooth_series(series, smooth_window)

    if len(smoothed) <= p + d + q + 3:
        return _linear_extrapolation_rul(smoothed, threshold)

    # d == 0 -> deterministic trend in original scale ("t")
    # d >= 1 -> constant in the differenced series = linear drift in original scale ("c")
    trend = "t" if d == 0 else "c"

    try:
        model = StatsARIMA(smoothed, order=(p, d, q), trend=trend).fit()
        preds = model.forecast(steps=MAX_HORIZON)
        return _estimate_rul_from_forecast(preds, smoothed, threshold)
    except Exception:
        return _linear_extrapolation_rul(smoothed, threshold)


# ─────────────────────────────────────────────
# 8. DATASET-LEVEL PREDICTION
# ─────────────────────────────────────────────

def predict_dataset(
    df: pd.DataFrame,
    predict_fn: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a forecasting-based ``predict_fn`` to every engine in ``df``.

    ``predict_fn`` must already have its hyperparameters (threshold, lags, etc.)
    bound via ``functools.partial``.

    Returns (y_true, y_pred, dataset_ids), one entry per engine.
    """
    df = df.sort_values(["engine_id", "cycle"])

    y_true, y_pred, dids = [], [], []
    for _, g in df.groupby("engine_id", sort=False):
        series     = g["health_index"].values
        true_rul   = float(g["RUL"].iloc[-1])
        dataset_id = int(g["dataset_id"].iloc[0])

        pred = predict_fn(series)
        pred = float(np.clip(pred, 0.0, RUL_CAP))

        y_true.append(true_rul)
        y_pred.append(pred)
        dids.append(dataset_id)

    return np.array(y_true), np.array(y_pred), np.array(dids)


# ─────────────────────────────────────────────
# 9. HYPERPARAMETER SEARCH (simulated-test on training engines)
# ─────────────────────────────────────────────

def simulate_test_from_train(
    train_df: pd.DataFrame,
    cutoff_fraction: float = 0.6,
    random_seed: int = 42,
    max_engines: int | None = None,
) -> pd.DataFrame:
    """
    Turn a subset of training engines into a *simulated* test set by truncating
    each engine's history at a random cycle. The ground-truth RUL at the cutoff
    is known from the full trajectory, so any forecasting model can be scored
    without touching the real test set.

    This is the correct way to tune AR/ARMA/ARIMA hyperparameters.
    """
    rng  = np.random.default_rng(random_seed)
    rows = []

    engines = train_df["engine_id"].unique()
    if max_engines is not None and len(engines) > max_engines:
        engines = rng.choice(engines, size=max_engines, replace=False)

    for eid in engines:
        g = train_df[train_df["engine_id"] == eid].sort_values("cycle")
        n = len(g)
        if n < 20:
            continue
        # random cutoff in the middle portion of the engine's life
        lo = max(10, int(n * 0.3))
        hi = max(lo + 1, int(n * cutoff_fraction) + 1)
        cutoff = int(rng.integers(lo, min(hi, n)))
        rows.append(g.iloc[:cutoff])

    return pd.concat(rows, ignore_index=True)
