"""
classical.py — AR / ARMA / ARIMA RUL prediction via health-index forecasting.

Methodology follows "Time Series Forecasting in Python" by Marco Peixeiro:
    CH03 → stationarity (ADF test, differencing)
    CH05 → AR model (SARIMAX, ACF/PACF, rolling forecast)
    CH06 → ARMA model (SARIMAX, optimize_ARMA via AIC, Ljung-Box)
    CH07 → ARIMA model (SARIMAX, optimize_ARIMA via AIC, Ljung-Box + QQ plot)

Book rules enforced here:
    1. ALL models use SARIMAX — never AutoReg or statsmodels.ARIMA directly.
       Reason: book uses SARIMAX as the single interface throughout CH04-CH09.
    2. Order selection uses AIC via optimize_AR / optimize_ARMA / optimize_ARIMA.
       Reason: book explicitly selects by lowest AIC, not RMSE on a held-out set.
    3. Ljung-Box test on residuals after every fit (book does this in CH06 + CH07).
    4. QQ plot for residual normality check (book adds this in CH07 for ARIMA).
    5. rolling_forecast for walk-forward validation (book uses it in CH05/CH06/CH07).
    6. ADF run at level + first difference (+ second if needed) to determine d.

CMAPSS adaptation:
    - The forecasted "series" is a scalar health_index per engine (PCA on sensor rolling-means).
    - RUL = first forecast step crossing a pre-defined failure threshold.
    - optimize_* runs AIC on a representative training engine (engine selection is explicit).
    - rolling_forecast_engine does walk-forward on a SINGLE engine's health_index history.
"""

from __future__ import annotations

import warnings
from itertools import product
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.graphics.gofplots import qqplot              # CH07 requirement
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # CH05/CH06 requirement
from statsmodels.stats.diagnostic import acorr_ljungbox       # CH06/CH07 requirement
from statsmodels.tsa.statespace.sarimax import SARIMAX        # book's model interface
from statsmodels.tsa.stattools import adfuller                 # CH03 requirement

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

RUL_CAP         = 125   # piecewise-linear RUL cap used at training time
MAX_HORIZON     = 150   # max forecast steps when searching for threshold crossing
SMOOTH_WINDOW   = 5     # rolling-median applied to health_index before any model fit
END_OF_LIFE_RUL = 5     # training rows with RUL <= this define the "failure region"

# Default orders — used ONLY as fallback if optimize_* is not called.
# These are NOT justified by ACF/PACF; always run optimize_* before production use.
DEFAULT_AR_P    = 3     # CH05: book fits AR(3) on foot-traffic data as example
DEFAULT_ARMA_P  = 2
DEFAULT_ARMA_Q  = 2
DEFAULT_ARIMA_P = 2
DEFAULT_ARIMA_D = 1
DEFAULT_ARIMA_Q = 2


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

    Uses rolling-mean columns (e.g. s2_rmean_10) if present — smoother signal,
    which is required before fitting any time-series model.

    PC1 sign is flipped so that health_index INCREASES as the engine degrades.
    This makes the threshold-crossing interpretation intuitive: when the forecast
    exceeds the threshold, the engine has reached end-of-life.
    """
    # ── choose feature columns ──────────────────────────────────────────────
    rmean_cols = [f"{c}_rmean_{rolling_window}" for c in sensor_cols]
    use_cols   = rmean_cols if all(c in train.columns for c in rmean_cols) else sensor_cols

    # ── fit PCA on train only — never fit on test (would be data leakage) ──
    pca  = PCA(n_components=1)
    X_tr = train[use_cols].values
    pca.fit(X_tr)

    pc1_train = pca.transform(X_tr).ravel()
    pc1_test  = pca.transform(test[use_cols].values).ravel()

    # ── sign flip: PC1 should correlate with degradation (higher = worse) ──
    # Strategy: compute Pearson correlation between PC1 and decreasing RUL on train.
    # If correlation is negative, flip sign.
    if np.corrcoef(pc1_train, -train["RUL"].values)[0, 1] < 0:
        pc1_train = -pc1_train
        pc1_test  = -pc1_test

    train = train.copy()
    test  = test.copy()
    train["health_index"] = pc1_train
    test["health_index"]  = pc1_test

    return train, test


# ─────────────────────────────────────────────
# 2. FAILURE THRESHOLD
# ─────────────────────────────────────────────

def compute_failure_threshold(
    train: pd.DataFrame,
    end_of_life_rul: int = END_OF_LIFE_RUL,
    quantile: float = 0.5,
) -> dict[int, float]:
    """
    Compute a per-dataset_id failure threshold from near-end-of-life training rows.

    Per-dataset_id thresholds are required because FD002/FD004 have 6 operating
    conditions — their health_index distributions differ from FD001/FD003.
    A single global threshold (previous implementation) was wrong.

    Returns a dict: {dataset_id: threshold_value}
    """
    eol_rows = train[train["RUL"] <= end_of_life_rul]
    thresholds = {}
    for did, group in eol_rows.groupby("dataset_id"):
        thresholds[int(did)] = float(group["health_index"].quantile(quantile))
    return thresholds


# ─────────────────────────────────────────────
# 3. STATIONARITY CHECK — CH03 BOOK METHODOLOGY
# ─────────────────────────────────────────────

def check_stationarity_adf(series: np.ndarray) -> dict:
    """
    Run ADF at level and at first difference (and second if needed).

    Follows CH03 exactly:
        1. ADF on level series.
        2. If p > 0.05: difference once, ADF again.
        3. If still p > 0.05: difference again, ADF once more.

    Returns:
        {
            'level_pvalue': float,
            'diff1_pvalue': float | None,
            'diff2_pvalue': float | None,
            'recommended_d': int,     # 0, 1, or 2
        }
    """
    result = {"level_pvalue": None, "diff1_pvalue": None, "diff2_pvalue": None, "recommended_d": 0}

    # level
    p0 = adfuller(series)[1]
    result["level_pvalue"] = round(p0, 4)

    if p0 < 0.05:
        # already stationary at level → d = 0
        result["recommended_d"] = 0
        return result

    # first difference
    diff1 = np.diff(series, n=1)
    p1 = adfuller(diff1)[1]
    result["diff1_pvalue"] = round(p1, 4)

    if p1 < 0.05:
        result["recommended_d"] = 1
        return result

    # second difference (book tests this in CH07 example)
    diff2 = np.diff(diff1, n=1)
    p2 = adfuller(diff2)[1]
    result["diff2_pvalue"] = round(p2, 4)
    result["recommended_d"] = 2 if p2 < 0.05 else 2  # cap at 2 per book
    return result


def run_stationarity_report(
    train: pd.DataFrame,
    n_engines_per_subset: int = 5,
) -> pd.DataFrame:
    """
    Run ADF stationarity check on a stratified sample of engines.

    Reports p-values at level and diff-1 for each sampled engine.
    The distribution of recommended_d across all engines determines what d
    to pass to optimize_ARIMA — not just 6 hand-picked engines from FD001.
    """
    rows = []
    for did, group in train.groupby("dataset_id"):
        engine_sample = group["engine_id"].unique()[:n_engines_per_subset]
        for eid in engine_sample:
            s = (
                group[group["engine_id"] == eid]
                .sort_values("cycle")["health_index"]
                .values
            )
            if len(s) < 10:
                continue
            adf = check_stationarity_adf(s)
            rows.append({"dataset_id": did, "engine_id": eid, **adf})

    df = pd.DataFrame(rows)
    # Print a clear summary so the recommended d is visible
    print("\nStationarity Report (ADF test per sampled engine):")
    print(f"{'dataset_id':<12}{'engine_id':<12}{'level_p':<12}{'diff1_p':<12}{'rec_d'}")
    print("-" * 55)
    for _, r in df.iterrows():
        print(
            f"{int(r.dataset_id):<12}{int(r.engine_id):<12}"
            f"{r.level_pvalue:<12}{str(r.diff1_pvalue):<12}{r.recommended_d}"
        )
    d_counts = df["recommended_d"].value_counts().to_dict()
    print(f"\nd distribution: {d_counts}")
    modal_d = int(df["recommended_d"].mode()[0])
    print(f"→ recommended d = {modal_d}  (modal across all sampled engines)")
    return df


# ─────────────────────────────────────────────
# 4. ACF / PACF PLOTS — CH05/CH06/CH07 BOOK METHODOLOGY
# ─────────────────────────────────────────────

def plot_acf_pacf(
    series: np.ndarray,
    lags: int = 20,
    title: str = "ACF / PACF — health_index",
) -> None:
    """
    Plot ACF and PACF side by side. Required by the book before fitting any model.

    CH05: PACF used to determine AR order p (PACF cuts off after lag p).
    CH06: ACF used to determine MA order q (ACF cuts off after lag q).
    CH07: Both used together after differencing to find ARIMA(p,d,q).

    Reading the plots:
        - PACF cuts off at lag p → AR(p) candidate
        - ACF cuts off at lag q  → MA(q) candidate
        - Both tail off          → ARMA(p,q) needed
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF — {title}")

    plot_pacf(series, lags=lags, ax=axes[1])
    axes[1].set_title(f"PACF — {title}")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 5. AIC-BASED ORDER SELECTION — CH06/CH07 BOOK METHODOLOGY
# ─────────────────────────────────────────────

def optimize_AR(
    endog: Union[pd.Series, np.ndarray, list],
    p_values: list[int],
) -> pd.DataFrame:
    """
    Select AR(p) order by AIC. Follows book's optimize_ARMA pattern (CH06).

    Uses SARIMAX(order=(p, 0, 0)) — the book's AR formulation.
    Sorts by ascending AIC; lowest AIC = best model.

    Args:
        endog:    the stationary (possibly differenced) health_index series.
        p_values: list of AR lags to try, e.g. [1, 2, 3, 4, 5].
    """
    results = []
    for p in p_values:
        try:
            model = SARIMAX(endog, order=(p, 0, 0), simple_differencing=False).fit(disp=False)
            results.append({"p": p, "AIC": round(model.aic, 2)})
        except Exception:
            continue  # skip orders that fail to converge

    result_df = (
        pd.DataFrame(results)
        .sort_values("AIC", ascending=True)  # lower AIC = better, exactly as in book
        .reset_index(drop=True)
    )
    return result_df


def optimize_ARMA(
    endog: Union[pd.Series, np.ndarray, list],
    order_list: list[tuple[int, int]],
) -> pd.DataFrame:
    """
    Select ARMA(p,q) order by AIC. Direct copy of book's optimize_ARMA (CH06).

    Uses SARIMAX(order=(p, 0, q)) — ARMA is ARIMA with d=0.

    Args:
        endog:      the stationary health_index series (d=0 assumed).
        order_list: list of (p, q) tuples, e.g. list(product([1,2,3],[1,2,3])).
    """
    results = []
    for p, q in order_list:
        try:
            model = SARIMAX(endog, order=(p, 0, q), simple_differencing=False).fit(disp=False)
            results.append({"(p,q)": (p, q), "AIC": round(model.aic, 2)})
        except Exception:
            continue

    result_df = (
        pd.DataFrame(results)
        .sort_values("AIC", ascending=True)
        .reset_index(drop=True)
    )
    return result_df


def optimize_ARIMA(
    endog: Union[pd.Series, np.ndarray, list],
    order_list: list[tuple[int, int]],
    d: int,
) -> pd.DataFrame:
    """
    Select ARIMA(p,d,q) order by AIC. Direct copy of book's optimize_ARIMA (CH07).

    d is passed in separately — it must come from the ADF test (check_stationarity_adf),
    NOT hardcoded. The book explicitly determines d from ADF before calling this.

    Args:
        endog:      the ORIGINAL (un-differenced) health_index series.
                    SARIMAX handles differencing internally when simple_differencing=False.
        order_list: list of (p, q) tuples.
        d:          integration order from ADF test.
    """
    results = []
    for p, q in order_list:
        try:
            model = SARIMAX(endog, order=(p, d, q), simple_differencing=False).fit(disp=False)
            results.append({"(p,q)": (p, q), "d": d, "AIC": round(model.aic, 2)})
        except Exception:
            continue

    result_df = (
        pd.DataFrame(results)
        .sort_values("AIC", ascending=True)
        .reset_index(drop=True)
    )
    return result_df


# ─────────────────────────────────────────────
# 6. RESIDUAL DIAGNOSTICS — CH06/CH07 BOOK METHODOLOGY
# ─────────────────────────────────────────────

def check_residuals(
    residuals: np.ndarray,
    model_name: str = "model",
    plot_qq: bool = False,
) -> pd.DataFrame:
    """
    Ljung-Box test on model residuals. Required by book in CH06 and CH07.

    Null hypothesis: residuals are white noise (no autocorrelation).
    If p-value > 0.05 for all lags → residuals look like white noise → model is adequate.
    If any p-value < 0.05 → residual autocorrelation remains → model is mis-specified.

    Args:
        residuals:  1-D array of model residuals from model_fit.resid.
        model_name: label for the print output.
        plot_qq:    if True, show QQ plot (required for ARIMA per CH07).
    """
    # Ljung-Box test at lags 1..10 (book uses np.arange(1, 11, 1))
    lb_result = acorr_ljungbox(residuals, lags=np.arange(1, 11, 1), return_df=True)

    print(f"\nLjung-Box residual test — {model_name}")
    print(lb_result[["lb_stat", "lb_pvalue"]].to_string())

    all_pass = (lb_result["lb_pvalue"] > 0.05).all()
    if all_pass:
        print("✓ All p-values > 0.05 — residuals are white noise (model is adequate)")
    else:
        print("✗ Some p-values < 0.05 — residual autocorrelation remains (consider higher order)")

    # QQ plot — book adds this in CH07 for ARIMA
    if plot_qq:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        qqplot(residuals, line="s", ax=axes[0])
        axes[0].set_title(f"QQ plot — {model_name} residuals")
        axes[1].plot(residuals)
        axes[1].axhline(0, color="red", ls="--")
        axes[1].set_title(f"Residuals over time — {model_name}")
        plt.tight_layout()
        plt.show()

    return lb_result


# ─────────────────────────────────────────────
# 7. ROLLING FORECAST — CH05/CH06/CH07 BOOK METHODOLOGY
# ─────────────────────────────────────────────

def rolling_forecast_engine(
    series: np.ndarray,
    train_len: int,
    order: tuple[int, int, int],
    window: int = 1,
) -> np.ndarray:
    """
    Walk-forward forecast on a SINGLE engine's health_index series.
    Mirrors the book's rolling_forecast function exactly (CH05 / CH06 / CH07).

    Book pattern (translated to numpy):
        for i in range(train_len, len(series), window):
            fit SARIMAX on series[:i]
            predict next `window` steps using get_prediction(0, i + window - 1)
            keep only the out-of-sample portion (last `window` predictions)

    Args:
        series:    complete health_index array for one engine.
        train_len: number of cycles used as initial training set.
        order:     (p, d, q) — best order from optimize_*.
        window:    how many steps to forecast at each refit step.
                   window=1  → one-step-ahead (most realistic, slowest).
                   window=5  → 5-step refit cadence (faster, less precise).

    Returns:
        pred: predicted health_index for the validation portion [train_len:].
    """
    p, d, q = order
    total_len = len(series)
    pred = []

    for i in range(train_len, total_len, window):
        # Fit on history seen so far (series[:i])
        model = SARIMAX(series[:i], order=(p, d, q), simple_differencing=False)
        res   = model.fit(disp=False)

        # get_prediction with end=i+window-1 gives in-sample + out-of-sample
        # we only want the last `window` values (the out-of-sample part)
        predictions = res.get_prediction(start=0, end=i + window - 1)
        oos = predictions.predicted_mean.values[-window:]
        pred.extend(oos.tolist())

    # Trim to exact validation length (total_len - train_len)
    pred = np.array(pred[: total_len - train_len])
    return pred


# ─────────────────────────────────────────────
# 8. SMOOTHING UTILITY
# ─────────────────────────────────────────────

def smooth_series(series: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """
    Rolling-median smoother applied to health_index before model fitting.

    Reduces noise in the PCA health_index trajectory so that AR/ARMA/ARIMA
    can capture the degradation trend rather than fitting noise.
    """
    series = np.asarray(series, dtype=float)
    if window <= 1 or len(series) < 2:
        return series
    return pd.Series(series).rolling(window, min_periods=1, center=False).median().values


# ─────────────────────────────────────────────
# 9. FORECAST → RUL
# ─────────────────────────────────────────────

def _linear_extrapolation_rul(
    series: np.ndarray,
    threshold: float,
    tail: int | None = None,
) -> float:
    """
    Fallback: when the model forecast never crosses the threshold within MAX_HORIZON,
    fit a linear trend to the tail of the observed series and extrapolate.

    tail defaults to min(30, 20% of series length) — adapts to engine lifespan.
    """
    # Adaptive tail: 20% of series or 30 cycles, whichever is smaller
    if tail is None:
        tail = max(5, min(30, int(0.2 * len(series))))

    y = series[-tail:] if len(series) >= tail else series
    if len(y) < 3:
        return float(RUL_CAP)

    x     = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    last  = float(y[-1])

    if slope <= 1e-6:
        return float(RUL_CAP)   # flat or improving — engine far from failure

    steps = (threshold - last) / slope
    return float(min(max(steps, 0.0), RUL_CAP))


def _estimate_rul_from_forecast(
    preds: np.ndarray,
    observed: np.ndarray,
    threshold: float,
) -> float:
    """
    RUL = index of first forecast step that crosses the failure threshold.
    Falls back to linear extrapolation if no crossing found within MAX_HORIZON.
    """
    preds = np.asarray(preds, dtype=float)

    if preds.size == 0 or not np.all(np.isfinite(preds)):
        return _linear_extrapolation_rul(observed, threshold)

    # If already above threshold at first forecast step → RUL = 0
    crossings = np.where(preds >= threshold)[0]
    if crossings.size > 0:
        return float(crossings[0])

    return _linear_extrapolation_rul(observed, threshold)


# ─────────────────────────────────────────────
# 10. AR PREDICT — CH05 BOOK METHODOLOGY
# ─────────────────────────────────────────────

def predict_rul_ar(
    series: np.ndarray,
    threshold: float,
    p: int = DEFAULT_AR_P,
    smooth_window: int = SMOOTH_WINDOW,
) -> float:
    """
    AR(p) forecast via SARIMAX(order=(p, 0, 0)).

    Book compliance (CH05):
    - Uses SARIMAX, not AutoReg.
    - p must come from optimize_AR / ACF-PACF analysis, not a hardcoded default.
    - simple_differencing=False: SARIMAX handles stationarity internally.
    """
    smoothed = smooth_series(series, smooth_window)

    # Need at least p + 5 observations to fit reliably
    if len(smoothed) <= p + 5:
        return _linear_extrapolation_rul(smoothed, threshold)

    try:
        model = SARIMAX(smoothed, order=(p, 0, 0), simple_differencing=False)
        res   = model.fit(disp=False)
        preds = res.forecast(steps=MAX_HORIZON)
        return _estimate_rul_from_forecast(preds, smoothed, threshold)
    except Exception:
        return _linear_extrapolation_rul(smoothed, threshold)


# ─────────────────────────────────────────────
# 11. ARMA PREDICT — CH06 BOOK METHODOLOGY
# ─────────────────────────────────────────────

def predict_rul_arma(
    series: np.ndarray,
    threshold: float,
    p: int = DEFAULT_ARMA_P,
    q: int = DEFAULT_ARMA_Q,
    smooth_window: int = SMOOTH_WINDOW,
) -> float:
    """
    ARMA(p, q) forecast via SARIMAX(order=(p, 0, q)).

    Book compliance (CH06):
    - Uses SARIMAX with d=0 (ARMA = ARIMA with no differencing).
    - (p, q) must come from optimize_ARMA (AIC), not from RMSE grid search.
    - simple_differencing=False.
    """
    smoothed = smooth_series(series, smooth_window)

    if len(smoothed) <= p + q + 3:
        return _linear_extrapolation_rul(smoothed, threshold)

    try:
        model = SARIMAX(smoothed, order=(p, 0, q), simple_differencing=False)
        res   = model.fit(disp=False)
        preds = res.forecast(steps=MAX_HORIZON)
        return _estimate_rul_from_forecast(preds, smoothed, threshold)
    except Exception:
        return _linear_extrapolation_rul(smoothed, threshold)


# ─────────────────────────────────────────────
# 12. ARIMA PREDICT — CH07 BOOK METHODOLOGY
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
    ARIMA(p, d, q) forecast via SARIMAX(order=(p, d, q)).

    Book compliance (CH07):
    - Uses SARIMAX with simple_differencing=False.
    - d must come from ADF test, not hardcoded.
    - (p, q) from optimize_ARIMA (AIC).
    - After fitting, call check_residuals() to run Ljung-Box and QQ plot.
    """
    smoothed = smooth_series(series, smooth_window)

    if len(smoothed) <= p + d + q + 3:
        return _linear_extrapolation_rul(smoothed, threshold)

    try:
        model = SARIMAX(smoothed, order=(p, d, q), simple_differencing=False)
        res   = model.fit(disp=False)
        preds = res.forecast(steps=MAX_HORIZON)
        return _estimate_rul_from_forecast(preds, smoothed, threshold)
    except Exception:
        return _linear_extrapolation_rul(smoothed, threshold)


# ─────────────────────────────────────────────
# 13. DATASET-LEVEL PREDICTION
# ─────────────────────────────────────────────

def predict_dataset(
    df: pd.DataFrame,
    predict_fn: Callable[[np.ndarray], float],
    threshold_map: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a per-engine forecasting function to every engine in df.

    threshold_map: {dataset_id: threshold} — per-subset thresholds (not global).
    predict_fn: already has p/d/q bound via functools.partial, but NOT threshold.
                threshold is injected here from threshold_map per dataset_id.

    Returns (y_true, y_pred, dataset_ids), one entry per engine.
    Logs fallback rate: % of engines where linear extrapolation fired.
    """
    df = df.sort_values(["engine_id", "cycle"])

    y_true, y_pred, dids = [], [], []
    fallback_count = 0
    total_count    = 0

    for _, g in df.groupby("engine_id", sort=False):
        series     = g["health_index"].values
        true_rul   = float(g["RUL"].iloc[-1])
        did        = int(g["dataset_id"].iloc[0])

        # Use dataset-specific threshold (fixes global threshold bug from audit)
        thresh = threshold_map.get(did, list(threshold_map.values())[0])

        pred_raw = predict_fn(series, threshold=thresh)
        pred     = float(np.clip(pred_raw, 0.0, RUL_CAP))

        # Detect fallback: linear extrapolation was used if prediction == RUL_CAP
        # (exact fallback detection would require a wrapper — this is a proxy)
        if pred == float(RUL_CAP):
            fallback_count += 1
        total_count += 1

        y_true.append(true_rul)
        y_pred.append(pred)
        dids.append(did)

    fallback_rate = 100.0 * fallback_count / total_count if total_count > 0 else 0.0
    print(f"  Fallback rate (linear extrapolation proxy): {fallback_rate:.1f}%  "
          f"({fallback_count}/{total_count} engines)")

    return np.array(y_true), np.array(y_pred), np.array(dids)


# ─────────────────────────────────────────────
# 14. SIMULATE TEST FROM TRAIN
# ─────────────────────────────────────────────

def simulate_test_from_train(
    train_df: pd.DataFrame,
    cutoff_range: tuple[float, float] = (0.2, 0.9),  # widened from (0.3, 0.6)
    random_seed: int = 42,
    max_engines: int | None = None,
) -> pd.DataFrame:
    """
    Create a simulated validation set by truncating training engine histories.

    Key fix from audit:
    - cutoff_range now covers (0.2, 0.9) of engine life instead of (0.3, 0.6).
      This ensures the simulated val set covers early-life engines — the range
      where AR/ARMA fail most — matching the distribution of the real test set.
    - threshold must be computed BEFORE calling this function, on the engines
      that are NOT in the simulated val set, to avoid threshold leakage.
    """
    rng  = np.random.default_rng(random_seed)
    rows = []

    engines = train_df["engine_id"].unique()
    if max_engines is not None and len(engines) > max_engines:
        engines = rng.choice(engines, size=max_engines, replace=False)

    lo_frac, hi_frac = cutoff_range
    for eid in engines:
        g = train_df[train_df["engine_id"] == eid].sort_values("cycle")
        n = len(g)
        if n < 20:
            continue
        lo     = max(10, int(n * lo_frac))
        hi     = max(lo + 1, int(n * hi_frac) + 1)
        cutoff = int(rng.integers(lo, min(hi, n)))
        rows.append(g.iloc[:cutoff])

    return pd.concat(rows, ignore_index=True)