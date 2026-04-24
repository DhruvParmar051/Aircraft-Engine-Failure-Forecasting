"""
classical.py — AR / ARMA / ARIMA RUL prediction via health-index forecasting.

Methodology follows "Time Series Forecasting in Python" by Marco Peixeiro:
    CH03 → stationarity (ADF test, differencing)
    CH05 → AR model (SARIMAX, ACF/PACF, rolling forecast)
    CH06 → ARMA model (SARIMAX, optimize_ARMA via AIC, Ljung-Box)
    CH07 → ARIMA model (SARIMAX, optimize_ARIMA via AIC, Ljung-Box + QQ plot)

Book rules enforced here:
    1. ALL models use SARIMAX — never AutoReg or statsmodels.ARIMA directly.
    2. Order selection uses AIC via optimize_AR / optimize_ARMA / optimize_ARIMA.
    3. Ljung-Box test on residuals after every fit (CH06 + CH07).
    4. QQ plot for residual normality check (CH07 for ARIMA).
    5. rolling_forecast for walk-forward validation (CH05/CH06/CH07).
    6. ADF run at level + first difference (+ second if needed) to determine d.

Single-dataset design:
    - One train DataFrame, one test DataFrame, multiple engines each.
    - No dataset_id column anywhere — single threshold (float) not a dict.
    - select_best_* samples N engines from the single dataset.
"""

from __future__ import annotations

import warnings
from collections import Counter
from itertools import product
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.isotonic import IsotonicRegression


warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

RUL_CAP         = 125
MAX_HORIZON     = 400
SMOOTH_WINDOW   = 5
END_OF_LIFE_RUL = 5

DEFAULT_AR_P    = 3
DEFAULT_ARMA_P  = 2
DEFAULT_ARMA_Q  = 2
DEFAULT_ARIMA_P = 2
DEFAULT_ARIMA_D = 2   # ADF shows d=2 for CMAPSS health_index
DEFAULT_ARIMA_Q = 2
_LAST_WAS_FALLBACK: bool = False
SAFETY_FACTOR = 0.88

# ─────────────────────────────────────────────
# 1. HEALTH INDEX — PCA on rolling-mean sensors
# ─────────────────────────────────────────────

def _combine_components(pca, X, signs, n_comp):
    pc = pca.transform(X)
    result = [pc[:, i] * signs[i] for i in range(n_comp)]
    return result[0] if n_comp == 1 else np.maximum(result[0], result[1])



def _enforce_monotone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-engine isotonic regression: forces health_index to be
    monotonically non-decreasing over cycle.
    WHY: PCA projection can oscillate due to op-condition noise.
    Isotonic regression is the minimal correction — it doesn't
    change the mean, only removes non-monotone bumps.
    """
    df = df.copy()
    ir = IsotonicRegression(increasing=True)  # higher cycle = more degraded
    for eid, grp in df.groupby("engine_id"):
        idx     = grp.index
        cycles  = grp["cycle"].values.astype(float)
        hi      = grp["health_index"].values
        df.loc[idx, "health_index"] = ir.fit_transform(cycles, hi)
    return df

def build_pca_health_index(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sensor_cols: list[str],
    rolling_window: int = 10,
    n_components: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    PCA health_index with operating condition removal.

    Key insight for multi-condition datasets (FD002/FD004):
    Per-cluster PCA produces incompatible scales across clusters.
    Correct approach:
        1. Subtract per-cluster mean from each sensor (removes op condition effect)
        2. Run global PCA on residuals (captures degradation, not op switching)
        3. Sign-flip so higher = more degraded
        4. Standardize using train stats
    """
    from sklearn.metrics import r2_score as _r2

    rmean_cols = [f"{c}_rmean_{rolling_window}" for c in sensor_cols]
    use_cols   = rmean_cols if all(c in train.columns for c in rmean_cols) else sensor_cols

    train = train.copy()
    test  = test.copy()

    # ── Step 1: remove per-cluster mean (op condition detrending) ────────
    # Fit cluster means on train only — no leakage
    cluster_means = (
        train.groupby("op_cluster")[use_cols].mean()
    )  # shape: (n_clusters, n_sensors)

    def subtract_cluster_mean(df, means):
        df = df.copy()
        for cluster_id, row in means.iterrows():
            mask = df["op_cluster"] == cluster_id
            df.loc[mask, use_cols] = df.loc[mask, use_cols].values - row.values
        return df

    train_detrended = subtract_cluster_mean(train, cluster_means)
    test_detrended  = subtract_cluster_mean(test,  cluster_means)

    # ── Step 2: global PCA on detrended sensors ───────────────────────────
    pca   = PCA(n_components=n_components).fit(train_detrended[use_cols].values)
    pc_tr = pca.transform(train_detrended[use_cols].values)

    # ── Step 3: sign-flip so higher = more degraded (corr with -RUL) ─────
    signs = []
    for i in range(n_components):
        c    = pc_tr[:, i]
        sign = 1.0 if np.corrcoef(c, -train["RUL"].values)[0, 1] >= 0 else -1.0
        signs.append(sign)

    train["health_index"] = _combine_components(
        pca, train_detrended[use_cols].values, signs, n_components
    )
    test["health_index"] = _combine_components(
        pca, test_detrended[use_cols].values, signs, n_components
    )

    # ── Step 4: standardize using train stats ────────────────────────────
    mu = train["health_index"].mean()
    sd = train["health_index"].std()
    if sd > 1e-6:
        train["health_index"] = (train["health_index"] - mu) / sd
        test["health_index"]  = (test["health_index"]  - mu) / sd

    train = _enforce_monotone(train)
    test  = _enforce_monotone(test)

    # Re-check R2 after monotone fix
    r2_rul = _r2(-train["RUL"].values, train["health_index"].values)
    print(f"health_index R2 with RUL (post-monotone): {r2_rul:.3f}  (target: > 0.3)")
    return train, test



# ─────────────────────────────────────────────
# 2. FAILURE THRESHOLD
# ─────────────────────────────────────────────

def compute_failure_threshold(
    train: pd.DataFrame,
    end_of_life_rul: int = END_OF_LIFE_RUL,
    quantile: float = 0.5,
) -> float:
    """
    Single failure threshold from near-end-of-life training rows.
    Returns a scalar float — not a dict.
    """
    eol_rows = train[train["RUL"] <= end_of_life_rul]
    return float(eol_rows["health_index"].quantile(quantile))


# ─────────────────────────────────────────────
# 3. STATIONARITY CHECK — CH03
# ─────────────────────────────────────────────

def check_stationarity_adf(series: np.ndarray) -> dict:
    """
    ADF at level → diff-1 → diff-2. Returns recommended d.

    CH03 rule:
        p > 0.05 at level  → difference once
        p > 0.05 at diff-1 → difference again (cap at d=2)
    """
    result = {
        "level_pvalue": None,
        "diff1_pvalue": None,
        "diff2_pvalue": None,
        "recommended_d": 0,
    }

    p0 = adfuller(series)[1]
    result["level_pvalue"] = round(p0, 4)
    if p0 < 0.05:
        result["recommended_d"] = 0
        return result

    diff1 = np.diff(series, n=1)
    p1    = adfuller(diff1)[1]
    result["diff1_pvalue"] = round(p1, 4)
    if p1 < 0.05:
        result["recommended_d"] = 1
        return result

    diff2 = np.diff(diff1, n=1)
    p2    = adfuller(diff2)[1]
    result["diff2_pvalue"]  = round(p2, 4)
    result["recommended_d"] = 2
    return result


def run_stationarity_report(
    train: pd.DataFrame,
    n_engines: int = 240,
) -> pd.DataFrame:
    """
    ADF report on a sample of engines from the single dataset.
    Prints per-engine p-values and the modal recommended_d.

    Args:
        n_engines: number of engines to sample (default 10).
    """
    engine_sample = train["engine_id"].unique()[:n_engines]
    rows = []

    for eid in engine_sample:
        s = (
            train[train["engine_id"] == eid]
            .sort_values("cycle")["health_index"]
            .values
        )
        if len(s) < 10:
            continue
        adf = check_stationarity_adf(s)
        rows.append({"engine_id": eid, **adf})

    df = pd.DataFrame(rows)

    print("\nStationarity Report (ADF test per sampled engine):")
    print(f"{'engine_id':<12}{'level_p':<12}{'diff1_p':<12}{'rec_d'}")
    print("-" * 44)
    for _, r in df.iterrows():
        print(
            f"{int(r.engine_id):<12}"
            f"{r.level_pvalue:<12}{str(r.diff1_pvalue):<12}{r.recommended_d}"
        )

    d_counts = df["recommended_d"].value_counts().to_dict()
    modal_d  = int(df["recommended_d"].mode()[0])
    print(f"\nd distribution: {d_counts}")
    print(f"→ recommended d = {modal_d}  (modal across {len(df)} sampled engines)")
    return df


# ─────────────────────────────────────────────
# 4. ACF / PACF — CH05/CH06/CH07
# ─────────────────────────────────────────────

def plot_acf_pacf(
    series: np.ndarray,
    lags: int = 20,
    title: str = "ACF / PACF — health_index",
) -> None:
    """
    ACF + PACF side by side.
    PACF cuts off at p → AR(p). ACF cuts off at q → MA(q). Both tail off → ARMA.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series,  lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF — {title}")
    plot_pacf(series, lags=lags, ax=axes[1])
    axes[1].set_title(f"PACF — {title}")
    plt.tight_layout()
    plt.show()


def plot_acf_pacf_multi(
    train: pd.DataFrame,
    d: int,
    n_engines: int = 3,
    lags: int = 20,
) -> None:
    """
    ACF/PACF grid for N sampled engines from the single dataset.
    Series is differenced d times before plotting (CH07 rule).

    Args:
        train:     training DataFrame with health_index column.
        d:         differencing order from run_stationarity_report.
        n_engines: number of engines to plot.
        lags:      number of lags for ACF/PACF.
    """
    eids      = train["engine_id"].unique()[:n_engines]
    fig, axes = plt.subplots(len(eids), 2, figsize=(12, 4 * len(eids)))
    axes      = np.atleast_2d(axes)

    for row, eid in enumerate(eids):
        raw         = train[train["engine_id"] == eid].sort_values("cycle")["health_index"].values
        smth        = smooth_series(raw)
        series_plot = np.diff(smth, n=d) if d > 0 else smth

        plot_acf(series_plot,  lags=lags, ax=axes[row][0])
        axes[row][0].set_title(f"ACF — engine {eid} (diff-{d})")
        plot_pacf(series_plot, lags=lags, ax=axes[row][1])
        axes[row][1].set_title(f"PACF — engine {eid} (diff-{d})")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 5. AIC-BASED ORDER SELECTION — CH06/CH07
# ─────────────────────────────────────────────

def optimize_AR(
    endog: Union[pd.Series, np.ndarray, list],
    p_values: list[int],
) -> pd.DataFrame:
    """Select AR(p) by AIC on a single stationary series."""
    results = []
    for p in p_values:
        try:
            model = SARIMAX(endog, order=(p, 0, 0), simple_differencing=False).fit(disp=False)
            results.append({"p": p, "AIC": round(model.aic, 2)})
        except Exception:
            continue
    return pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)


def optimize_ARMA(
    endog: Union[pd.Series, np.ndarray, list],
    order_list: list[tuple[int, int]],
) -> pd.DataFrame:
    """Select ARMA(p,q) by AIC on a single stationary series."""
    results = []
    for p, q in order_list:
        try:
            model = SARIMAX(endog, order=(p, 0, q), simple_differencing=False).fit(disp=False)
            results.append({"(p,q)": (p, q), "AIC": round(model.aic, 2)})
        except Exception:
            continue
    return pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)


def optimize_ARIMA(
    endog: Union[pd.Series, np.ndarray, list],
    order_list: list[tuple[int, int]],
    d: int,
) -> pd.DataFrame:
    """Select ARIMA(p,d,q) by AIC on the original (un-differenced) series."""
    results = []
    for p, q in order_list:
        try:
            model = SARIMAX(endog, order=(p, d, q), simple_differencing=False).fit(disp=False)
            results.append({"(p,q)": (p, q), "d": d, "AIC": round(model.aic, 2)})
        except Exception:
            continue
    return pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)


# ─────────────────────────────────────────────
# 5b. MULTI-ENGINE ORDER SELECTION
# ─────────────────────────────────────────────

def _get_representative_engine(train: pd.DataFrame) -> tuple[int, np.ndarray]:
    """
    Return (eid, smoothed_series) for the longest engine.
    Used for Ljung-Box diagnostic fit in notebooks.
    """
    eid = train.groupby("engine_id")["cycle"].count().idxmax()
    raw = train[train["engine_id"] == eid].sort_values("cycle")["health_index"].values
    return int(eid), smooth_series(raw)


def select_best_ar_order(
    train: pd.DataFrame,
    d: int,
    p_values: list[int] | None = None,
    n_engines: int = 5,
) -> int:
    """
    Sample N engines, run optimize_AR on each diff-d series,
    return modal best p.

    Args:
        train:     training DataFrame.
        d:         differencing order from run_stationarity_report.
        p_values:  AR lag candidates. Defaults to [1..10].
        n_engines: engines to sample.
    Returns:
        best_p: modal best AR order.
    """
    if p_values is None:
        p_values = list(range(1, 11))

    eids    = train["engine_id"].unique()[:n_engines]
    best_ps = []

    for eid in eids:
        raw  = train[train["engine_id"] == eid].sort_values("cycle")["health_index"].values
        smth = smooth_series(raw)
        diff = np.diff(smth, n=d) if d > 0 else smth

        if len(diff) < max(p_values) + 5:
            continue

        aic_df = optimize_AR(diff, p_values)
        if aic_df.empty:
            continue

        best_p = int(aic_df.iloc[0]["p"])
        best_ps.append(best_p)
        print(f"  engine {eid}: best p={best_p}  (AIC={aic_df.iloc[0]['AIC']})")

    if not best_ps:
        print("  WARNING: no valid engines found, returning default p=3")
        return 3

    modal_p = Counter(best_ps).most_common(1)[0][0]
    print(f"\n→ Modal best AR order: p={modal_p}  "
          f"(from {len(best_ps)} engines, freq={Counter(best_ps).most_common(5)})")
    return modal_p


def select_best_arma_order(
    train: pd.DataFrame,
    d: int,
    p_range: range | None = None,
    q_range: range | None = None,
    n_engines: int = 5,
) -> tuple[int, int]:
    """
    Sample N engines, run optimize_ARMA on each diff-d series,
    return modal best (p,q).

    Args:
        train:    training DataFrame.
        d:        differencing order from run_stationarity_report.
        p_range:  range of p values. Defaults to range(1,6).
        q_range:  range of q values. Defaults to range(1,6).
        n_engines: engines to sample.
    Returns:
        (best_p, best_q): modal best ARMA order.
    """
    if p_range is None:
        p_range = range(1, 4)
    if q_range is None:
        q_range = range(1, 4)

    order_list = list(product(p_range, q_range))
    eids       = train["engine_id"].unique()[:n_engines]
    best_pqs   = []

    for eid in eids:
        raw  = train[train["engine_id"] == eid].sort_values("cycle")["health_index"].values
        smth = smooth_series(raw)
        diff = np.diff(smth, n=d) if d > 0 else smth

        if len(diff) < max(p_range) + max(q_range) + 5:
            continue

        aic_df = optimize_ARMA(diff, order_list)
        if aic_df.empty:
            continue

        best_pq = aic_df.iloc[0]["(p,q)"]
        best_pqs.append(tuple(best_pq))
        print(f"  engine {eid}: best (p,q)={best_pq}  (AIC={aic_df.iloc[0]['AIC']})")

    if not best_pqs:
        print("  WARNING: no valid engines found, returning default (2,2)")
        return (2, 2)

    modal_pq       = Counter(best_pqs).most_common(1)[0][0]
    best_p, best_q = int(modal_pq[0]), int(modal_pq[1])
    print(f"\n→ Modal best ARMA order: ({best_p},{best_q})  "
          f"(from {len(best_pqs)} engines, freq={Counter(best_pqs).most_common(5)})")
    return best_p, best_q


def select_best_arima_order(
    train: pd.DataFrame,
    d: int,
    p_range: range | None = None,
    q_range: range | None = None,
    n_engines: int = 5,
) -> tuple[int, int]:
    """
    Sample N engines, run optimize_ARIMA on each original series,
    return modal best (p,q).

    Passes original (un-differenced) series — SARIMAX handles d internally.

    Args:
        train:    training DataFrame.
        d:        differencing order from run_stationarity_report.
        p_range:  range of p values. Defaults to range(1,6).
        q_range:  range of q values. Defaults to range(1,6).
        n_engines: engines to sample.
    Returns:
        (best_p, best_q): modal best (p,q) for ARIMA(p,d,q).
    """
    if p_range is None:
        p_range = range(1, 4)
    if q_range is None:
        q_range = range(1, 4)

    order_list = list(product(p_range, q_range))
    eids       = train["engine_id"].unique()[:n_engines]
    best_pqs   = []

    for eid in eids:
        raw  = train[train["engine_id"] == eid].sort_values("cycle")["health_index"].values
        smth = smooth_series(raw)

        if len(smth) < max(p_range) + d + max(q_range) + 5:
            continue

        aic_df = optimize_ARIMA(smth, order_list, d=d)
        if aic_df.empty:
            continue

        best_pq = aic_df.iloc[0]["(p,q)"]
        best_pqs.append(tuple(best_pq))
        print(f"  engine {eid}: best (p,q)={best_pq}  (AIC={aic_df.iloc[0]['AIC']})")

    if not best_pqs:
        print("  WARNING: no valid engines found, returning default (2,2)")
        return (2, 2)

    modal_pq       = Counter(best_pqs).most_common(1)[0][0]
    best_p, best_q = int(modal_pq[0]), int(modal_pq[1])
    print(f"\n→ Modal best ARIMA order: ({best_p},{d},{best_q})  "
          f"(from {len(best_pqs)} engines, freq={Counter(best_pqs).most_common(5)})")
    return best_p, best_q


# ─────────────────────────────────────────────
# 6. RESIDUAL DIAGNOSTICS — CH06/CH07
# ─────────────────────────────────────────────

def check_residuals(
    residuals: np.ndarray,
    model_name: str = "model",
    plot_qq: bool = False,
) -> pd.DataFrame:
    """
    Ljung-Box on residuals (CH06/CH07).
    p > 0.05 all lags → white noise → adequate model.
    plot_qq=True adds QQ plot (CH07 ARIMA requirement).
    """
    lb_result = acorr_ljungbox(residuals, lags=np.arange(1, 11, 1), return_df=True)

    print(f"\nLjung-Box residual test — {model_name}")
    print(lb_result[["lb_stat", "lb_pvalue"]].to_string())

    if (lb_result["lb_pvalue"] > 0.05).all():
        print("✓ All p-values > 0.05 — residuals are white noise (model is adequate)")
    else:
        print("✗ Some p-values < 0.05 — residual autocorrelation remains")

    if plot_qq:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        qqplot(residuals, line="s", ax=axes[0])
        axes[0].set_title(f"QQ plot — {model_name} residuals")
        axes[1].plot(residuals)
        axes[1].axhline(0, color="red", ls="--")
        axes[1].set_title(f"Residuals over time — {model_name}")
        plt.tight_layout()
        plt.show()

    return lb_result


# ─────────────────────────────────────────────
# 7. ROLLING FORECAST — CH05/CH06/CH07
# ─────────────────────────────────────────────

def rolling_forecast_engine(
    series: np.ndarray,
    train_len: int,
    order: tuple[int, int, int],
    window: int = 1,
) -> np.ndarray:
    """
    Walk-forward forecast on one engine's series (CH05/CH06/CH07).
    window=1 → one-step-ahead, most accurate.
    """
    p, d, q   = order
    total_len = len(series)
    pred      = []

    for i in range(train_len, total_len, window):
        try:
            model = SARIMAX(
                series[:i],
                order=(p, d, q),
                simple_differencing=False,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)

            predictions = res.get_prediction(start=0, end=i + window - 1)
            oos = np.asarray(predictions.predicted_mean)[-window:]

        except Exception:
            # 🔥 fallback: persistence model
            oos = np.repeat(series[i-1], window)

        pred.extend(oos.tolist())

    return np.array(pred[: total_len - train_len])


# ─────────────────────────────────────────────
# 8. SMOOTHING
# ─────────────────────────────────────────────

def smooth_series(series: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Rolling-median smoother applied before any model fit."""
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
    global _LAST_WAS_FALLBACK
    if tail is None:
        tail = max(5, min(30, int(0.2 * len(series))))
    y = series[-tail:] if len(series) >= tail else series
    if len(y) < 3:
        return _health_index_to_rul(float(series[-1]))

    x                = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    if slope <= 1e-4:
        # Flat tail — use regressor instead of capping at 125
        _LAST_WAS_FALLBACK = True   # flat tail → regressor used
        return _health_index_to_rul(float(series[-1]))

    steps = (threshold - float(y[-1])) / slope
    return float(min(max(steps, 0.0), RUL_CAP))

# Module-level variable — set once by fit_rul_from_health_index()
_RUL_REGRESSOR = None   # stores (slope, intercept) from train fit

# Instead fit only on recent history (last 30% of each engine's life)
def fit_rul_from_health_index(train: pd.DataFrame) -> None:
    global _RUL_REGRESSOR

    # Keep only rows from the last 30% of each engine's life
    # WHY: regressor is only used as fallback when forecast fails
    #      it should predict "given current state" not "given any state"
    recent_rows = []
    for eid, grp in train.groupby("engine_id"):
        g   = grp.sort_values("cycle")
        n   = len(g)
        cut = max(1, int(n * 0.4))   # last 60% of life
        recent_rows.append(g.iloc[cut:])
    
    recent = pd.concat(recent_rows)
    x      = recent["health_index"].values
    y      = recent["RUL"].values
    slope, intercept = np.polyfit(x, y, 1)
    _RUL_REGRESSOR   = (float(slope), float(intercept))
    r2 = 1 - np.sum((y - (slope*x + intercept))**2) / np.sum((y - y.mean())**2)
    print(f"  RUL regressor (recent 60%): RUL = {slope:.2f} * hi + {intercept:.2f}  (R2={r2:.3f})")

def _health_index_to_rul(health_index_val: float) -> float:
    if _RUL_REGRESSOR is None:
        return float(RUL_CAP)   # 125, then SAFETY_FACTOR=0.88 gives 110
    slope, intercept = _RUL_REGRESSOR
    pred = slope * health_index_val + intercept
    return float(np.clip(pred, 0.0, RUL_CAP))


def _estimate_rul_from_forecast(
    preds: np.ndarray,
    observed: np.ndarray,
    threshold: float = None,
) -> float:
    """
    1. Direct crossing within forecast horizon → return that step.
    2. No crossing → fit slope to forecast tail → extrapolate to threshold.
    3. Flat/negative forecast → fall back to observed series.
    """
    global _LAST_WAS_FALLBACK
    preds = np.asarray(preds, dtype=float)
    if preds.size == 0 or not np.all(np.isfinite(preds)):
        return _health_index_to_rul(float(observed[-1]))

    # Step 1: direct crossing within forecast
    crossings = np.where(preds >= threshold)[0]
    if crossings.size > 0:
        return float(max(crossings[0], 3))

    # Step 2: forecast slope extrapolation (last 50% of forecast)
    tail_start = max(1, len(preds) // 2)
    tail       = preds[tail_start:]
    x          = np.arange(len(tail), dtype=float)
    f_slope, _ = np.polyfit(x, tail, 1)

    if f_slope > 1e-4:
        extra     = (threshold - float(preds[-1])) / f_slope
        total_rul = float(len(preds)) + extra
        # WHY: if extrapolation says >300 cycles, forecast slope is too flat to trust
        # Use regressor instead — it's more honest about current health state
        if total_rul > 300:
            _LAST_WAS_FALLBACK = True
            return _health_index_to_rul(float(observed[-1]))
        return float(np.clip(total_rul, 0.0, RUL_CAP))

    # Step 3: forecast flat — use health_index → RUL regressor
    # More informative than linear extrapolation on flat observed tail
    _LAST_WAS_FALLBACK = True
    return _health_index_to_rul(float(observed[-1]))

def _invert_diff(
    diff_preds: np.ndarray,
    original: np.ndarray,
    d: int,
) -> np.ndarray:
    """
    Invert d levels of differencing on forecast values.

    d=2: diff2 → diff1 (seed = last observed velocity) → level (seed = last observed value)
    d=1: diff1 → level only.
    d=0: no inversion.
    """
    preds = np.asarray(diff_preds, dtype=float)
    if d == 0:
        return preds

    seeds = []
    temp  = original.copy()
    for _ in range(d):
        seeds.append(np.diff(temp, n=1)[-1])
        temp = np.diff(temp, n=1)

    current = preds
    for level in range(d - 1, -1, -1):
        seed    = seeds[level] if level < len(seeds) else original[-1]
        current = seed + np.cumsum(current)

    return current


# ─────────────────────────────────────────────
# 10. AR PREDICT — CH05
# ─────────────────────────────────────────────

def predict_rul_ar(
    series: np.ndarray,
    threshold: float,
    p: int = DEFAULT_AR_P,
    d: int = DEFAULT_ARIMA_D,
    smooth_window: int = SMOOTH_WINDOW,
) -> float:
    smoothed = smooth_series(series, smooth_window)
    if len(smoothed) <= p + d + 5:
        return _linear_extrapolation_rul(smoothed, threshold)
    try:
        # WHY: pass original series with d built into SARIMAX order
        # instead of manually diffing + inverting (which accumulates error)
        # This is identical to how predict_rul_arima works — which scores 83k NASA
        res   = SARIMAX(smoothed, order=(p, d, 0), simple_differencing=False).fit(disp=False)
        preds = res.forecast(steps=MAX_HORIZON)
        # preds are already at original scale — no inversion needed
        return _estimate_rul_from_forecast(preds, smoothed, threshold)
    except Exception:
        return _linear_extrapolation_rul(smoothed, threshold)


# ─────────────────────────────────────────────
# 11. ARMA PREDICT — CH06
# ─────────────────────────────────────────────

def predict_rul_arma(
    series: np.ndarray,
    threshold: float,
    p: int = DEFAULT_ARMA_P,
    q: int = DEFAULT_ARMA_Q,
    d: int = DEFAULT_ARIMA_D,
    smooth_window: int = SMOOTH_WINDOW,
) -> float:
    smoothed = smooth_series(series, smooth_window)
    if len(smoothed) <= p + q + d + 3:
        return _linear_extrapolation_rul(smoothed, threshold)
    try:
        # WHY: same fix as AR — let SARIMAX handle differencing internally
        # Manual diff → fit → invert was causing forecast divergence in v1
        res   = SARIMAX(smoothed, order=(p, d, q), simple_differencing=False).fit(disp=False)
        preds = res.forecast(steps=MAX_HORIZON)
        return _estimate_rul_from_forecast(preds, smoothed, threshold)
    except Exception:
        return _linear_extrapolation_rul(smoothed, threshold)


# ─────────────────────────────────────────────
# 12. ARIMA PREDICT — CH07
# ─────────────────────────────────────────────

def predict_rul_arima(
    series: np.ndarray,
    threshold: float,
    p: int = DEFAULT_ARIMA_P,
    d: int = DEFAULT_ARIMA_D,
    q: int = DEFAULT_ARIMA_Q,
    smooth_window: int = SMOOTH_WINDOW,
) -> float:
    """ARIMA(p,d,q) via SARIMAX. Passes original series — SARIMAX handles differencing."""
    smoothed = smooth_series(series, smooth_window)
    if len(smoothed) <= p + d + q + 3:
        return _linear_extrapolation_rul(smoothed, threshold)
    try:
        res   = SARIMAX(smoothed, order=(p, d, q), simple_differencing=False).fit(disp=False)
        ARIMA_HORIZON = min(MAX_HORIZON, 150)
        preds = res.forecast(steps=ARIMA_HORIZON)
        return _estimate_rul_from_forecast(preds, smoothed, threshold)
    except Exception:
        return _linear_extrapolation_rul(smoothed, threshold)


# ─────────────────────────────────────────────
# 13. DATASET-LEVEL PREDICTION
# ─────────────────────────────────────────────

# Module-level flag — predict_rul_* functions set this to True when they fall back
# This avoids changing function signatures


def predict_dataset(
    df: pd.DataFrame,
    predict_fn: Callable[[np.ndarray], float],
    threshold: float,
    verbose_engines: bool = False,   # NEW: print per-engine if True — helps debug
) -> tuple[np.ndarray, np.ndarray]:
    
    global _LAST_WAS_FALLBACK

    df = df.sort_values(["engine_id", "cycle"])

    y_true, y_pred = [], []
    fallback_count = 0
    total_count    = 0

    for _, g in df.groupby("engine_id", sort=False):
        series   = g["health_index"].values
        true_rul = float(g["RUL"].iloc[-1])
        eid      = g["engine_id"].iloc[0]

        # ── Step 1: call model ────────────────────────────────────────
        _LAST_WAS_FALLBACK = False          # reset flag before each call
        pred_raw = predict_fn(series, threshold=threshold)

        # ── Step 2: detect fallback BEFORE clipping ───────────────────
        # WHY: checking after np.clip(pred_raw, 0, 125) is wrong —
        #      a legitimate pred of 125 looks identical to a capped fallback
        is_fallback = _LAST_WAS_FALLBACK
        if is_fallback:
            fallback_count += 1

        # ── Step 3: clip to valid RUL range ───────────────────────────
        pred = float(np.clip(pred_raw * SAFETY_FACTOR, 0.0, RUL_CAP))
        # ── Step 4: optional per-engine print for debugging ───────────
        if verbose_engines:
            tag = " [FALLBACK]" if is_fallback else ""
            print(f"    engine {eid:>4d}  true={true_rul:6.1f}  "
                  f"pred={pred:6.1f}  err={pred-true_rul:+.1f}{tag}")

        total_count += 1
        y_true.append(true_rul)
        y_pred.append(pred)

    fallback_rate = 100.0 * fallback_count / total_count if total_count > 0 else 0.0
    print(f"  Fallback rate: {fallback_rate:.1f}%  ({fallback_count}/{total_count} engines)")

    return np.array(y_true), np.array(y_pred)


# ─────────────────────────────────────────────
# 14. SIMULATE TEST FROM TRAIN
# ─────────────────────────────────────────────

def simulate_test_from_train(
    train_df: pd.DataFrame,
    cutoff_range: tuple[float, float] = (0.2, 0.9),
    random_seed: int = 42,
    max_engines: int | None = None,
) -> pd.DataFrame:
    """
    Simulated validation set by truncating training engine histories.
    cutoff_range=(0.2, 0.9) matches the distribution of the real test set.
    """
    rng     = np.random.default_rng(random_seed)
    engines = train_df["engine_id"].unique()

    if max_engines is not None and len(engines) > max_engines:
        engines = rng.choice(engines, size=max_engines, replace=False)

    lo_frac, hi_frac = cutoff_range
    rows = []

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


def validate_model_rolling(
    train: pd.DataFrame,
    order: tuple,
    n_engines: int = 10,
    train_split: float = 0.7,
    model_name: str = "ARIMA",
):
    """
    Walk-forward validation on N engines from train set.
    
    Logic Flow:
    1. Sample N engines from train
    2. For each engine: split into train/val (70/30)
    3. Rolling one-step-ahead forecast on val portion
    4. Compute RMSE per engine
    5. Plot observed vs rolling forecast
    """
    from src.models.classical import rolling_forecast_engine, smooth_series

    eids       = train["engine_id"].unique()[:n_engines]
    all_rmse   = []

    fig, axes  = plt.subplots(n_engines, 1,
                              figsize=(12, 3 * n_engines),
                              sharex=False)
    axes = np.atleast_1d(axes)

    for ax, eid in zip(axes, eids):

        # ── Step 1: get smoothed series ───────────────────────────────
        raw    = (train[train["engine_id"] == eid]
                  .sort_values("cycle")["health_index"].values)
        series = smooth_series(raw)

        if len(series) < 20:
            continue

        # ── Step 2: train/val split ───────────────────────────────────
        train_len = int(len(series) * train_split)
        val_len   = len(series) - train_len

        # ── Step 3: rolling one-step-ahead forecast ───────────────────
        # WHY: window=1 = most honest validation — each step only uses
        # data available at that point in time (no lookahead)
        rolled = rolling_forecast_engine(
            series    = series,
            train_len = train_len,
            order     = order,
            window    = 1,
        )

        actual_val = series[train_len: train_len + len(rolled)]

        # ── Step 4: RMSE ──────────────────────────────────────────────
        rmse_eng = float(np.sqrt(np.mean((actual_val - rolled) ** 2)))
        all_rmse.append(rmse_eng)

        # ── Step 5: plot ──────────────────────────────────────────────
        obs_x    = np.arange(len(series))
        rolled_x = np.arange(train_len, train_len + len(rolled))

        ax.plot(obs_x,    series, color="steelblue", lw=1.5,
                label="Observed health_index")
        ax.plot(rolled_x, rolled, color="orange",    lw=1.5, ls="--",
                label=f"Rolling forecast (RMSE={rmse_eng:.3f})")
        ax.axvline(train_len, color="gray", ls=":", lw=1,
                   label="Train/Val split")
        ax.set_title(f"Engine {eid} — {model_name} Walk-Forward Validation")
        ax.set_ylabel("health_index")
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Cycle")
    plt.tight_layout()
    plt.show()

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"{model_name} Walk-Forward Validation Summary")
    print(f"{'='*40}")
    print(f"Engines validated : {len(all_rmse)}")
    print(f"Mean RMSE         : {np.mean(all_rmse):.4f}")
    print(f"Std RMSE          : {np.std(all_rmse):.4f}")
    print(f"Best engine RMSE  : {np.min(all_rmse):.4f}")
    print(f"Worst engine RMSE : {np.max(all_rmse):.4f}")
    print(f"{'='*40}")
    return np.array(all_rmse)
