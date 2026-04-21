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

Multi-engine order selection (NEW):
    select_best_ar_order   → samples N engines per dataset → modal best p
    select_best_arma_order → samples N engines per dataset → modal best (p,q)
    select_best_arima_order→ samples N engines per dataset → modal best (p,q) with fixed d

    Notebooks call ONE function instead of looping engines manually.
    The representative engine for Ljung-Box diagnostics is the longest engine.
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

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

RUL_CAP         = 125
MAX_HORIZON     = 150
SMOOTH_WINDOW   = 5
END_OF_LIFE_RUL = 5
EOL_HEALTH      = None

DEFAULT_AR_P    = 3
DEFAULT_ARMA_P  = 2
DEFAULT_ARMA_Q  = 2
DEFAULT_ARIMA_P = 2
DEFAULT_ARIMA_D = 2   # updated: ADF shows d=2 for CMAPSS health_index
DEFAULT_ARIMA_Q = 2


# ─────────────────────────────────────────────
# 1. HEALTH INDEX (PCA on rolling-mean sensors)
# ─────────────────────────────────────────────

def _combine_components(pca, X, signs, n_comp):
    pc = pca.transform(X)
    result = []
    for i in range(n_comp):
        c = pc[:, i] * signs[i]
        result.append(c)
    return result[0] if n_comp == 1 else np.maximum(result[0], result[1])


def build_pca_health_index(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sensor_cols: list[str],
    rolling_window: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert sensor readings into a scalar health_index via PCA.

    Strategy:
    1. Check if op_cluster explains >30% of PC1 variance.
       If yes → per-cluster PCA (FD002/FD004).
       If no  → global PCA (FD001/FD003).
    2. Use 2 PCA components and take element-wise max for FD003/FD004.
    3. Sign-flip each component so higher = more degraded.
    4. Standardize per dataset_id using train statistics.
    """
    from sklearn.metrics import r2_score as _r2
    from sklearn.preprocessing import LabelEncoder

    rmean_cols = [f"{c}_rmean_{rolling_window}" for c in sensor_cols]
    use_cols   = rmean_cols if all(c in train.columns for c in rmean_cols) else sensor_cols

    train = train.copy()
    test  = test.copy()

    use_per_cluster_map = {}
    use_2components_map = {}

    for did, grp in train.groupby("dataset_id"):
        X_grp  = grp[use_cols].values
        pc1    = PCA(n_components=1).fit_transform(X_grp).ravel()
        op_enc = LabelEncoder().fit_transform(grp["op_cluster"].values)
        r2_val = _r2(op_enc, pc1)
        use_per_cluster_map[did] = r2_val > 0.3
        use_2components_map[did] = did in (3, 4)
        print(f"FD00{did}: R2={r2_val:.3f} | per_cluster={use_per_cluster_map[did]} | 2comp={use_2components_map[did]}")

    train["health_index"] = np.nan
    test["health_index"]  = np.nan

    for did, tr_grp in train.groupby("dataset_id"):
        te_grp = test[test["dataset_id"] == did]
        n_comp = 2 if use_2components_map[did] else 1

        if use_per_cluster_map[did]:
            for cluster_id in tr_grp["op_cluster"].unique():
                tr_c = tr_grp[tr_grp["op_cluster"] == cluster_id]
                te_c = te_grp[te_grp["op_cluster"] == cluster_id]
                if len(tr_c) < 4:
                    continue
                pca   = PCA(n_components=n_comp).fit(tr_c[use_cols].values)
                pc_tr = pca.transform(tr_grp[use_cols].values)
                signs = []
                for i in range(n_comp):
                    c    = pc_tr[:, i]
                    sign = 1.0 if np.corrcoef(c, -tr_grp["RUL"].values)[0, 1] >= 0 else -1.0
                    signs.append(sign)
                hi_tr = _combine_components(pca, tr_grp[use_cols].values, signs, n_comp)
                hi_te = _combine_components(pca, te_grp[use_cols].values, signs, n_comp)
                train.loc[tr_grp.index, "health_index"] = hi_tr
                test.loc[te_grp.index, "health_index"]  = hi_te
        else:
            pca   = PCA(n_components=n_comp).fit(tr_grp[use_cols].values)
            pc_tr = pca.transform(tr_grp[use_cols].values)
            signs = []
            for i in range(n_comp):
                c    = pc_tr[:, i]
                sign = 1.0 if np.corrcoef(c, -tr_grp["RUL"].values)[0, 1] >= 0 else -1.0
                signs.append(sign)
            hi_tr = _combine_components(pca, tr_grp[use_cols].values, signs, n_comp)
            hi_te = _combine_components(pca, te_grp[use_cols].values, signs, n_comp)
            train.loc[tr_grp.index, "health_index"] = hi_tr
            test.loc[te_grp.index, "health_index"]  = hi_te

    # Standardize per dataset_id using train stats
    for did in train["dataset_id"].unique():
        tr_mask = train["dataset_id"] == did
        te_mask = test["dataset_id"]  == did
        mu = train.loc[tr_mask, "health_index"].mean()
        sd = train.loc[tr_mask, "health_index"].std()
        if sd < 1e-6:
            continue
        train.loc[tr_mask, "health_index"] = (train.loc[tr_mask, "health_index"] - mu) / sd
        test.loc[te_mask,  "health_index"] = (test.loc[te_mask,  "health_index"] - mu) / sd

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
    Per-dataset_id failure threshold from near-end-of-life training rows.
    Returns {dataset_id: threshold_value}.
    """
    eol_rows   = train[train["RUL"] <= end_of_life_rul]
    thresholds = {}
    for did, group in eol_rows.groupby("dataset_id"):
        thresholds[int(did)] = float(group["health_index"].quantile(quantile))
    return thresholds


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
    result["recommended_d"] = 2   # cap at 2 per book
    return result


def run_stationarity_report(
    train: pd.DataFrame,
    n_engines_per_subset: int = 5,
) -> pd.DataFrame:
    """
    Stratified ADF report across all dataset_ids.
    Prints per-engine p-values and the modal recommended_d.
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
# 4. ACF / PACF — CH05/CH06/CH07
# ─────────────────────────────────────────────

def plot_acf_pacf(
    series: np.ndarray,
    lags: int = 20,
    title: str = "ACF / PACF — health_index",
) -> None:
    """
    ACF + PACF side by side.

    Reading guide:
        PACF cuts off at lag p → AR(p) candidate
        ACF  cuts off at lag q → MA(q) candidate
        Both tail off          → ARMA(p,q) needed
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
    n_engines_per_dataset: int = 2,
    lags: int = 20,
) -> None:
    """
    ACF/PACF grid across multiple engines from all 4 datasets.

    Notebooks call this ONE function instead of picking a single eid.
    Series is differenced d times before plotting (per CH07 rule).

    Args:
        train:                 full training DataFrame with health_index column.
        d:                     differencing order from run_stationarity_report.
        n_engines_per_dataset: how many engines to sample per FD subset.
        lags:                  number of lags for ACF/PACF.
    """
    dataset_ids = sorted(train["dataset_id"].unique())
    total       = len(dataset_ids) * n_engines_per_dataset
    fig, axes   = plt.subplots(total, 2, figsize=(12, 4 * total))
    axes        = np.atleast_2d(axes)

    row = 0
    for did in dataset_ids:
        eids = train[train["dataset_id"] == did]["engine_id"].unique()[:n_engines_per_dataset]
        for eid in eids:
            raw  = train[train["engine_id"] == eid].sort_values("cycle")["health_index"].values
            smth = smooth_series(raw)
            # Difference d times before plotting — shows what the model actually sees
            series_plot = np.diff(smth, n=d) if d > 0 else smth

            plot_acf(series_plot,  lags=lags, ax=axes[row][0])
            axes[row][0].set_title(f"ACF — FD00{did} engine {eid} (diff-{d})")
            plot_pacf(series_plot, lags=lags, ax=axes[row][1])
            axes[row][1].set_title(f"PACF — FD00{did} engine {eid} (diff-{d})")
            row += 1

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 5. AIC-BASED ORDER SELECTION — CH06/CH07
# ─────────────────────────────────────────────

def optimize_AR(
    endog: Union[pd.Series, np.ndarray, list],
    p_values: list[int],
) -> pd.DataFrame:
    """
    Select AR(p) by AIC on a SINGLE already-stationary series.
    Called internally by select_best_ar_order — no need to call directly.
    """
    results = []
    for p in p_values:
        try:
            model = SARIMAX(endog, order=(p, 0, 0), simple_differencing=False).fit(disp=False)
            results.append({"p": p, "AIC": round(model.aic, 2)})
        except Exception:
            continue
    return (
        pd.DataFrame(results)
        .sort_values("AIC", ascending=True)
        .reset_index(drop=True)
    )


def optimize_ARMA(
    endog: Union[pd.Series, np.ndarray, list],
    order_list: list[tuple[int, int]],
) -> pd.DataFrame:
    """
    Select ARMA(p,q) by AIC on a SINGLE already-stationary series.
    Called internally by select_best_arma_order — no need to call directly.
    """
    results = []
    for p, q in order_list:
        try:
            model = SARIMAX(endog, order=(p, 0, q), simple_differencing=False).fit(disp=False)
            results.append({"(p,q)": (p, q), "AIC": round(model.aic, 2)})
        except Exception:
            continue
    return (
        pd.DataFrame(results)
        .sort_values("AIC", ascending=True)
        .reset_index(drop=True)
    )


def optimize_ARIMA(
    endog: Union[pd.Series, np.ndarray, list],
    order_list: list[tuple[int, int]],
    d: int,
) -> pd.DataFrame:
    """
    Select ARIMA(p,d,q) by AIC on the ORIGINAL (un-differenced) series.
    SARIMAX handles differencing internally.
    Called internally by select_best_arima_order — no need to call directly.
    """
    results = []
    for p, q in order_list:
        try:
            model = SARIMAX(endog, order=(p, d, q), simple_differencing=False).fit(disp=False)
            results.append({"(p,q)": (p, q), "d": d, "AIC": round(model.aic, 2)})
        except Exception:
            continue
    return (
        pd.DataFrame(results)
        .sort_values("AIC", ascending=True)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────
# 5b. MULTI-ENGINE ORDER SELECTION (NEW)
# ─────────────────────────────────────────────

def _get_representative_engine(train: pd.DataFrame) -> tuple[int, np.ndarray]:
    """
    Return (eid, smoothed_series) for the longest engine across all datasets.
    Used for Ljung-Box diagnostic fit in notebooks.
    """
    eid = train.groupby("engine_id")["cycle"].count().idxmax()
    raw = train[train["engine_id"] == eid].sort_values("cycle")["health_index"].values
    return int(eid), smooth_series(raw)


def select_best_ar_order(
    train: pd.DataFrame,
    d: int,
    p_values: list[int] | None = None,
    n_engines_per_dataset: int = 3,
) -> int:
    """
    Sample N engines per FD dataset, run optimize_AR on each diff-d series,
    return the modal best p across all engines.

    Notebooks call this ONE function — no manual engine loop needed.

    Args:
        train:                 full training DataFrame.
        d:                     differencing order from run_stationarity_report.
        p_values:              AR lag candidates. Defaults to [1..10].
        n_engines_per_dataset: engines to sample per FD subset.

    Returns:
        best_p: modal best AR order (int).
    """
    if p_values is None:
        p_values = list(range(1, 11))

    best_ps = []
    for did in sorted(train["dataset_id"].unique()):
        eids = train[train["dataset_id"] == did]["engine_id"].unique()[:n_engines_per_dataset]
        for eid in eids:
            raw  = train[train["engine_id"] == eid].sort_values("cycle")["health_index"].values
            smth = smooth_series(raw)
            diff = np.diff(smth, n=d) if d > 0 else smth

            if len(diff) < max(p_values) + 5:
                continue   # too short to fit reliably

            aic_df = optimize_AR(diff, p_values)
            if aic_df.empty:
                continue

            best_p = int(aic_df.iloc[0]["p"])
            best_ps.append(best_p)
            print(f"  FD00{did} engine {eid}: best p={best_p}  (AIC={aic_df.iloc[0]['AIC']})")

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
    n_engines_per_dataset: int = 3,
) -> tuple[int, int]:
    """
    Sample N engines per FD dataset, run optimize_ARMA on each diff-d series,
    return the modal best (p,q) across all engines.

    Notebooks call this ONE function — no manual engine loop needed.

    Args:
        train:                 full training DataFrame.
        d:                     differencing order from run_stationarity_report.
        p_range:               range of p values. Defaults to range(1,6).
        q_range:               range of q values. Defaults to range(1,6).
        n_engines_per_dataset: engines to sample per FD subset.

    Returns:
        (best_p, best_q): modal best ARMA order tuple.
    """
    if p_range is None:
        p_range = range(1, 6)
    if q_range is None:
        q_range = range(1, 6)

    order_list = list(product(p_range, q_range))
    best_pqs   = []

    for did in sorted(train["dataset_id"].unique()):
        eids = train[train["dataset_id"] == did]["engine_id"].unique()[:n_engines_per_dataset]
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
            print(f"  FD00{did} engine {eid}: best (p,q)={best_pq}  (AIC={aic_df.iloc[0]['AIC']})")

    if not best_pqs:
        print("  WARNING: no valid engines found, returning default (2,2)")
        return (2, 2)

    modal_pq  = Counter(best_pqs).most_common(1)[0][0]
    best_p, best_q = int(modal_pq[0]), int(modal_pq[1])
    print(f"\n→ Modal best ARMA order: ({best_p},{best_q})  "
          f"(from {len(best_pqs)} engines, freq={Counter(best_pqs).most_common(5)})")
    return best_p, best_q


def select_best_arima_order(
    train: pd.DataFrame,
    d: int,
    p_range: range | None = None,
    q_range: range | None = None,
    n_engines_per_dataset: int = 3,
) -> tuple[int, int]:
    """
    Sample N engines per FD dataset, run optimize_ARIMA on each ORIGINAL series,
    return the modal best (p,q) across all engines.

    Note: passes original (un-differenced) smth to SARIMAX — it handles d internally.

    Notebooks call this ONE function — no manual engine loop needed.

    Args:
        train:                 full training DataFrame.
        d:                     differencing order (from run_stationarity_report).
        p_range:               range of p values. Defaults to range(1,6).
        q_range:               range of q values. Defaults to range(1,6).
        n_engines_per_dataset: engines to sample per FD subset.

    Returns:
        (best_p, best_q): modal best (p,q) for ARIMA(p,d,q).
    """
    if p_range is None:
        p_range = range(1, 6)
    if q_range is None:
        q_range = range(1, 6)

    order_list = list(product(p_range, q_range))
    best_pqs   = []

    for did in sorted(train["dataset_id"].unique()):
        eids = train[train["dataset_id"] == did]["engine_id"].unique()[:n_engines_per_dataset]
        for eid in eids:
            raw  = train[train["engine_id"] == eid].sort_values("cycle")["health_index"].values
            smth = smooth_series(raw)   # ORIGINAL — no manual diff for ARIMA

            if len(smth) < max(p_range) + d + max(q_range) + 5:
                continue

            aic_df = optimize_ARIMA(smth, order_list, d=d)
            if aic_df.empty:
                continue

            best_pq = aic_df.iloc[0]["(p,q)"]
            best_pqs.append(tuple(best_pq))
            print(f"  FD00{did} engine {eid}: best (p,q)={best_pq}  (AIC={aic_df.iloc[0]['AIC']})")

    if not best_pqs:
        print("  WARNING: no valid engines found, returning default (2,2)")
        return (2, 2)

    modal_pq   = Counter(best_pqs).most_common(1)[0][0]
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
    Ljung-Box test on residuals. Required by book CH06 and CH07.

    p > 0.05 for all lags → white noise → model adequate.
    p < 0.05 for any lag  → residual autocorrelation → consider higher order.

    plot_qq=True adds QQ plot (CH07 ARIMA requirement).
    """
    lb_result = acorr_ljungbox(residuals, lags=np.arange(1, 11, 1), return_df=True)

    print(f"\nLjung-Box residual test — {model_name}")
    print(lb_result[["lb_stat", "lb_pvalue"]].to_string())

    all_pass = (lb_result["lb_pvalue"] > 0.05).all()
    if all_pass:
        print("✓ All p-values > 0.05 — residuals are white noise (model is adequate)")
    else:
        print("✗ Some p-values < 0.05 — residual autocorrelation remains")

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
# 7. ROLLING FORECAST — CH05/CH06/CH07
# ─────────────────────────────────────────────

def rolling_forecast_engine(
    series: np.ndarray,
    train_len: int,
    order: tuple[int, int, int],
    window: int = 1,
) -> np.ndarray:
    """
    Walk-forward forecast on ONE engine's health_index series.
    Mirrors the book's rolling_forecast exactly (CH05/CH06/CH07).

    Args:
        series:    complete health_index for one engine.
        train_len: initial training window size.
        order:     (p, d, q) — best from select_best_* functions.
        window:    refit cadence. 1 = most accurate, slower.
    """
    p, d, q   = order
    total_len = len(series)
    pred      = []

    for i in range(train_len, total_len, window):
        model       = SARIMAX(series[:i], order=(p, d, q), simple_differencing=False)
        res         = model.fit(disp=False)
        predictions = res.get_prediction(start=0, end=i + window - 1)
        oos         = np.asarray(predictions.predicted_mean)[-window:]
        pred.extend(oos.tolist())

    return np.array(pred[: total_len - train_len])


# ─────────────────────────────────────────────
# 8. SMOOTHING
# ─────────────────────────────────────────────

def smooth_series(series: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Rolling-median smoother. Applied before any model fit."""
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
    Fallback: linear trend on tail of observed series → extrapolate to threshold.
    Used when forecast never crosses threshold within MAX_HORIZON.
    """
    if tail is None:
        tail = max(5, min(30, int(0.2 * len(series))))

    y = series[-tail:] if len(series) >= tail else series
    if len(y) < 3:
        return float(RUL_CAP)

    x                = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    last             = float(y[-1])

    if slope <= 1e-6:
        return float(RUL_CAP)

    steps = (threshold - last) / slope
    return float(min(max(steps, 0.0), RUL_CAP))


def _estimate_rul_from_forecast(
    preds: np.ndarray,
    observed: np.ndarray,
    threshold: float = None,
) -> float:
    """
    Find first forecast step that crosses threshold → that index = RUL.
    Falls back to linear extrapolation if no crossing within MAX_HORIZON.
    """
    preds = np.asarray(preds, dtype=float)
    if preds.size == 0 or not np.all(np.isfinite(preds)):
        return _linear_extrapolation_rul(observed, threshold)

    crossings = np.where(preds >= threshold)[0]
    if crossings.size > 0:
        return float(crossings[0])

    return _linear_extrapolation_rul(observed, threshold)


# ─────────────────────────────────────────────
# 10. AR PREDICT — CH05
# ─────────────────────────────────────────────

def predict_rul_ar(
    series: np.ndarray,
    threshold: float,
    p: int = DEFAULT_AR_P,
    d: int = DEFAULT_ARIMA_D,   # pre-diff before fitting since AR uses d=0
    smooth_window: int = SMOOTH_WINDOW,
) -> float:
    """
    AR(p) forecast via SARIMAX(order=(p, 0, 0)) on diff-d series.

    d is applied manually (np.diff) since AR = ARIMA(p,0,0).
    Forecast is inverted back to original scale before RUL estimation.
    """
    smoothed = smooth_series(series, smooth_window)

    if len(smoothed) <= p + d + 5:
        return _linear_extrapolation_rul(smoothed, threshold)

    try:
        diff_series = np.diff(smoothed, n=d) if d > 0 else smoothed

        model = SARIMAX(diff_series, order=(p, 0, 0), simple_differencing=False)
        res   = model.fit(disp=False)
        diff_preds = res.forecast(steps=MAX_HORIZON)

        # Invert differencing to restore original health_index scale
        level_preds = _invert_diff(diff_preds, smoothed, d)

        return _estimate_rul_from_forecast(level_preds, smoothed, threshold)
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
    d: int = DEFAULT_ARIMA_D,   # pre-diff manually — ARMA uses d=0
    smooth_window: int = SMOOTH_WINDOW,
) -> float:
    """
    ARMA(p,q) on diff-d series via SARIMAX(order=(p, 0, q)).

    d is applied manually (np.diff), forecast inverted back to original scale.
    This is equivalent to ARIMA(p,d,q) but uses the ARMA interface as the book does.
    """
    smoothed = smooth_series(series, smooth_window)

    if len(smoothed) <= p + q + d + 3:
        return _linear_extrapolation_rul(smoothed, threshold)

    try:
        diff_series = np.diff(smoothed, n=d) if d > 0 else smoothed

        model      = SARIMAX(diff_series, order=(p, 0, q), simple_differencing=False)
        res        = model.fit(disp=False)
        diff_preds = res.forecast(steps=MAX_HORIZON)

        # Invert differencing: diff-d preds → original health_index scale
        level_preds = _invert_diff(diff_preds, smoothed, d)

        return _estimate_rul_from_forecast(level_preds, smoothed, threshold)
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
    """
    ARIMA(p,d,q) via SARIMAX(order=(p, d, q)).

    Passes ORIGINAL series — SARIMAX handles differencing and inversion internally.
    This is more numerically stable than manual inversion in predict_rul_arma.
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
# 12b. DIFFERENCING INVERSION UTILITY
# ─────────────────────────────────────────────

def _invert_diff(
    diff_preds: np.ndarray,
    original: np.ndarray,
    d: int,
) -> np.ndarray:
    """
    Invert np.diff(original, n=d) applied to forecast values.

    Logic for d=2 (generalises to any d):
        diff2_preds → diff1_preds:
            seed = original[-1] - original[-2]   (last observed velocity)
            diff1_preds = seed + cumsum(diff2_preds)

        diff1_preds → level_preds:
            seed = original[-1]                  (last observed level)
            level_preds = seed + cumsum(diff1_preds)

    For d=1: only the second step above is needed.
    For d=0: no inversion needed, return diff_preds unchanged.
    """
    preds = np.asarray(diff_preds, dtype=float)

    if d == 0:
        return preds

    # Iteratively invert one difference at a time, from highest order down to level
    # seeds[k] = last value of the k-th difference of original
    #   seeds[d]   = original[-1]                      (level)
    #   seeds[d-1] = first diff of original: [-1] - [-2]
    #   seeds[d-2] = second diff ...
    # We invert from order d down to order 0

    # Compute seeds for each level of differentiation
    seeds = []
    temp  = original.copy()
    for _ in range(d):
        seeds.append(np.diff(temp, n=1)[-1])   # last value of each diff level
        temp = np.diff(temp, n=1)
    # seeds[0] = last of diff-1, seeds[1] = last of diff-2, ...
    # We invert from innermost (diff-d) outward to level

    current = preds
    for level in range(d - 1, -1, -1):
        # seed for this inversion level
        seed    = seeds[level] if level < len(seeds) else original[-1]
        current = seed + np.cumsum(current)

    return current


# ─────────────────────────────────────────────
# 13. DATASET-LEVEL PREDICTION
# ─────────────────────────────────────────────

def predict_dataset(
    df: pd.DataFrame,
    predict_fn: Callable[[np.ndarray], float],
    threshold_map: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply predict_fn to every engine in df. Returns (y_true, y_pred, dataset_ids).

    predict_fn must already have p/d/q bound via functools.partial.
    threshold is injected per dataset_id from threshold_map.
    Logs fallback rate.
    """
    df = df.sort_values(["engine_id", "cycle"])

    y_true, y_pred, dids = [], [], []
    fallback_count = 0
    total_count    = 0

    for _, g in df.groupby("engine_id", sort=False):
        series   = g["health_index"].values
        true_rul = float(g["RUL"].iloc[-1])
        did      = int(g["dataset_id"].iloc[0])
        thresh   = threshold_map.get(did, list(threshold_map.values())[0])

        pred_raw = predict_fn(series, threshold=thresh)
        pred     = float(np.clip(pred_raw, 0.0, RUL_CAP))

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
    cutoff_range: tuple[float, float] = (0.2, 0.9),
    random_seed: int = 42,
    max_engines: int | None = None,
) -> pd.DataFrame:
    """
    Create simulated validation set by truncating training engine histories.
    cutoff_range=(0.2, 0.9) covers early-life engines matching real test set.
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