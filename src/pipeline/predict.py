"""
predict.py — inference pipeline for all model types.

Loads saved weights (DL) or re-fits from data (classical), then
produces PredictionResult objects for every test engine.

Usage
-----
    from src.pipeline.predict import predict_dl, predict_classical, predict_mc_dropout

    # Point DL model (saved weights)
    results = predict_dl("GRU")

    # Quantile DL model
    results = predict_dl("Q_Transformer")

    # Classical with CI
    results = predict_classical("ARIMA", p=1, d=2, q=2)

    # MC Dropout uncertainty on a point model
    results = predict_mc_dropout("GRU")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.config import (
    PROC_DIR, ARTIFACTS_DIR, DL_CONFIG, CLASSICAL_CONFIG,
    DL_SENSOR_COLS, SENSOR_COLS, RUL_CAP,
)
from src.models.base import PredictionResult


# ══════════════════════════════════════════════════════════════════════════════
# DL INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def predict_dl(
    model_name: str,
    n_features: int | None = None,
    weights_path: Path | None = None,
    verbose: bool = True,
    **model_kwargs,
) -> list[PredictionResult]:
    """
    Load saved DL weights and run inference on the test set.

    Parameters
    ----------
    model_name   : key in ALL_MODELS registry, e.g. "GRU", "Q_Transformer"
    n_features   : override feature count (auto-detected from data if None)
    weights_path : explicit path to .pt file; defaults to ARTIFACTS_DIR/<model_name>.pt
    **model_kwargs : passed to build_model() to reproduce the architecture

    Returns
    -------
    list of PredictionResult, one per test engine.
    Point models: lower_bound = upper_bound = rul_pred (no uncertainty; use predict_mc_dropout instead).
    Quantile models: lower_bound = Q10, rul_pred = Q50, upper_bound = Q90.
    """
    import torch
    import torch.utils.data as tud
    from src.models.deep_learning import (
        load_data, select_features, build_windows,
        make_loaders, predict_test, predict_quantiles,
        DEVICE, WINDOW_SIZE, BATCH_SIZE,
    )
    from src.models.dl_architectures import build_model, QUANTILE_MODELS

    is_quantile = model_name in QUANTILE_MODELS

    # ── Load data ──────────────────────────────────────────────────────────────
    _, test_df = load_data()
    feat_cols = select_features(test_df)
    _n_feat   = n_features or len(feat_cols)

    # ── Build test windows ─────────────────────────────────────────────────────
    X_test, y_test = build_windows(test_df, feat_cols, is_test=True)
    X_tensor  = torch.tensor(X_test,  dtype=torch.float32)
    y_tensor  = torch.tensor(y_test,  dtype=torch.float32)
    test_loader = tud.DataLoader(
        tud.TensorDataset(X_tensor, y_tensor),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # ── Load model ─────────────────────────────────────────────────────────────
    model = build_model(model_name, n_features=_n_feat, **model_kwargs).to(DEVICE)
    path  = weights_path or (ARTIFACTS_DIR / f"{model_name}.pt")
    if not path.exists():
        raise FileNotFoundError(
            f"No saved weights at {path}. Run run_dl_training('{model_name}', save=True) first."
        )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    if verbose:
        print(f"[{model_name}] loaded weights from {path.relative_to(ARTIFACTS_DIR.parent)}")

    # ── Predict ────────────────────────────────────────────────────────────────
    engine_ids = list(range(len(y_test)))

    if is_quantile:
        y_true, q10, q50, q90 = predict_quantiles(model, test_loader)
        return [
            PredictionResult(
                engine_id=i,
                rul_pred=float(q50[i]),
                lower_bound=float(q10[i]),
                upper_bound=float(q90[i]),
                confidence_width=float(q90[i] - q10[i]),
                model_name=model_name,
            )
            for i in engine_ids
        ]
    else:
        y_true, y_pred = predict_test(model, test_loader)
        return [
            PredictionResult(
                engine_id=i,
                rul_pred=float(y_pred[i]),
                lower_bound=float(y_pred[i]),
                upper_bound=float(y_pred[i]),
                confidence_width=0.0,
                model_name=model_name,
            )
            for i in engine_ids
        ]


# ══════════════════════════════════════════════════════════════════════════════
# MC DROPOUT INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def predict_mc_dropout(
    model_name: str,
    n_features: int | None = None,
    weights_path: Path | None = None,
    n_samples: int = DL_CONFIG["mc_dropout_samples"],
    p_drop: float = 0.1,
    verbose: bool = True,
    **model_kwargs,
) -> list[PredictionResult]:
    """
    Load a saved point-prediction DL model and wrap it with MC Dropout
    to produce Q10/Q50/Q90 uncertainty estimates.

    Parameters
    ----------
    model_name : key in POINT_MODELS, e.g. "GRU", "LSTM"
    n_samples  : stochastic forward passes (default 30)
    p_drop     : additional dropout probability applied to each forward pass

    Returns
    -------
    list of PredictionResult with Q10 lower_bound, Q50 rul_pred, Q90 upper_bound.
    """
    import torch
    from src.models.deep_learning import load_data, select_features, build_windows, DEVICE, BATCH_SIZE
    from src.models.dl_architectures import build_model, QUANTILE_MODELS
    from src.models.uncertainty import MCDropout

    if model_name in QUANTILE_MODELS:
        raise ValueError(
            f"'{model_name}' is already a quantile model. Use predict_dl() instead."
        )

    # ── Load data & build test windows ─────────────────────────────────────────
    _, test_df = load_data()
    feat_cols  = select_features(test_df)
    _n_feat    = n_features or len(feat_cols)
    X_test, y_test = build_windows(test_df, feat_cols, is_test=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    model = build_model(model_name, n_features=_n_feat, **model_kwargs).to(DEVICE)
    path  = weights_path or (ARTIFACTS_DIR / f"{model_name}.pt")
    if not path.exists():
        raise FileNotFoundError(
            f"No saved weights at {path}. Run run_dl_training('{model_name}', save=True) first."
        )
    model.load_state_dict(torch.load(path, map_location=DEVICE))

    # ── MC Dropout predict ─────────────────────────────────────────────────────
    mc = MCDropout(model, p_drop=p_drop)
    q_low, q_mid, q_high, std = mc.predict(
        X_test, n_samples=n_samples, device=DEVICE
    )

    if verbose:
        print(
            f"[MC-{model_name}] n_samples={n_samples}  "
            f"mean_width={float(np.mean(q_high - q_low)):.1f}  "
            f"mean_std={float(np.mean(std)):.2f}"
        )

    return [
        PredictionResult(
            engine_id=i,
            rul_pred=float(q_mid[i]),
            lower_bound=float(q_low[i]),
            upper_bound=float(q_high[i]),
            confidence_width=float(q_high[i] - q_low[i]),
            model_name=f"MC_{model_name}",
        )
        for i in range(len(y_test))
    ]


# ══════════════════════════════════════════════════════════════════════════════
# CLASSICAL INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def predict_classical(
    model_type: str,
    n_selection_engines: int = 15,
    verbose: bool = True,
    **order_kwargs,
) -> list[PredictionResult]:
    """
    Run classical model inference (AR or ARIMA) on the test set.

    This function is stateless — it re-derives model orders from training data
    (or uses supplied order_kwargs) each call, matching the statsmodels workflow.

    Parameters
    ----------
    model_type : "AR" or "ARIMA" (covers "ARMA" — internally always ARIMA)
    **order_kwargs : p, d, q overrides to skip AIC order selection

    Returns
    -------
    list of PredictionResult with SARIMAX conf_int uncertainty bounds.
    """
    from functools import partial
    from src.models.classical import (
        load_and_prepare,
        run_stationarity_report,
        select_best_ar_order,
        select_best_arima_order,
        predict_rul_ar_with_ci,
        predict_rul_arima_with_ci,
        predict_dataset_with_ci,
    )

    # ── Load ───────────────────────────────────────────────────────────────────
    train, test, threshold = load_and_prepare(PROC_DIR, SENSOR_COLS)

    # ── Stationarity → d ───────────────────────────────────────────────────────
    if "d" not in order_kwargs:
        stat_df = run_stationarity_report(train, n_engines=20)
        d = int(stat_df["recommended_d"].mode()[0])
    else:
        d = order_kwargs.pop("d")

    # ── Order selection ────────────────────────────────────────────────────────
    if model_type == "AR":
        p = order_kwargs.pop("p", None) or select_best_ar_order(
            train, d=d, n_engines=n_selection_engines
        )
        model_label = f"AR({p})"
        predict_fn  = partial(predict_rul_ar_with_ci, p=p, pre_diff_d=d)

    elif model_type in ("ARIMA", "ARMA"):
        p_val = order_kwargs.pop("p", None)
        q_val = order_kwargs.pop("q", None)
        if p_val is None or q_val is None:
            p_val, q_val = select_best_arima_order(
                train, d=d, n_engines=n_selection_engines
            )
        p, q = p_val, q_val
        model_label = f"ARIMA({p},{d},{q})"
        predict_fn  = partial(predict_rul_arima_with_ci, p=p, d=d, q=q)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'AR' or 'ARIMA'.")

    # ── Predict ────────────────────────────────────────────────────────────────
    y_true, y_pred, y_lower, y_upper, engine_ids = predict_dataset_with_ci(
        test, predict_fn, threshold, verbose_engines=verbose
    )

    return [
        PredictionResult(
            engine_id=int(eid),
            rul_pred=float(yp),
            lower_bound=float(yl),
            upper_bound=float(yu),
            confidence_width=float(yu - yl),
            model_name=model_label,
        )
        for eid, yp, yl, yu in zip(engine_ids, y_pred, y_lower, y_upper)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# CONFORMAL CALIBRATION WRAPPER
# ══════════════════════════════════════════════════════════════════════════════


def apply_conformal(
    results: list[PredictionResult],
    cal_results: list[PredictionResult],
    target_coverage: float = 0.80,
) -> list[PredictionResult]:
    """
    Post-hoc conformal calibration: expand prediction intervals from
    a test-set results list using a calibration-set results list.

    Parameters
    ----------
    results      : list[PredictionResult] for the TEST set (to be calibrated)
    cal_results  : list[PredictionResult] for a held-out CAL set
                   (must have meaningful y_true — use results from a val fold)
    target_coverage : desired marginal coverage (default 0.80)

    Returns
    -------
    New list[PredictionResult] with expanded lower/upper bounds.
    """
    from src.models.uncertainty import conformal_calibrate, apply_conformal_margin

    if any(r.rul_true is None for r in cal_results):
        raise ValueError(
            "apply_conformal: all calibration PredictionResult objects must have rul_true set. "
            "Pass results from a held-out set where ground truth is known."
        )
    y_cal_true  = np.array([r.rul_true  for r in cal_results])
    y_cal_lower = np.array([r.lower_bound for r in cal_results])
    y_cal_upper = np.array([r.upper_bound for r in cal_results])

    delta = conformal_calibrate(y_cal_true, y_cal_lower, y_cal_upper, target_coverage)

    y_lower = np.array([r.lower_bound for r in results])
    y_upper = np.array([r.upper_bound for r in results])
    cal_lower, cal_upper = apply_conformal_margin(y_lower, y_upper, delta)

    model_name = results[0].model_name if results else "unknown"

    return [
        PredictionResult(
            engine_id=r.engine_id,
            rul_pred=r.rul_pred,
            lower_bound=float(cal_lower[i]),
            upper_bound=float(cal_upper[i]),
            confidence_width=float(cal_upper[i] - cal_lower[i]),
            model_name=f"{model_name}+conformal",
        )
        for i, r in enumerate(results)
    ]
