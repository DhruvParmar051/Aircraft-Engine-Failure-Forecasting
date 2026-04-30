"""
train.py — unified training pipeline for all model types.

Replaces the repeated 4-cell boilerplate in every DL notebook:
    load_data → build_windows → make_loaders → train_model/train_quantile_model

Usage
-----
    from src.pipeline.train import run_dl_training, run_classical_training

    # DL point model
    results = run_dl_training("GRU", save=True)

    # DL quantile model
    results = run_dl_training("Q_Transformer", save=True)

    # Classical (AR / ARIMA)
    results = run_classical_training("ARIMA", p=1, d=2, q=2, save=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import (
    PROC_DIR, ARTIFACTS_DIR, DL_CONFIG, CLASSICAL_CONFIG,
    DL_SENSOR_COLS, SENSOR_COLS, RUL_CAP,
)


# ══════════════════════════════════════════════════════════════════════════════
# DL TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def run_dl_training(
    model_name: str,
    n_features: int | None = None,
    save: bool = True,
    verbose: bool = True,
    **model_kwargs,
) -> dict[str, Any]:
    """
    End-to-end training for any registered DL model.

    Steps
    -----
    1. Load processed features (PROC_DIR).
    2. Select sensor columns.
    3. Build sliding-window arrays.
    4. Engine-level 80/20 train/val split.
    5. Build test windows (last window per engine).
    6. Instantiate model from registry.
    7. Train with NASALoss (point) or PinballLoss (quantile).
    8. Evaluate on test set.
    9. Save model weights to ARTIFACTS_DIR/<model_name>.pt (optional).

    Parameters
    ----------
    model_name   : key in ALL_MODELS registry, e.g. "GRU", "Q_Transformer"
    n_features   : feature count override (auto-detected from data if None)
    save         : persist trained weights to artifacts/
    **model_kwargs : passed to build_model() — override hidden_size, n_layers, etc.

    Returns
    -------
    dict with keys: model, y_true, y_pred, results, train_losses, val_losses
                    + q10, q50, q90 for quantile models
    """
    import torch
    from src.models.deep_learning import (
        load_data, select_features, build_windows, engine_split,
        make_loaders, train_model, train_quantile_model,
        predict_test, predict_quantiles, evaluate_quantile_model,
        DEVICE, WINDOW_SIZE, BATCH_SIZE,
    )
    from src.models.dl_architectures import build_model, QUANTILE_MODELS
    from src.evaluation.metrics import evaluate, save_model_results

    is_quantile = model_name in QUANTILE_MODELS

    # ── 1. Load ───────────────────────────────────────────────────────────────
    train_df, test_df = load_data()
    feat_cols  = select_features(train_df)
    _n_feat    = n_features or len(feat_cols)

    # ── 2. Windows ───────────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val = engine_split(train_df, feat_cols)
    X_test,  y_test                = build_windows(test_df, feat_cols, is_test=True)
    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # ── 3. Build model ────────────────────────────────────────────────────────
    model = build_model(model_name, n_features=_n_feat, **model_kwargs).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[{model_name}] parameters: {n_params:,}  device: {DEVICE}")

    # ── 4. Train ─────────────────────────────────────────────────────────────
    if is_quantile:
        model, train_losses, val_losses = train_quantile_model(
            model, train_loader, val_loader,
            quantiles=DL_CONFIG["quantiles"],
            epochs=DL_CONFIG["epochs"],
            lr=DL_CONFIG["lr"],
            model_name=model_name,
            patience=DL_CONFIG["patience"],
        )
    else:
        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            epochs=DL_CONFIG["epochs"],
            lr=DL_CONFIG["lr"],
            model_name=model_name,
            patience=DL_CONFIG["patience"],
        )

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    out = {
        "model":        model,
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "feat_cols":    feat_cols,
    }

    if is_quantile:
        y_true, q10, q50, q90 = predict_quantiles(model, test_loader)
        results, width, cov   = evaluate_quantile_model(y_true, q10, q50, q90, model_name)
        save_model_results(
            model_name=model_name, model_type="quantile",
            y_true=y_true, y_pred=q50, y_lower=q10, y_upper=q90,
        )
        out.update({"y_true": y_true, "q10": q10, "q50": q50, "q90": q90,
                    "results": results, "interval_width": width, "coverage": cov})
    else:
        y_true, y_pred = predict_test(model, test_loader)
        results = evaluate(y_true, y_pred, model_name=model_name, verbose=verbose)
        save_model_results(
            model_name=model_name, model_type="dl",
            y_true=y_true, y_pred=y_pred,
        )
        out.update({"y_true": y_true, "y_pred": y_pred, "results": results})

    # ── 6. Save weights ───────────────────────────────────────────────────────
    if save:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        weight_path = ARTIFACTS_DIR / f"{model_name}.pt"
        torch.save(model.state_dict(), weight_path)
        print(f"  → Weights saved to {weight_path.relative_to(ARTIFACTS_DIR.parent)}")

    return out


# ══════════════════════════════════════════════════════════════════════════════
# CLASSICAL TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def run_classical_training(
    model_type: str,              # "AR" | "ARIMA"
    n_selection_engines: int = 15,
    save: bool = True,
    verbose: bool = True,
    **order_kwargs,
) -> dict[str, Any]:
    """
    End-to-end training and evaluation for AR or ARIMA (covers T09 "ARMA" too).

    Steps
    -----
    1. Load processed features.
    2. Build PCA health index.
    3. Run ADF stationarity → determine d.
    4. AIC-based order selection on n_selection_engines.
    5. Fit and run Ljung-Box diagnostic.
    6. Evaluate on full test set with confidence intervals.
    7. Save per-engine predictions to results/predictions/.

    Parameters
    ----------
    model_type : "AR" or "ARIMA" (the T09 "ARMA" is ARIMA internally)
    **order_kwargs : override p, d, q directly (skip AIC selection)

    Returns
    -------
    dict with y_true, y_pred, y_lower, y_upper, results, engine_ids
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
    from src.evaluation.metrics import save_model_results, save_predictions_csv

    # ── 1. Load ───────────────────────────────────────────────────────────────
    train, test, threshold = load_and_prepare(PROC_DIR, SENSOR_COLS)

    # ── 2. Stationarity ───────────────────────────────────────────────────────
    if "d" not in order_kwargs:
        stat_df = run_stationarity_report(train, n_engines=20)
        d = int(stat_df["recommended_d"].mode()[0])
    else:
        d = order_kwargs.pop("d")

    # ── 3. Order selection ────────────────────────────────────────────────────
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
        # Always label as ARIMA — "ARMA" label was the naming bug
        model_label = f"ARIMA({p},{d},{q})"
        predict_fn  = partial(predict_rul_arima_with_ci, p=p, d=d, q=q)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'AR' or 'ARIMA'.")

    # ── 4. Predict test set with CI ───────────────────────────────────────────
    y_true, y_pred, y_lower, y_upper, engine_ids = predict_dataset_with_ci(
        test, predict_fn, threshold, verbose_engines=verbose
    )

    # ── 5. Evaluate and save ──────────────────────────────────────────────────
    from src.evaluation.metrics import evaluate
    results = evaluate(y_true, y_pred, model_name=model_label, verbose=True)
    save_model_results(
        model_name=model_label, model_type="classical",
        y_true=y_true, y_pred=y_pred, y_lower=y_lower, y_upper=y_upper,
    )
    if save:
        save_predictions_csv(engine_ids, y_true, y_pred, y_lower, y_upper, model_label)

    return {
        "model_label":  model_label,
        "y_true":       y_true,
        "y_pred":       y_pred,
        "y_lower":      y_lower,
        "y_upper":      y_upper,
        "engine_ids":   engine_ids,
        "results":      results,
        "threshold":    threshold,
    }
