"""
tft.py — Temporal Fusion Transformer pipeline for CMAPSS RUL prediction.

Built on ``pytorch-forecasting``. The TFT jointly learns:
    * static encoders for engine-level metadata (dataset_id, op_cluster)
    * per-timestep variable-selection networks for sensors + operating conditions
    * a multi-head attention block over the past encoder timesteps

Pipeline:
    1. ``prepare_tft_dataframe``  : add ``time_idx`` + cap RUL; keep the
       columns the model actually uses (keeps dataset light).
    2. ``build_tft_datasets``     : build the ``TimeSeriesDataSet`` objects
       for training and for the LAST-window-per-engine test evaluation.
    3. ``build_tft_model``        : instantiate TFT with sensible defaults.
    4. ``train_tft``              : run PyTorch Lightning training with
       early stopping + LR-reducer.
    5. ``predict_last_window``    : one RUL prediction per engine.
    6. ``interpret_tft``          : attention weights + variable importance.

Notes on framing RUL regression as TFT forecasting:
    - ``max_encoder_length = 30`` cycles of history.
    - ``max_prediction_length = 1``: we predict the RUL at the cycle just
      after the encoder window. Combined with how pytorch-forecasting treats
      each engine's data, this naturally yields "predict RUL at cycle N from
      cycles [N-30, N-1]".
    - At test time we use the LAST cycle of each engine as the decoder step,
      giving exactly one prediction per engine (matching the AR/ARMA/ARIMA
      evaluation protocol).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss, RMSE


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

RUL_CAP               = 125
MAX_ENCODER_LENGTH    = 30
MAX_PREDICTION_LENGTH = 1
SENSOR_COLS           = [f"s{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
OP_COLS               = ["op1", "op2", "op3"]


@dataclass
class TFTConfig:
    """Training and architecture hyperparameters."""
    hidden_size:       int   = 32
    attention_heads:   int   = 4
    dropout:           float = 0.2
    hidden_continuous: int   = 16
    lr:                float = 3e-3
    batch_size:        int   = 128
    max_epochs:        int   = 15
    patience:          int   = 3
    gradient_clip_val: float = 0.3
    accelerator:       str   = "auto"      # "cpu", "mps", "cuda", or "auto"
    seed:              int   = 42


# ─────────────────────────────────────────────
# 1. DATAFRAME PREP
# ─────────────────────────────────────────────

def prepare_tft_dataframe(
    df: pd.DataFrame,
    sensor_cols: list[str] = SENSOR_COLS,
    op_cols:     list[str] = OP_COLS,
) -> pd.DataFrame:
    """
    Shape the processed features for TFT consumption.

    - Adds a zero-based integer ``time_idx`` per engine (required by TFT).
    - Caps ``RUL`` at ``RUL_CAP`` (matches the training convention).
    - Casts group/static columns to string so pytorch-forecasting treats
      them as categoricals.
    - Keeps only the columns used by the model (much smaller memory).
    """
    needed = [
        "engine_id", "cycle", "dataset_id", "op_cluster", "RUL",
        *sensor_cols, *op_cols,
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    out = df[needed].copy()

    # per-engine time_idx (0..T-1) — TFT requires strictly increasing ints
    out = out.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    out["time_idx"] = out.groupby("engine_id").cumcount()

    # RUL cap
    out["RUL"] = out["RUL"].clip(upper=RUL_CAP).astype(np.float32)

    # categoricals as strings
    out["engine_id"]  = out["engine_id"].astype(str)
    out["dataset_id"] = out["dataset_id"].astype(str)
    out["op_cluster"] = out["op_cluster"].astype(str)

    return out


# ─────────────────────────────────────────────
# 2. DATASET CONSTRUCTION
# ─────────────────────────────────────────────

def build_training_dataset(
    train_df: pd.DataFrame,
    sensor_cols: list[str] = SENSOR_COLS,
    op_cols:     list[str] = OP_COLS,
    max_encoder_length:    int = MAX_ENCODER_LENGTH,
    max_prediction_length: int = MAX_PREDICTION_LENGTH,
) -> TimeSeriesDataSet:
    """Build the training ``TimeSeriesDataSet`` over all windows in ``train_df``."""
    return TimeSeriesDataSet(
        train_df,
        time_idx           = "time_idx",
        target             = "RUL",
        group_ids          = ["engine_id"],
        min_encoder_length = max_encoder_length // 2,  # allow shorter histories
        max_encoder_length = max_encoder_length,
        min_prediction_length = max_prediction_length,
        max_prediction_length = max_prediction_length,

        static_categoricals        = ["dataset_id", "op_cluster"],
        time_varying_known_reals   = ["time_idx"],   # cycle index — known in advance
        time_varying_unknown_reals = [*sensor_cols, *op_cols],

        target_normalizer = GroupNormalizer(groups=["engine_id"], transformation="softplus"),

        # allow unseen engine_id / dataset_id / op_cluster levels at inference
        categorical_encoders = {
            "engine_id":  NaNLabelEncoder(add_nan=True),
            "dataset_id": NaNLabelEncoder(add_nan=True),
            "op_cluster": NaNLabelEncoder(add_nan=True),
        },

        add_relative_time_idx        = True,
        add_target_scales            = True,
        add_encoder_length           = True,
        allow_missing_timesteps      = False,
    )


def build_validation_dataset(
    training_ds: TimeSeriesDataSet,
    val_df:      pd.DataFrame,
) -> TimeSeriesDataSet:
    """Validation dataset re-uses the training encoders/normalizers."""
    return TimeSeriesDataSet.from_dataset(training_ds, val_df, stop_randomization=True)


def build_test_last_window_dataset(
    training_ds: TimeSeriesDataSet,
    test_df:     pd.DataFrame,
) -> TimeSeriesDataSet:
    """
    Test dataset that yields exactly ONE sample per engine — its last
    ``max_prediction_length`` cycles. This matches the CMAPSS evaluation
    protocol: one RUL prediction per test engine.
    """
    return TimeSeriesDataSet.from_dataset(
        training_ds,
        test_df,
        stop_randomization = True,
        predict            = True,  # keyword: last window per group
    )


# ─────────────────────────────────────────────
# 3. MODEL + TRAINING
# ─────────────────────────────────────────────

def build_tft_model(
    training_ds: TimeSeriesDataSet,
    cfg:         TFTConfig = TFTConfig(),
    use_quantile_loss: bool = True,
) -> TemporalFusionTransformer:
    """
    Instantiate TFT. With ``use_quantile_loss=True`` (default) the model
    produces 7 quantiles per step — the median is used as the point forecast
    and the quantiles give a free uncertainty band.
    """
    loss = QuantileLoss() if use_quantile_loss else RMSE()

    return TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate        = cfg.lr,
        hidden_size          = cfg.hidden_size,
        attention_head_size  = cfg.attention_heads,
        dropout              = cfg.dropout,
        hidden_continuous_size = cfg.hidden_continuous,
        loss                 = loss,
        log_interval         = 0,
        reduce_on_plateau_patience = 2,
        optimizer            = "adam",
    )


def train_tft(
    model:       TemporalFusionTransformer,
    train_loader,
    val_loader,
    cfg:         TFTConfig = TFTConfig(),
    ckpt_dir:    str | Path | None = None,
) -> pl.Trainer:
    """Fit TFT with early stopping + LR monitoring. Returns the trainer."""
    pl.seed_everything(cfg.seed, workers=True)

    callbacks: list[Any] = [
        EarlyStopping(monitor="val_loss", patience=cfg.patience, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs          = cfg.max_epochs,
        accelerator         = cfg.accelerator,
        devices             = 1,
        gradient_clip_val   = cfg.gradient_clip_val,
        callbacks           = callbacks,
        default_root_dir    = str(ckpt_dir) if ckpt_dir else None,
        enable_progress_bar = True,
        log_every_n_steps   = 25,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return trainer


# ─────────────────────────────────────────────
# 4. INFERENCE
# ─────────────────────────────────────────────

def predict_last_window(
    model:      TemporalFusionTransformer,
    test_ds:    TimeSeriesDataSet,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce one RUL prediction per engine (the last cycle of each trajectory).

    Returns ``(y_pred, engine_ids, dataset_ids)`` as numpy arrays.
    The median (0.5 quantile) is used as the point forecast when the loss
    is QuantileLoss; otherwise the raw scalar prediction.
    """
    loader = test_ds.to_dataloader(train=False, batch_size=batch_size, shuffle=False)

    # ``predictions`` returns (n_samples, prediction_length) for point loss,
    # or (n_samples, prediction_length, n_quantiles) for QuantileLoss.
    raw = model.predict(loader, mode="prediction", return_index=True)
    preds  = raw.output.cpu().numpy()
    index  = raw.index  # DataFrame with engine_id + time_idx
    if preds.ndim == 3:            # quantiles axis
        preds = preds[..., preds.shape[-1] // 2]  # median
    preds = preds.squeeze(-1).astype(np.float32)  # (n_engines,)

    # engine id + dataset id in the same order as preds
    engine_ids = index["engine_id"].astype(int).to_numpy()

    return preds, engine_ids


# ─────────────────────────────────────────────
# 5. INTERPRETABILITY (attention + variable importance)
# ─────────────────────────────────────────────

def interpret_tft(
    model:   TemporalFusionTransformer,
    dataset: TimeSeriesDataSet,
    batch_size: int = 128,
) -> dict[str, np.ndarray | list[str]]:
    """
    Run TFT's built-in interpretation on ``dataset``.

    Returns a dict with:
        * ``attention``             (encoder_length,)                 mean attention weights per encoder lag
        * ``static_variables``      (n_static_features,)              mean importance of each static input
        * ``encoder_variables``     (n_encoder_features,)             mean importance over the encoder sequence
        * ``decoder_variables``     (n_decoder_features,)             mean importance over the decoder horizon
        * ``*_names``               column names matching each array
    """
    loader = dataset.to_dataloader(train=False, batch_size=batch_size, shuffle=False)
    raw    = model.predict(loader, mode="raw", return_x=True)
    interp = model.interpret_output(raw.output, reduction="mean")

    def _to_np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    return {
        "attention":          _to_np(interp["attention"]),
        "static_variables":   _to_np(interp["static_variables"]),
        "encoder_variables":  _to_np(interp["encoder_variables"]),
        "decoder_variables":  _to_np(interp["decoder_variables"]),
        "static_names":       list(model.static_variables),
        "encoder_names":      list(model.encoder_variables),
        "decoder_names":      list(model.decoder_variables),
    }
