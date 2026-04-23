"""
deep_learning.py
================
Shared constants, dataset utilities, training loop, evaluation helpers,
and plotting functions used by all four DL model notebooks:
    T11a_RNN.ipynb
    T11b_LSTM.ipynb
    T11c_GRU.ipynb
    T11d_Transformer.ipynb

Import in each notebook with:
    from deep_learning import *
"""

import sys, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Project root (two levels above this file) ─────────────────────────────────
ROOT = Path(os.getcwd()).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE  = 30        # cycles per sliding-window sample
RUL_CAP      = 125       # cap RUL at this value (same as classical models)
BATCH_SIZE   = 128
EPOCHS       = 50
LR           = 1e-3
RANDOM_SEED  = 42
PATIENCE     = 10        # early-stopping patience (epochs)

PROC_DIR     = ROOT / "data" / "processed"

# 14 informative sensor columns (standard CMAPSS selection)
SENSOR_COLS  = [f"s{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load processed train / test feature CSVs.

    Returns
    -------
    train_df, test_df : pd.DataFrame
    """
    train_df = pd.read_csv(PROC_DIR / "train_features.csv")
    test_df  = pd.read_csv(PROC_DIR / "test_features.csv")
    print(f"Train shape : {train_df.shape}  ({train_df['engine_id'].nunique()} engines)")
    print(f"Test  shape : {test_df.shape}   ({test_df['engine_id'].nunique()} engines)")
    return train_df, test_df


def select_features(train_df: pd.DataFrame) -> list[str]:
    """
    Prefer rolling-mean sensor features when available (richer signal);
    fall back to raw sensor columns otherwise.

    Returns
    -------
    feat_cols : list[str]
    """
    rmean_cols = [c for c in train_df.columns if "_rmean_" in c]
    feat_cols  = rmean_cols if len(rmean_cols) >= 10 else SENSOR_COLS
    print(f"Feature columns ({len(feat_cols)}): {feat_cols[:5]} ...")
    return feat_cols


def build_windows(
    df: pd.DataFrame,
    feat_cols: list[str],
    window_size: int = WINDOW_SIZE,
    is_test: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window samples from per-engine time series.

    Train mode : ALL windows from every engine.
    Test  mode : LAST window only per engine (one prediction per engine).

    Short engines are front-padded with the first row to reach `window_size`.

    Parameters
    ----------
    df          : DataFrame with columns engine_id, cycle, RUL, + feat_cols
    feat_cols   : list of feature column names
    window_size : number of cycles per window
    is_test     : if True, emit only the last window per engine

    Returns
    -------
    X : (n_samples, window_size, n_features)  float32
    y : (n_samples,)                          float32  RUL at window end
    """
    X_list, y_list = [], []

    for _, grp in df.groupby("engine_id"):
        grp    = grp.sort_values("cycle")
        feats  = grp[feat_cols].values.astype(np.float32)   # (T, F)
        labels = grp["RUL"].values.astype(np.float32)       # (T,)
        T      = len(feats)

        # Pad engines shorter than the window
        if T < window_size:
            pad    = np.tile(feats[0], (window_size - T, 1))
            feats  = np.vstack([pad, feats])
            labels = np.concatenate(
                [np.full(window_size - T, labels[0]), labels]
            )
            T = window_size

        if is_test:
            X_list.append(feats[-window_size:])   # (W, F)
            y_list.append(labels[-1])             # scalar
        else:
            for i in range(window_size, T + 1):
                X_list.append(feats[i - window_size: i])  # (W, F)
                y_list.append(labels[i - 1])              # RUL at end

    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y


def engine_split(
    train_df: pd.DataFrame,
    feat_cols: list[str],
    val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split by engine_id (80/20) to avoid within-engine data leakage.

    Returns
    -------
    X_train, y_train, X_val, y_val
    """
    all_eids = train_df["engine_id"].unique()
    np.random.shuffle(all_eids)
    split_idx  = int(len(all_eids) * (1 - val_ratio))
    train_eids = set(all_eids[:split_idx])
    val_eids   = set(all_eids[split_idx:])
    print(f"Train engines: {len(train_eids)}  Val engines: {len(val_eids)}")

    train_sub = train_df[train_df["engine_id"].isin(train_eids)]
    val_sub   = train_df[train_df["engine_id"].isin(val_eids)]

    X_train, y_train = build_windows(train_sub, feat_cols, is_test=False)
    X_val,   y_val   = build_windows(val_sub,   feat_cols, is_test=True)
    print(f"X_train: {X_train.shape}  X_val: {X_val.shape}")
    return X_train, y_train, X_val, y_val


# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════

class RULDataset(Dataset):
    """Wraps numpy (X, y) arrays into a PyTorch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(
    X_train, y_train, X_val, y_val, X_test, y_test,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap split arrays into DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        RULDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader   = DataLoader(
        RULDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False
    )
    test_loader  = DataLoader(
        RULDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False
    )
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LR,
    model_name: str = "model",
    patience: int = PATIENCE,
) -> tuple[nn.Module, list[float], list[float]]:
    """
    Train *model* with MSE loss, Adam, LR-on-plateau, and early stopping.

    Logic
    -----
    1. Adam + ReduceLROnPlateau (halves LR after 5 stagnant epochs).
    2. Each batch: forward → MSE → backward → grad-clip (max_norm=1) → step.
    3. After each epoch: validate, update scheduler, check best/early-stop.
    4. Restores the best-val-loss weights before returning.

    Returns
    -------
    model        : nn.Module with best weights loaded
    train_losses : list of per-epoch train MSE
    val_losses   : list of per-epoch val MSE
    """
    model     = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights  = None
    no_improve    = 0
    train_losses  = []
    val_losses    = []

    for epoch in range(1, epochs + 1):

        # ── Training ──────────────────────────────────────────────────
        model.train()
        batch_losses = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_b)
            loss  = criterion(preds, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = float(np.mean(batch_losses))

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                preds = model(X_b)
                val_batch_losses.append(criterion(preds, y_b).item())
        val_loss = float(np.mean(val_batch_losses))

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ── Best / early-stop ─────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(
                f"  [{model_name}] Epoch {epoch:3d} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"best={best_val_loss:.4f}"
            )

        if no_improve >= patience:
            print(f"  [{model_name}] Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_weights)
    return model, train_losses, val_losses


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def predict_test(
    model: nn.Module,
    test_loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on the test set.

    Predictions are clipped to [0, RUL_CAP] — consistent with classical models.

    Returns
    -------
    y_true, y_pred : np.ndarray, one entry per engine
    """
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds = model(X_b.to(DEVICE))
            preds = torch.clamp(preds, 0, RUL_CAP)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y_b.numpy())
    return np.array(all_true), np.array(all_preds)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    model_name: str = "Model",
) -> None:
    """Plot train vs. validation MSE loss curves."""
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(train_losses, label="Train loss", color="steelblue")
    ax.plot(val_losses,   label="Val loss",   color="orange", ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{model_name} — Training Curve")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> None:
    """Scatter plot + sorted-sequence comparison of true vs. predicted RUL."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Scatter: predicted vs actual
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.4, color="steelblue", s=15)
    ax.plot([0, RUL_CAP], [0, RUL_CAP], "r--", lw=1.5, label="perfect")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(f"{model_name} — Predicted vs Actual")
    ax.legend()
    ax.set_xlim(0, RUL_CAP + 5)
    ax.set_ylim(0, RUL_CAP + 5)

    # Sorted sequence
    ax   = axes[1]
    sidx = np.argsort(y_true)
    smooth = (
        pd.Series(y_pred[sidx])
        .rolling(10, center=True, min_periods=1)
        .mean()
        .values
    )
    ax.plot(y_true[sidx], color="steelblue", label="True RUL",            lw=1.5)
    ax.plot(smooth,       color="orange",    label="Predicted (smoothed)", lw=1.5)
    ax.set_xlabel("Engine (sorted by true RUL)")
    ax.set_ylabel("RUL")
    ax.set_title(f"{model_name} — Sorted Predictions")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_comparison(combined: dict) -> None:
    """
    Bar chart comparing RMSE and NASA-score-mean across all models.

    Parameters
    ----------
    combined : {model_name: {'rmse': float, 'nasa_score_mean': float, ...}}
                First N classical models coloured steelblue, rest darkorange.
    """
    models    = list(combined.keys())
    rmse_vals = [combined[m]["rmse"]            for m in models]
    nasa_vals = [combined[m]["nasa_score_mean"] for m in models]

    # Detect how many classical vs DL models
    n_classical = sum(1 for m in models if m in ("AR", "ARMA", "ARIMA"))
    colors = ["steelblue"] * n_classical + ["darkorange"] * (len(models) - n_classical)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, vals, ylabel, title in zip(
        axes,
        [rmse_vals, nasa_vals],
        ["RMSE (lower is better)", "NASA Score Mean (lower is better)"],
        ["RMSE — Classical vs Deep Learning", "NASA Score — Classical vs Deep Learning"],
    ):
        bars = ax.bar(models, vals, color=colors, edgecolor="white")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=9,
            )

    axes[0].legend(handles=[
        Patch(color="steelblue",  label="Classical"),
        Patch(color="darkorange", label="Deep Learning"),
    ])

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.show()
    print("Saved: model_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK (run: python deep_learning.py)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Device : {DEVICE}")
    print(f"ROOT   : {ROOT}")
    print(f"PROC_DIR exists: {PROC_DIR.exists()}")
    print("Constants and utilities loaded successfully.")