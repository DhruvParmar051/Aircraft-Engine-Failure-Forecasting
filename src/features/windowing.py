"""
windowing.py — sliding window sequences for sequence models (LSTM, GRU, RNN, TCN, TFT)
shared utility so all team members produce identical 3D tensors and evaluation is comparable
a window at position t covers cycles [t-W+1 … t]; target is RUL at cycle t

design decisions:
- windows never cross engine boundaries
- create_windows: training — engines shorter than window_size are skipped with a warning
- create_last_window_per_engine: evaluation — short engines are zero-padded (never skipped)
- split_by_engine: random engine-level split (no dataset_id stratification — single dataset)
- output shape: X=(n_samples, window_size, n_features), y=(n_samples,)
"""

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 30


def create_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = DEFAULT_WINDOW,
    group_col: str = "engine_id",
    target_col: str = "RUL",
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding window sequences grouped by engine.
    Windows never cross engine boundaries.

    Returns:
        X          : (n_samples, window_size, n_features) float32
        y          : (n_samples,) float32 — RUL at last cycle of each window
        engine_ids : (n_samples,) int64   — source engine for train/val splitting
    """
    df = df.sort_values([group_col, "cycle"]).reset_index(drop=True)

    X_list, y_list, id_list = [], [], []
    skipped = []

    for engine_id, group in df.groupby(group_col, sort=False):
        n = len(group)

        if n < window_size:
            skipped.append((engine_id, n))
            continue

        features = group[feature_cols].values.astype(np.float32)
        targets  = group[target_col].values.astype(np.float32)

        for end in range(window_size - 1, n, step):
            start = end - window_size + 1
            X_list.append(features[start: end + 1])
            y_list.append(targets[end])
            id_list.append(engine_id)

    if skipped:
        print(f"  [WARN] {len(skipped)} engines skipped (shorter than window={window_size}): "
              f"{[e for e, _ in skipped[:5]]}{'...' if len(skipped) > 5 else ''}")

    if not X_list:
        raise ValueError(
            f"No valid windows produced. All engines have fewer than {window_size} cycles. "
            "Reduce window_size or check your data."
        )

    X          = np.stack(X_list, axis=0)
    y          = np.array(y_list,  dtype=np.float32)
    engine_ids = np.array(id_list, dtype=np.int64)

    print(f"  windows: X={X.shape}, y={y.shape} "
          f"({df[group_col].nunique()} engines, {len(skipped)} skipped)")
    return X, y, engine_ids


def create_last_window_per_engine(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = DEFAULT_WINDOW,
    group_col: str = "engine_id",
    target_col: str = "RUL",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract only the LAST window per engine — used at evaluation time.
    Engines shorter than window_size are zero-padded (never skipped).

    Returns X, y, engine_ids — one row per engine.
    """
    df = df.sort_values([group_col, "cycle"]).reset_index(drop=True)

    X_list, y_list, id_list = [], [], []
    padded_engines = []

    for engine_id, group in df.groupby(group_col, sort=False):
        n = len(group)

        if n < window_size:
            padded_engines.append((engine_id, n))
            pad_len  = window_size - n
            features = group[feature_cols].values.astype(np.float32)
            padded   = np.zeros((window_size, len(feature_cols)), dtype=np.float32)
            padded[pad_len:] = features
            X_list.append(padded)
        else:
            X_list.append(group[feature_cols].values[-window_size:].astype(np.float32))

        # cycle-based last row — not positional
        last_idx = group["cycle"].idxmax()
        y_list.append(float(group.loc[last_idx, target_col]))
        id_list.append(engine_id)

    if padded_engines:
        print(f"  [WARN] {len(padded_engines)} engines zero-padded (shorter than window={window_size}): "
              f"{[e for e, _ in padded_engines[:5]]}")

    X          = np.stack(X_list, axis=0)
    y          = np.array(y_list,  dtype=np.float32)
    engine_ids = np.array(id_list, dtype=np.int64)

    print(f"  last-window eval: X={X.shape}, y={y.shape}")
    return X, y, engine_ids


def split_by_engine(
    X: np.ndarray,
    y: np.ndarray,
    engine_ids: np.ndarray,
    val_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Random engine-level train/validation split.
    Splits on engine_ids so windows from the same engine never span both splits.

    Returns X_train, X_val, y_train, y_val.
    """
    rng            = np.random.default_rng(random_seed)
    unique_engines = np.unique(engine_ids)
    rng.shuffle(unique_engines)

    n_val          = max(1, int(len(unique_engines) * val_fraction))
    val_engines    = set(unique_engines[:n_val])

    val_mask   = np.isin(engine_ids, list(val_engines))
    train_mask = ~val_mask

    n_train = len(np.unique(engine_ids[train_mask]))
    n_val   = len(np.unique(engine_ids[val_mask]))
    print(f"  train/val split: "
          f"{train_mask.sum()} train samples ({n_train} engines) | "
          f"{val_mask.sum()} val samples ({n_val} engines)")

    return X[train_mask], X[val_mask], y[train_mask], y[val_mask]


def get_arima_splits(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    random_seed: int = 42,
    group_col: str = "engine_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train/validation split on the raw DataFrame for classical models (AR, ARMA, ARIMA).
    Splits by engine_id — returns (train_df, val_df) as DataFrames, not windowed arrays.
    """
    rng     = np.random.default_rng(random_seed)
    engines = np.array(sorted(df[group_col].unique()))
    rng.shuffle(engines)

    n_val       = max(1, int(len(engines) * val_fraction))
    val_engines = set(engines[:n_val].tolist())

    val_mask = df[group_col].isin(val_engines)
    train_df = df[~val_mask].copy()
    val_df   = df[val_mask].copy()

    print(f"  train/val: {train_df[group_col].nunique()} train engines, "
          f"{val_df[group_col].nunique()} val engines")
    return train_df, val_df