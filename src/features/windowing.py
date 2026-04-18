"""
windowing.py — sliding window sequences for sequence models (LSTM, GRU, RNN, TCN, TFT)
shared utility so all team members produce identical 3D tensors and evaluation is comparable
a window at position t covers cycles [t-W+1 … t]; target is RUL at cycle t

design decisions:
- windows never cross engine boundaries
- create_windows: training — engines shorter than window_size are skipped with a warning
- create_last_window_per_engine: evaluation — short engines are zero-padded (never skipped)
- split_by_engine: stratified by dataset_id so FD001–FD004 are proportionally represented
- output shape: X=(n_samples, window_size, n_features), y=(n_samples,)
"""

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 30  # cycles; matches shared convention in task_allocation.md


def create_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = DEFAULT_WINDOW,
    group_col: str = "engine_id",
    target_col: str = "RUL",
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    create sliding window sequences from a time-series DataFrame, grouped by engine
    windows never cross engine boundaries

    returns:
        X           : (n_samples, window_size, n_features) float32
        y           : (n_samples,) float32 — RUL at last cycle of each window
        engine_ids  : (n_samples,) int64   — source engine for train/val splitting
        dataset_ids : (n_samples,) int64   — source FD subset for per-subset evaluation

    IMPROVEMENT: now also returns dataset_ids so evaluate_per_subset can be called
    directly on windowed outputs without needing a separate lookup
    """
    if "dataset_id" not in df.columns:
        raise ValueError("DataFrame must contain 'dataset_id' column")

    df = df.sort_values([group_col, "cycle"]).reset_index(drop=True)

    X_list, y_list, id_list, did_list = [], [], [], []
    skipped = []

    for engine_id, group in df.groupby(group_col, sort=False):
        n = len(group)
        dataset_id = int(group["dataset_id"].iloc[0])

        if n < window_size:
            skipped.append((engine_id, n))
            continue

        features = group[feature_cols].values.astype(np.float32)
        targets  = group[target_col].values.astype(np.float32)

        for end in range(window_size - 1, n, step):
            start = end - window_size + 1
            X_list.append(features[start : end + 1])
            y_list.append(targets[end])
            id_list.append(engine_id)
            did_list.append(dataset_id)

    if skipped:
        print(f"  [WARN] {len(skipped)} engines skipped (shorter than window={window_size}): "
              f"{[e for e, _ in skipped[:5]]}{'...' if len(skipped) > 5 else ''}")

    if not X_list:
        raise ValueError(
            f"No valid windows produced. All engines have fewer than {window_size} cycles. "
            "Reduce window_size or check your data."
        )

    X           = np.stack(X_list, axis=0)
    y           = np.array(y_list,   dtype=np.float32)
    engine_ids  = np.array(id_list,  dtype=np.int64)
    dataset_ids = np.array(did_list, dtype=np.int64)

    print(f"  windows: X={X.shape}, y={y.shape} "
          f"({df[group_col].nunique()} engines, {len(skipped)} skipped)")
    return X, y, engine_ids, dataset_ids


def create_last_window_per_engine(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = DEFAULT_WINDOW,
    group_col: str = "engine_id",
    target_col: str = "RUL",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    extract only the LAST window per engine — used at evaluation time
    test engines shorter than window_size are zero-padded (never skipped —
    every test engine must be evaluated)

    BUG FIX: target now uses cycle-based idxmax instead of positional iloc[-1]
    returns X, y, engine_ids, dataset_ids (one row per engine)
    """
    if "dataset_id" not in df.columns:
        raise ValueError("DataFrame must contain 'dataset_id' column")

    df = df.sort_values([group_col, "cycle"]).reset_index(drop=True)

    X_list, y_list, id_list, did_list = [], [], [], []
    padded_engines = []

    for engine_id, group in df.groupby(group_col, sort=False):
        n = len(group)
        dataset_id = int(group["dataset_id"].iloc[0])

        if n < window_size:
            padded_engines.append((engine_id, n))
            pad_len  = window_size - n
            features = group[feature_cols].values.astype(np.float32)
            padded   = np.zeros((window_size, len(feature_cols)), dtype=np.float32)
            padded[pad_len:] = features
            X_list.append(padded)
        else:
            X_list.append(group[feature_cols].values[-window_size:].astype(np.float32))

        # use cycle-based last row — not positional (safe here since sorted, but explicit)
        last_idx = group["cycle"].idxmax()
        y_list.append(float(group.loc[last_idx, target_col]))
        id_list.append(engine_id)
        did_list.append(dataset_id)

    if padded_engines:
        print(f"  [WARN] {len(padded_engines)} engines zero-padded (shorter than window={window_size}): "
              f"{[e for e, _ in padded_engines[:5]]}")

    X           = np.stack(X_list, axis=0)
    y           = np.array(y_list,   dtype=np.float32)
    engine_ids  = np.array(id_list,  dtype=np.int64)
    dataset_ids = np.array(did_list, dtype=np.int64)

    print(f"  last-window eval: X={X.shape}, y={y.shape}")
    return X, y, engine_ids, dataset_ids


def split_by_engine(
    X: np.ndarray,
    y: np.ndarray,
    engine_ids: np.ndarray,
    dataset_ids: np.ndarray,
    val_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    train/validation split stratified by dataset_id
    splits within each FD subset proportionally — prevents random shuffle from
    accidentally under-representing FD001 (100 engines) vs FD004 (248 engines)

    BUG FIX: was a single shuffle across all engines — no stratification despite docstring claim
    now splits per dataset_id and combines, so each subset contributes proportionally to val

    returns X_train, X_val, y_train, y_val
    """
    rng = np.random.default_rng(random_seed)
    val_mask = np.zeros(len(engine_ids), dtype=bool)

    for did in np.unique(dataset_ids):
        subset_mask    = dataset_ids == did
        subset_engines = np.unique(engine_ids[subset_mask])
        rng.shuffle(subset_engines)

        n_val = max(1, int(len(subset_engines) * val_fraction))
        val_engines_subset = set(subset_engines[:n_val])

        # mark rows from val engines in this subset
        val_mask |= (subset_mask & np.isin(engine_ids, list(val_engines_subset)))

    train_mask = ~val_mask

    n_train_engines = len(np.unique(engine_ids[train_mask]))
    n_val_engines   = len(np.unique(engine_ids[val_mask]))
    print(f"  stratified train/val split: "
          f"{train_mask.sum()} train samples ({n_train_engines} engines) | "
          f"{val_mask.sum()} val samples ({n_val_engines} engines)")

    return X[train_mask], X[val_mask], y[train_mask], y[val_mask]


def get_arima_splits(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    random_seed: int = 42,
    group_col: str = "engine_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    train/validation split on the raw DataFrame for classical models (ARIMA, AR, ARMA)
    splits by engine_id stratified per dataset_id — same logic as split_by_engine
    returns (train_df, val_df) as DataFrames — NOT windowed arrays

    IMPROVEMENT: classical models need raw time-series DataFrames not numpy windows
    without this utility, ARIMA authors would write inline splits that may differ
    from the sequence-model split and produce incomparable validation sets
    """
    rng = np.random.default_rng(random_seed)
    val_engines: set[int] = set()

    for did, group_df in df.groupby("dataset_id"):
        engines = np.array(sorted(group_df[group_col].unique()))
        rng.shuffle(engines)
        n_val = max(1, int(len(engines) * val_fraction))
        val_engines.update(engines[:n_val].tolist())

    val_mask   = df[group_col].isin(val_engines)
    train_df   = df[~val_mask].copy()
    val_df     = df[val_mask].copy()

    print(f"  ARIMA train/val: {train_df[group_col].nunique()} train engines, "
          f"{val_df[group_col].nunique()} val engines (stratified per dataset)")
    return train_df, val_df