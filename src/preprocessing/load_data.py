"""
load_data.py — raw CMAPSS text files → unified train/test DataFrames
handles all 4 subsets (FD001–FD004), globally unique engine IDs

BUG FIX (engine_id collision): train and test use independent offset counters
starting from 0. This means train engine #1 and test engine #1 share the same
integer ID. This is intentional — they are in separate DataFrames and never
merged — but is documented explicitly to prevent accidental cross-DataFrame joins.
If you ever combine train+test into a single DataFrame (e.g. for semi-supervised
learning), offset test engine IDs by train["engine_id"].max() first.
"""

import pandas as pd
from pathlib import Path

# 26-column schema: 2 ids + 3 op settings + 21 sensors
COLS = ["engine_id", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

SUBSET_IDS = [1, 2, 3, 4]


def load_single_train(path: Path, dataset_id: int, engine_id_offset: int) -> pd.DataFrame:
    """load one FD00X train file, offset engine IDs to be globally unique within train set"""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    df["dataset_id"] = dataset_id
    df["engine_id"] = df["engine_id"] + engine_id_offset
    return df


def load_single_test(
    test_path: Path,
    rul_path: Path,
    dataset_id: int,
    engine_id_offset: int,
) -> pd.DataFrame:
    """
    load one FD00X test file + its RUL ground truth file
    rul_last stored as a column — consumed by compute_test_rul() in rul.py
    engine IDs are unique within the test set (independent from train IDs — see module docstring)
    """
    df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLS)
    rul_series = pd.read_csv(rul_path, header=None, names=["rul_last"])

    df["dataset_id"] = dataset_id
    df["engine_id"] = df["engine_id"] + engine_id_offset

    # map rul_last to each engine in sorted ID order
    unique_engines = sorted(df["engine_id"].unique())
    if len(unique_engines) != len(rul_series):
        raise ValueError(
            f"FD00{dataset_id}: engine count ({len(unique_engines)}) "
            f"!= RUL entries ({len(rul_series)})"
        )
    rul_map = pd.DataFrame({
        "engine_id": unique_engines,
        "rul_last": rul_series["rul_last"].values,
    })
    df = df.merge(rul_map, on="engine_id")
    return df


def load_all_train(data_dir: str | Path) -> pd.DataFrame:
    """
    combine all 4 FD train files into a single DataFrame
    engine IDs are globally unique within this DataFrame
    NOTE: engine IDs start from 1 and are independent from test engine IDs
    returns raw combined data — RUL not yet computed (done in T02)
    """
    data_dir = Path(data_dir)
    frames = []
    offset = 0
    for sid in SUBSET_IDS:
        path = data_dir / f"train_FD00{sid}.txt"
        df = load_single_train(path, dataset_id=sid, engine_id_offset=offset)
        offset = int(df["engine_id"].max())
        frames.append(df)
        print(f"  FD00{sid}: {df['engine_id'].nunique()} engines, {len(df)} rows "
              f"[engine_id {df['engine_id'].min()}–{df['engine_id'].max()}]")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  combined train: {combined.shape}, engines: {combined['engine_id'].nunique()}")
    return combined


def load_all_test(data_dir: str | Path) -> pd.DataFrame:
    """
    combine all 4 FD test files + RUL ground truth
    returns raw combined data with rul_last column — consumed and dropped by T02
    NOTE: test engine IDs are independent from train IDs (see module docstring)
    """
    data_dir = Path(data_dir)
    frames = []
    offset = 0
    for sid in SUBSET_IDS:
        test_path = data_dir / f"test_FD00{sid}.txt"
        rul_path = data_dir / f"RUL_FD00{sid}.txt"
        df = load_single_test(test_path, rul_path, dataset_id=sid, engine_id_offset=offset)
        offset = int(df["engine_id"].max())
        frames.append(df)
        print(f"  FD00{sid}: {df['engine_id'].nunique()} engines, {len(df)} rows "
              f"[engine_id {df['engine_id'].min()}–{df['engine_id'].max()}]")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  combined test: {combined.shape}, engines: {combined['engine_id'].nunique()}")
    return combined