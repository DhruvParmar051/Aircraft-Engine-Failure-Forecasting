"""
load_data.py — raw CMAPSS text file → train/test DataFrames
loads a single FD00X subset — no multi-file merging, no dataset_id column
"""

import pandas as pd
from pathlib import Path

# 26-column schema: 2 ids + 3 op settings + 21 sensors
COLS = ["engine_id", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load_train(path: str | Path) -> pd.DataFrame:
    """
    Load a single CMAPSS train file (e.g. train_FD001.txt).
    Returns raw DataFrame — RUL not yet computed.
    """
    path = Path(path)
    df   = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    print(f"  train: {df['engine_id'].nunique()} engines, {len(df)} rows  [{path.name}]")
    return df


def load_test(test_path: str | Path, rul_path: str | Path) -> pd.DataFrame:
    """
    Load a single CMAPSS test file + its RUL ground truth file.
    Attaches rul_last column — consumed and dropped by compute_test_rul() in rul.py.
    """
    test_path = Path(test_path)
    rul_path  = Path(rul_path)

    df         = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLS)
    rul_series = pd.read_csv(rul_path,  header=None, names=["rul_last"])

    unique_engines = sorted(df["engine_id"].unique())
    if len(unique_engines) != len(rul_series):
        raise ValueError(
            f"Engine count ({len(unique_engines)}) != RUL entries ({len(rul_series)})"
        )

    # map rul_last to each engine in sorted ID order
    rul_map = pd.DataFrame({
        "engine_id": unique_engines,
        "rul_last":  rul_series["rul_last"].values,
    })
    df = df.merge(rul_map, on="engine_id")
    print(f"  test: {df['engine_id'].nunique()} engines, {len(df)} rows  [{test_path.name}]")
    return df