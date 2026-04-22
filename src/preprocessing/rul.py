"""
rul.py — Remaining Useful Life computation for CMAPSS train and test sets
train: RUL = max_cycle - cycle  (failure is the last observed cycle)
test:  RUL = rul_last + (max_cycle - cycle)  (truncated series)
both:  RUL capped at RUL_CAP after computation
"""

import pandas as pd

RUL_CAP = 125  # standard academic cap; focuses model on critical degradation window


def compute_train_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Add RUL column to training data.
    Last cycle per engine = failure point → RUL = 0 there.
    Sorts by (engine_id, cycle) before computing to guarantee correct ordering.
    """
    df = df.copy()
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    max_cycles = (
        df.groupby("engine_id")["cycle"]
        .max()
        .rename("max_cycle")
        .reset_index()
    )
    df           = df.merge(max_cycles, on="engine_id")
    df["RUL"]    = (df["max_cycle"] - df["cycle"]).clip(upper=cap)
    df.drop(columns="max_cycle", inplace=True)
    return df


def compute_test_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Add RUL column to test data.
    Test series are truncated — rul_last is the true RUL at the final observed cycle.
    RUL at earlier cycles = rul_last + (last_observed_cycle - current_cycle).
    Requires 'rul_last' column from load_test() in load_data.py.
    """
    if "rul_last" not in df.columns:
        raise ValueError("test DataFrame must contain 'rul_last' column (from load_test)")

    df = df.copy()
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    max_cycles = (
        df.groupby("engine_id")["cycle"]
        .max()
        .rename("max_cycle")
        .reset_index()
    )
    df           = df.merge(max_cycles, on="engine_id")
    df["RUL"]    = (df["rul_last"] + (df["max_cycle"] - df["cycle"])).clip(upper=cap)
    df.drop(columns=["max_cycle", "rul_last"], inplace=True)
    return df


def verify_train_rul(df: pd.DataFrame) -> None:
    """
    Sanity checks after train RUL computation.
    Uses cycle-sorted last row per engine — not positional last() — to avoid ordering bugs.
    """
    last_rul = (
        df.sort_values(["engine_id", "cycle"])
        .groupby("engine_id")["RUL"]
        .last()
    )
    assert (last_rul == 0).all(), (
        f"Not all engines end at RUL=0. "
        f"Offending engines: {last_rul[last_rul != 0].index.tolist()}"
    )
    assert df["RUL"].min() >= 0,          "Negative RUL found"
    assert df["RUL"].max() <= RUL_CAP,    f"RUL exceeds cap of {RUL_CAP}"
    print(f"  [PASS] train RUL: range [{df['RUL'].min()}, {df['RUL'].max()}], all engines end at 0")


def verify_test_rul(df: pd.DataFrame) -> None:
    """
    Sanity checks after test RUL computation.
    Confirms rul_last was removed (leakage risk if kept as a feature).
    """
    assert "rul_last" not in df.columns, (
        "rul_last still present — leakage risk if used as a model feature"
    )
    assert df["RUL"].min() >= 0,       "Negative RUL found in test"
    assert df["RUL"].max() <= RUL_CAP, f"Test RUL exceeds cap of {RUL_CAP}"
    print(f"  [PASS] test RUL: range [{df['RUL'].min()}, {df['RUL'].max()}]")
    print(f"  [PASS] rul_last absent — no leakage risk")