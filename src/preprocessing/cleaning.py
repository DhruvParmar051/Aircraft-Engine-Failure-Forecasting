"""
cleaning.py — sensor selection for CMAPSS
drops sensors with near-zero variance that carry no degradation signal
uses per-dataset_id variance check so sensors useful in FD002/FD004
are not discarded because they're flat in FD001
"""

import pandas as pd

KNOWN_CONSTANT_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
ALL_SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
VARIANCE_THRESHOLD = 0.1


def find_low_variance_sensors(
    df: pd.DataFrame,
    threshold: float = VARIANCE_THRESHOLD,
) -> list[str]:
    """
    identify sensors that are near-constant in ALL dataset subsets
    a sensor must be flat across every subset to be flagged — not just one
    returns list of sensor column names to drop
    """
    if "dataset_id" not in df.columns:
        raise ValueError("DataFrame must have 'dataset_id' column for per-subset check")

    per_subset_std = df.groupby("dataset_id")[ALL_SENSOR_COLS].std()
    flat_in_all = (per_subset_std < threshold).all(axis=0)
    data_driven = flat_in_all[flat_in_all].index.tolist()

    # union with known constants as a safety net
    to_drop = sorted(set(data_driven) | set(KNOWN_CONSTANT_SENSORS))
    return to_drop


def drop_sensors(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sensors_to_drop: list[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    remove low-variance sensors from both train and test
    if sensors_to_drop is None, auto-detect from training data (never from test — no leakage)
    returns (train_clean, test_clean, sensors_dropped)

    BUG FIX: existence check now uses the intersection of train and test columns
    previously checked train.columns only, then applied to test — if train/test had
    different columns (e.g. after a bad manual edit), test.drop would raise KeyError
    """
    if sensors_to_drop is None:
        sensors_to_drop = find_low_variance_sensors(train)

    both_cols = set(train.columns) & set(test.columns)
    existing = [s for s in sensors_to_drop if s in both_cols]

    # warn about any asymmetry between train and test column sets
    only_in_train = [s for s in sensors_to_drop if s in train.columns and s not in test.columns]
    only_in_test  = [s for s in sensors_to_drop if s in test.columns and s not in train.columns]
    if only_in_train and verbose:
        print(f"  [WARN] sensors in train but not test (dropping from train only): {only_in_train}")
    if only_in_test and verbose:
        print(f"  [WARN] sensors in test but not train (dropping from test only): {only_in_test}")

    missing = [s for s in sensors_to_drop if s not in train.columns and s not in test.columns]
    if missing and verbose:
        print(f"  [WARN] sensors not found in either DataFrame (already dropped?): {missing}")

    sensors_kept = [s for s in ALL_SENSOR_COLS if s not in existing]

    train_clean = train.drop(columns=[s for s in sensors_to_drop if s in train.columns])
    test_clean  = test.drop(columns=[s for s in sensors_to_drop if s in test.columns])

    if verbose:
        print(f"  dropped {len(existing)} sensors (from both): {existing}")
        print(f"  kept   {len(sensors_kept)} sensors: {sensors_kept}")

    return train_clean, test_clean, existing


def get_sensor_cols(df: pd.DataFrame) -> list[str]:
    """return sensor columns present in df (s1–s21 that still exist after dropping)"""
    return [c for c in ALL_SENSOR_COLS if c in df.columns]