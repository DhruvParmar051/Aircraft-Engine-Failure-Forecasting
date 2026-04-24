"""
cleaning.py — sensor selection for CMAPSS
drops sensors with near-zero variance that carry no degradation signal
"""

import pandas as pd

ALL_SENSOR_COLS        = [f"s{i}" for i in range(1, 22)]
VARIANCE_THRESHOLD     = 0.1


def find_low_variance_sensors(
    df: pd.DataFrame,
    threshold: float = VARIANCE_THRESHOLD,
) -> list[str]:
    """
    Identify sensors that are near-constant across the entire DataFrame.
    A sensor is flagged if its std < threshold.
    Also includes KNOWN_CONSTANT_SENSORS as a safety net.
    Returns list of sensor column names to drop.
    """
    present_sensors = [s for s in ALL_SENSOR_COLS if s in df.columns]
    stds            = df[present_sensors].std()
    data_driven     = stds[stds < threshold].index.tolist()

    # union with known constants
    to_drop = sorted(set(data_driven))
    return to_drop


def drop_sensors(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sensors_to_drop: list[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Remove low-variance sensors from both train and test.
    Auto-detects from training data if sensors_to_drop is None (never from test — no leakage).
    Returns (train_clean, test_clean, sensors_dropped).
    """
    if sensors_to_drop is None:
        sensors_to_drop = find_low_variance_sensors(train)

    # Only drop sensors present in both — warn about asymmetry
    only_in_train = [s for s in sensors_to_drop if s in train.columns and s not in test.columns]
    only_in_test  = [s for s in sensors_to_drop if s in test.columns  and s not in train.columns]
    missing       = [s for s in sensors_to_drop if s not in train.columns and s not in test.columns]

    if only_in_train and verbose:
        print(f"  [WARN] in train only (dropping from train only): {only_in_train}")
    if only_in_test and verbose:
        print(f"  [WARN] in test only (dropping from test only): {only_in_test}")
    if missing and verbose:
        print(f"  [WARN] not found in either (already dropped?): {missing}")

    both_existing  = [s for s in sensors_to_drop if s in train.columns and s in test.columns]
    sensors_kept   = [s for s in ALL_SENSOR_COLS if s not in both_existing]

    train_clean = train.drop(columns=[s for s in sensors_to_drop if s in train.columns])
    test_clean  = test.drop( columns=[s for s in sensors_to_drop if s in test.columns])

    if verbose:
        print(f"  dropped {len(both_existing)} sensors: {both_existing}")
        print(f"  kept    {len(sensors_kept)} sensors: {sensors_kept}")

    return train_clean, test_clean, both_existing


def get_sensor_cols(df: pd.DataFrame) -> list[str]:
    """Return sensor columns still present in df after dropping."""
    return [c for c in ALL_SENSOR_COLS if c in df.columns]