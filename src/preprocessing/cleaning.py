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


# ──────────────────────────────────────────────────────────────────────────────
# SENSOR OUTLIER CLIPPING
# ──────────────────────────────────────────────────────────────────────────────

def fit_outlier_bounds(
    train: pd.DataFrame,
    sensor_cols: list[str],
    n_iqr: float = 5.0,
    group_col: str = "op_cluster",
) -> dict[str, tuple[float, float]]:
    """
    Compute per-sensor IQR-based clip bounds from TRAIN data only.

    Why IQR over z-score:
        Sensor distributions are multi-modal across operating conditions even
        after per-cluster scaling. IQR is robust to multi-modality; z-score
        assumes unimodality and would clip valid values in minority clusters.

    Why n_iqr=5 (not 3):
        n_iqr=3 is commonly used for anomaly detection. For preprocessing we
        want to remove only gross sensor faults (stuck-at-max, dropout-to-zero)
        while preserving the degradation signal in extreme-but-valid readings.
        n_iqr=5 corresponds to values beyond 5× the IQR range — these are
        almost certainly sensor faults rather than real engine state.

    Why fit on train only:
        Fitting on test would leak information about the test distribution.
        If a test engine is genuinely more degraded than any training engine,
        its extreme sensor readings are valid signal and should NOT be clipped.
        Using training bounds is therefore the conservative, non-leaking choice.

    Parameters
    ----------
    group_col : column used to stratify IQR computation. Defaults to
                'op_cluster' (available after T04). Pass None to compute
                global bounds (acceptable for FD001/FD003, single-condition).

    Returns
    -------
    bounds : {sensor_name: (lower_clip, upper_clip)}
             Persist with joblib for inference.
    """
    bounds: dict[str, tuple[float, float]] = {}
    present = [s for s in sensor_cols if s in train.columns]

    if group_col and group_col in train.columns:
        # Stratified: compute IQR within each cluster, take union of bounds
        all_lower: dict[str, list[float]] = {s: [] for s in present}
        all_upper: dict[str, list[float]] = {s: [] for s in present}
        for _, grp in train.groupby(group_col):
            q1 = grp[present].quantile(0.25)
            q3 = grp[present].quantile(0.75)
            iqr = q3 - q1
            for s in present:
                all_lower[s].append(float(q1[s] - n_iqr * iqr[s]))
                all_upper[s].append(float(q3[s] + n_iqr * iqr[s]))
        for s in present:
            bounds[s] = (min(all_lower[s]), max(all_upper[s]))
    else:
        q1  = train[present].quantile(0.25)
        q3  = train[present].quantile(0.75)
        iqr = q3 - q1
        for s in present:
            bounds[s] = (
                float(q1[s] - n_iqr * iqr[s]),
                float(q3[s] + n_iqr * iqr[s]),
            )

    clipped_sensors = [s for s, (lo, hi) in bounds.items() if lo > train[s].min() or hi < train[s].max()]
    print(f"  [outlier bounds] {len(clipped_sensors)} sensors have values outside "
          f"±{n_iqr}×IQR: {clipped_sensors}")
    return bounds


def apply_outlier_bounds(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """
    Clip sensor values to pre-fitted bounds.
    Sensors not present in bounds are left unchanged.
    """
    df = df.copy()
    clipped_count = 0
    for sensor, (lo, hi) in bounds.items():
        if sensor not in df.columns:
            continue
        n_before = ((df[sensor] < lo) | (df[sensor] > hi)).sum()
        df[sensor] = df[sensor].clip(lower=lo, upper=hi)
        clipped_count += n_before
    if clipped_count:
        print(f"  [outlier clip] clipped {clipped_count} values across {len(bounds)} sensors")
    return df