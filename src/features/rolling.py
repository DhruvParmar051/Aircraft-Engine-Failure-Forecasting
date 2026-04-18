"""
rolling.py — rolling statistics features for CMAPSS time-series per engine
single sensor values are noisy; rolling mean reveals trend, rolling std reveals instability
all windows computed within each engine group (no cross-engine contamination)

BUG FIX: DataFrame must be sorted by (engine_id, cycle) before computing rolling features
groupby(sort=False).transform() respects the physical row order — if rows are not sorted
by cycle, rolling windows silently compute across non-consecutive cycles or across
engine boundaries after a merge reorders rows.
"""

import pandas as pd

DEFAULT_WINDOWS = [5, 10, 20]
# 5  = local noise filter (smooths cycle-to-cycle sensor noise)
# 10 = medium-term trend detection
# 20 = long-term drift — critical for sequence models with window_size=30:
#      without a w=20 feature, rolling stats at cycle positions 10-20 inside a
#      30-cycle window are computed from only 10 cycles of history, missing the
#      degradation trend entirely at those positions


def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    windows: list[int] = DEFAULT_WINDOWS,
    group_col: str = "engine_id",
) -> pd.DataFrame:
    """
    add rolling mean and std for each sensor over specified windows, grouped by engine
    sorts by (group_col, cycle) first — rolling requires consecutive cycle order
    min_periods=1 prevents NaN at early cycles where window is not yet full
    rolling std at first cycle is 0 (single observation — no variance yet)
    """
    # sort guarantees rolling windows are computed over consecutive cycles within each engine
    # without this, a merge or concat upstream could reorder rows and corrupt windows
    df = df.sort_values([group_col, "cycle"]).reset_index(drop=True)

    new_cols: dict[str, pd.Series] = {}
    for window in windows:
        for sensor in sensor_cols:
            grouped = df.groupby(group_col, sort=False)[sensor]
            new_cols[f"{sensor}_rmean_{window}"] = grouped.transform(
                lambda x, w=window: x.rolling(w, min_periods=1).mean()
            )
            new_cols[f"{sensor}_rstd_{window}"] = grouped.transform(
                lambda x, w=window: x.rolling(w, min_periods=1).std().fillna(0)
            )

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def get_rolling_col_names(sensor_cols: list[str], windows: list[int] = DEFAULT_WINDOWS) -> list[str]:
    """return the column names that add_rolling_features produces — useful for selecting features"""
    cols = []
    for window in windows:
        for sensor in sensor_cols:
            cols.append(f"{sensor}_rmean_{window}")
            cols.append(f"{sensor}_rstd_{window}")
    return cols


def verify_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    windows: list[int] = DEFAULT_WINDOWS,
    group_col: str = "engine_id",
) -> None:
    """
    verify rolling features:
    - all expected columns exist
    - no NaN values
    - rolling std is 0 at first cycle of each engine (sorted check, not positional)
    """
    expected = get_rolling_col_names(sensor_cols, windows)
    missing = [c for c in expected if c not in df.columns]
    assert not missing, f"Missing rolling feature columns: {missing}"
    print(f"  [PASS] all {len(expected)} rolling feature columns present")

    nan_count = df[expected].isnull().sum().sum()
    assert nan_count == 0, f"NaN values in rolling features: {nan_count}"
    print("  [PASS] no NaN values in rolling features")

    # find first cycle per engine by actual min cycle value, not positional
    first_cycle_idx = df.groupby(group_col)["cycle"].idxmin()
    rstd_cols = [c for c in expected if "_rstd_" in c]
    first_rstd = df.loc[first_cycle_idx, rstd_cols]
    assert (first_rstd == 0).all().all(), (
        "Rolling std is not 0 at first cycle of some engines — "
        "likely caused by unsorted input or wrong group_col"
    )
    print("  [PASS] rolling std is 0 at first cycle of each engine")