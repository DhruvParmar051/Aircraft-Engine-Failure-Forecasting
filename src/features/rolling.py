"""
rolling.py — rolling statistics features for CMAPSS time-series per engine
single sensor values are noisy; rolling mean reveals trend, rolling std reveals instability
all windows computed within each engine group (no cross-engine contamination)

sort by (engine_id, cycle) before computing — groupby transform respects physical row order,
so unsorted input silently corrupts rolling windows across non-consecutive cycles.
"""

import pandas as pd

DEFAULT_WINDOWS = [5, 10, 20]
# 5  = local noise filter
# 10 = medium-term trend
# 20 = long-term drift (critical for sequence models with window_size=30)


def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    windows: list[int] = DEFAULT_WINDOWS,
    group_col: str = "engine_id",
) -> pd.DataFrame:
    """
    Add rolling mean and std per sensor per window, grouped by engine.
    Sorts by (group_col, cycle) first — rolling requires consecutive cycle order.
    min_periods=1 prevents NaN at early cycles where the window is not yet full.
    """
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


def get_rolling_col_names(
    sensor_cols: list[str],
    windows: list[int] = DEFAULT_WINDOWS,
) -> list[str]:
    """Return column names that add_rolling_features produces — useful for feature selection."""
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
    Verify rolling features:
    - all expected columns exist
    - no NaN values
    - rolling std is 0 at first cycle of each engine
    """
    expected = get_rolling_col_names(sensor_cols, windows)

    missing = [c for c in expected if c not in df.columns]
    assert not missing, f"Missing rolling feature columns: {missing}"
    print(f"  [PASS] all {len(expected)} rolling feature columns present")

    nan_count = df[expected].isnull().sum().sum()
    assert nan_count == 0, f"NaN values in rolling features: {nan_count}"
    print("  [PASS] no NaN values in rolling features")

    # first cycle per engine by actual min cycle — not positional
    first_cycle_idx = df.groupby(group_col)["cycle"].idxmin()
    rstd_cols       = [c for c in expected if "_rstd_" in c]
    first_rstd      = df.loc[first_cycle_idx, rstd_cols]
    assert (first_rstd == 0).all().all(), (
        "Rolling std is not 0 at first cycle of some engines — "
        "likely caused by unsorted input or wrong group_col"
    )
    print("  [PASS] rolling std is 0 at first cycle of each engine")