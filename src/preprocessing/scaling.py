"""
scaling.py — per-op_cluster StandardScaler for CMAPSS sensors

StandardScaler chosen over MinMaxScaler: test sensor values can exceed train min/max
bounds as engines degrade — StandardScaler handles this as high z-scores,
not silent out-of-bounds values.

Adaptive n_clusters: single-condition subsets (e.g. FD001) collapse to 1 cluster.
n_clusters is clamped to the actual number of distinct operating conditions found.

fit ONLY on train; transform test with same fitted scalers (no leakage).
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

OP_COLS       = ["op1", "op2", "op3"]
N_OP_CLUSTERS = 6
RANDOM_SEED   = 42


def fit_op_clusters(train: pd.DataFrame, n_clusters: int = N_OP_CLUSTERS) -> KMeans:
    """
    Cluster operating conditions using KMeans — fit on train op settings only.
    Clamps n_clusters to the actual number of distinct op conditions found.
    """
    n_distinct = len(train[OP_COLS].round(2).drop_duplicates())
    effective_k = min(n_clusters, n_distinct)
    if effective_k < n_clusters:
        print(f"  [INFO] clamped op clusters: {n_clusters} → {effective_k} "
              f"({n_distinct} distinct op conditions found)")

    km = KMeans(n_clusters=effective_k, random_state=RANDOM_SEED, n_init=10)
    km.fit(train[OP_COLS])
    return km


def assign_op_clusters(df: pd.DataFrame, km: KMeans) -> pd.DataFrame:
    """Assign op_cluster labels using a pre-fitted KMeans."""
    df = df.copy()
    df["op_cluster"] = km.predict(df[OP_COLS])
    return df


def fit_scalers(
    train: pd.DataFrame,
    sensor_cols: list[str],
) -> dict[int, StandardScaler]:
    """
    Fit one StandardScaler per op_cluster on training data.
    Groups with < 2 rows fall back to full-dataset statistics.
    Returns dict keyed by op_cluster (int).
    """
    scalers: dict[int, StandardScaler] = {}

    for op_cluster, idx in train.groupby("op_cluster").groups.items():
        group_data = train.loc[idx, sensor_cols]
        if len(group_data) < 2:
            print(f"  [WARN] op_cluster={op_cluster} has only {len(group_data)} row(s); "
                  f"falling back to full-dataset statistics")
            group_data = train[sensor_cols]

        scaler = StandardScaler()
        scaler.fit(group_data)
        scalers[int(op_cluster)] = scaler

    return scalers


def apply_scalers(
    df: pd.DataFrame,
    scalers: dict[int, StandardScaler],
    sensor_cols: list[str],
) -> pd.DataFrame:
    """
    Transform df using pre-fitted scalers keyed by op_cluster.
    Raises KeyError on unseen op_cluster values.
    """
    df = df.copy()

    for op_cluster, idx in df.groupby("op_cluster").groups.items():
        key = int(op_cluster)
        if key not in scalers:
            raise KeyError(
                f"No scaler for op_cluster={key}. "
                "Ensure KMeans was fit on train and same model used to assign test clusters."
            )
        df.loc[idx, sensor_cols] = scalers[key].transform(df.loc[idx, sensor_cols])

    return df


def scale_sensors(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sensor_cols: list[str],
    n_op_clusters: int = N_OP_CLUSTERS,
) -> tuple[pd.DataFrame, pd.DataFrame, KMeans, dict]:
    """
    Full scaling pipeline — no leakage:
    1. Fit KMeans on train op settings (adaptive k)
    2. Assign op_cluster to train and test using same KMeans
    3. Fit one StandardScaler per op_cluster on train only
    4. Transform both train and test with those scalers

    Returns (train_scaled, test_scaled, km, scalers).
    """
    km    = fit_op_clusters(train, n_clusters=n_op_clusters)
    train = assign_op_clusters(train, km)
    test  = assign_op_clusters(test,  km)

    scalers = fit_scalers(train, sensor_cols)
    train   = apply_scalers(train, scalers, sensor_cols)
    test    = apply_scalers(test,  scalers, sensor_cols)

    print(f"  fitted {len(scalers)} StandardScalers across {len(scalers)} op_clusters")
    return train, test, km, scalers


def save_scaling_artifacts(
    km: KMeans,
    scalers: dict,
    artifacts_dir: str | Path,
) -> None:
    """Persist KMeans and scalers for inference — must not re-fit on new data."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(km,      artifacts_dir / "kmeans_op_clusters.pkl")
    joblib.dump(scalers, artifacts_dir / "scalers.pkl")
    print(f"  saved kmeans_op_clusters.pkl and scalers.pkl → {artifacts_dir}")


def load_scaling_artifacts(artifacts_dir: str | Path) -> tuple[KMeans, dict]:
    """Load persisted KMeans and scalers for inference or downstream notebooks."""
    artifacts_dir = Path(artifacts_dir)
    km      = joblib.load(artifacts_dir / "kmeans_op_clusters.pkl")
    scalers = joblib.load(artifacts_dir / "scalers.pkl")
    print(f"  loaded scaling artifacts from {artifacts_dir}")
    return km, scalers


def verify_scaling(
    train: pd.DataFrame,
    sensor_cols: list[str],
    mean_tol: float = 0.05,
) -> None:
    """
    Assert per-op_cluster means are near zero and stds are non-zero.
    Raises AssertionError on failure — not informational-only.
    Global mean is NOT asserted (StandardScaler normalises per-cluster, not globally).
    """
    for op_cluster, group_df in train.groupby("op_cluster"):
        group_means = group_df[sensor_cols].mean().abs()
        group_stds  = group_df[sensor_cols].std()

        max_mean = group_means.max()
        assert max_mean < mean_tol, (
            f"op_cluster={op_cluster}: sensor mean abs max {max_mean:.4f} "
            f"exceeds tolerance {mean_tol}. Worst sensor: {group_means.idxmax()}"
        )

        zero_std = group_stds[group_stds == 0]
        assert zero_std.empty, (
            f"op_cluster={op_cluster}: zero-variance sensors after scaling: "
            f"{zero_std.index.tolist()} — these should have been dropped."
        )

    n_groups     = train["op_cluster"].nunique()
    global_stds  = train[sensor_cols].std()
    print(f"  [PASS] per-cluster means < {mean_tol} and stds > 0 for all {n_groups} op_clusters")
    print(f"  [INFO] global sensor std range: [{global_stds.min():.4f}, {global_stds.max():.4f}]")