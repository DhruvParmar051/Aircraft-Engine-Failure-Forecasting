"""
scaling.py — per-(dataset_id, op_cluster) StandardScaler for CMAPSS sensors

StandardScaler chosen over MinMaxScaler: test sensor values can exceed train min/max
bounds as engines degrade further — StandardScaler handles this as high z-scores,
not silent out-of-bounds values.

Adaptive n_clusters: FD001/FD003 have only 1 operating condition. Fitting
KMeans(n_clusters=6) on near-identical points creates statistically unstable scalers.
n_clusters is clamped to the actual number of distinct global operating conditions.

fit ONLY on train; transform test with same fitted scalers (no leakage)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

OP_COLS = ["op1", "op2", "op3"]
N_OP_CLUSTERS = 6
RANDOM_SEED = 42


def fit_op_clusters(train: pd.DataFrame, n_clusters: int = N_OP_CLUSTERS) -> KMeans:
    """
    cluster operating conditions using KMeans — fit on all training op settings
    clamps n_clusters to the actual number of distinct global op conditions
    FD001-only data collapses to 1 cluster; full combined data uses up to 6
    """
    n_distinct_global = len(train[OP_COLS].round(2).drop_duplicates())
    effective_k = min(n_clusters, n_distinct_global)
    if effective_k < n_clusters:
        print(f"  [INFO] clamped op clusters: {n_clusters} → {effective_k} "
              f"({n_distinct_global} distinct op conditions found)")

    km = KMeans(n_clusters=effective_k, random_state=RANDOM_SEED, n_init=10)
    km.fit(train[OP_COLS])
    return km


def assign_op_clusters(df: pd.DataFrame, km: KMeans) -> pd.DataFrame:
    """assign operating condition cluster labels using a pre-fitted KMeans"""
    df = df.copy()
    df["op_cluster"] = km.predict(df[OP_COLS])
    return df


def fit_scalers(
    train: pd.DataFrame,
    sensor_cols: list[str],
) -> dict[tuple[int, int], StandardScaler]:
    """
    fit one StandardScaler per (dataset_id, op_cluster) group on training data
    groups with < 2 rows cannot compute std — fall back to full subset statistics
    returns dict of scalers keyed by (dataset_id, op_cluster)
    """
    scalers: dict[tuple[int, int], StandardScaler] = {}
    for (dataset_id, op_cluster), idx in train.groupby(["dataset_id", "op_cluster"]).groups.items():
        group_data = train.loc[idx, sensor_cols]
        if len(group_data) < 2:
            print(f"  [WARN] group (dataset={dataset_id}, cluster={op_cluster}) has only "
                  f"{len(group_data)} row(s); falling back to full subset statistics")
            group_data = train.loc[train["dataset_id"] == dataset_id, sensor_cols]
        scaler = StandardScaler()
        scaler.fit(group_data)
        scalers[(dataset_id, op_cluster)] = scaler
    return scalers


def apply_scalers(
    df: pd.DataFrame,
    scalers: dict[tuple[int, int], StandardScaler],
    sensor_cols: list[str],
) -> pd.DataFrame:
    """
    transform df using pre-fitted scalers
    raises KeyError on unseen (dataset_id, op_cluster) combinations
    """
    df = df.copy()
    for (dataset_id, op_cluster), idx in df.groupby(["dataset_id", "op_cluster"]).groups.items():
        key = (dataset_id, op_cluster)
        if key not in scalers:
            raise KeyError(
                f"No scaler for (dataset_id={dataset_id}, op_cluster={op_cluster}). "
                "Ensure KMeans was fit on train and test clusters assigned with the same model."
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
    full scaling pipeline (no leakage):
    1. fit KMeans on train op settings only (adaptive k)
    2. assign clusters to train and test using same KMeans
    3. fit StandardScaler per (dataset_id, op_cluster) on train only
    4. transform both with those scalers
    returns (train_scaled, test_scaled, km, scalers)
    """
    km = fit_op_clusters(train, n_clusters=n_op_clusters)
    train = assign_op_clusters(train, km)
    test = assign_op_clusters(test, km)

    scalers = fit_scalers(train, sensor_cols)
    train = apply_scalers(train, scalers, sensor_cols)
    test = apply_scalers(test, scalers, sensor_cols)

    print(f"  fitted {len(scalers)} StandardScalers across (dataset_id × op_cluster) groups")
    return train, test, km, scalers


def save_scaling_artifacts(
    km: KMeans,
    scalers: dict,
    artifacts_dir: str | Path,
) -> None:
    """persist KMeans and scalers for inference — must not re-fit on new data"""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, artifacts_dir / "kmeans_op_clusters.pkl")
    joblib.dump(scalers, artifacts_dir / "scalers.pkl")
    print(f"  saved kmeans_op_clusters.pkl and scalers.pkl → {artifacts_dir}")


def load_scaling_artifacts(artifacts_dir: str | Path) -> tuple[KMeans, dict]:
    """load persisted KMeans and scalers for inference or downstream notebooks"""
    artifacts_dir = Path(artifacts_dir)
    km = joblib.load(artifacts_dir / "kmeans_op_clusters.pkl")
    scalers = joblib.load(artifacts_dir / "scalers.pkl")
    print(f"  loaded scaling artifacts from {artifacts_dir}")
    return km, scalers


def verify_scaling(
    train: pd.DataFrame,
    sensor_cols: list[str],
    group_cols: list[str] = ["dataset_id", "op_cluster"],
    mean_tol: float = 0.05,
) -> None:
    """
    assert that per-group means are near zero and stds are non-zero
    IMPROVEMENT: was informational-only — now raises AssertionError on failure
    global mean is NOT asserted (StandardScaler normalises per-group, not globally)
    """
    for group_key, group_df in train.groupby(group_cols):
        group_means = group_df[sensor_cols].mean().abs()
        group_stds  = group_df[sensor_cols].std()

        max_mean = group_means.max()
        assert max_mean < mean_tol, (
            f"Group {group_key}: sensor mean abs max {max_mean:.4f} exceeds tolerance {mean_tol}. "
            f"Worst sensor: {group_means.idxmax()}"
        )

        zero_std = group_stds[group_stds == 0]
        assert zero_std.empty, (
            f"Group {group_key}: zero-variance sensors after scaling: {zero_std.index.tolist()}. "
            "These sensors are constant within this group — likely should have been dropped."
        )

    print(f"  [PASS] per-group sensor means < {mean_tol} and stds > 0 for all "
          f"{train.groupby(group_cols).ngroups} groups")
    global_stds = train[sensor_cols].std()
    print(f"  [INFO] global sensor std range: [{global_stds.min():.4f}, {global_stds.max():.4f}]")