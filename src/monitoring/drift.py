"""
drift.py — PSI-based covariate drift detection for CMAPSS sensor streams

Why PSI over KS-test:
    PSI has agreed industry thresholds (0.1 / 0.25) with a graded severity scale —
    a monitoring dashboard can show "stable / minor / major" without tuning p-value
    cutoffs. KS-test gives a binary alarm per sensor with no natural ops threshold,
    and multiple-testing inflation across 14 sensors inflates false positives.

Why not MMD:
    MMD detects joint distributional shift in one statistic, but requires kernel
    bandwidth tuning and has no agreed severity scale. For per-sensor actionability
    (which sensor drifted and by how much?) PSI is more interpretable.

Why per-cluster computation:
    FD004 has 6 operating conditions. A batch of test engines running at a new
    op-point looks like global sensor drift but is just regime change. Computing
    PSI within each op_cluster before aggregating prevents false positives from
    fleet-composition shifts masquerading as sensor degradation.

Design:
    - fit_psi_reference  : build bin edges from TRAIN (never from test — no leakage)
    - compute_psi_sensor : PSI for one sensor given reference + current histograms
    - compute_psi_report : full report; per-cluster breakdown when cluster_col present
    - DriftReport        : structured output importable in evaluation notebooks
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── severity thresholds (industry standard) ──────────────────────────────────
PSI_STABLE = 0.10   # below → no action
PSI_MINOR  = 0.25   # [0.10, 0.25) → investigate
# above PSI_MINOR → major shift, retrain alert

N_BINS_DEFAULT = 10  # uniform-quantile bins; 10 is standard for PSI


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class SensorDrift:
    sensor:   str
    psi:      float
    severity: str           # "stable" | "minor" | "major"
    n_ref:    int
    n_cur:    int
    bin_edges: np.ndarray   # for reproducibility / plotting

    def __repr__(self) -> str:
        return f"SensorDrift({self.sensor}: PSI={self.psi:.4f} [{self.severity}])"


@dataclass
class DriftReport:
    sensor_reports: list[SensorDrift]
    cluster:        str | None        # None = global report
    max_psi:        float = field(init=False)
    overall:        str   = field(init=False)
    drifted:        list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.max_psi  = max((s.psi for s in self.sensor_reports), default=0.0)
        self.overall  = _severity(self.max_psi)
        self.drifted  = [s.sensor for s in self.sensor_reports if s.severity != "stable"]

    def __repr__(self) -> str:
        tag = f"cluster={self.cluster}" if self.cluster else "global"
        return (f"DriftReport({tag}): max_PSI={self.max_psi:.4f} [{self.overall}] "
                f"| drifted sensors: {self.drifted or 'none'}")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"sensor": s.sensor, "psi": s.psi, "severity": s.severity,
             "n_ref": s.n_ref, "n_cur": s.n_cur}
            for s in self.sensor_reports
        ]).sort_values("psi", ascending=False).reset_index(drop=True)


# ── internal helpers ─────────────────────────────────────────────────────────

def _severity(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_MINOR:
        return "minor"
    return "major"


def _safe_psi(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    """
    PSI = Σ (p_cur - p_ref) * ln(p_cur / p_ref)

    Zero bins in either distribution are handled by adding a small epsilon
    (1e-4 proportional floor) — the standard approach described in
    Siddiqi (2006) 'Credit Risk Scorecards'.  Replacing with 0 would silently
    ignore missing bins; replacing with NaN would break the sum.
    """
    eps     = 1e-4
    p_ref   = np.where(p_ref == 0, eps, p_ref)
    p_cur   = np.where(p_cur == 0, eps, p_cur)
    # renormalise after eps-filling so they sum to 1
    p_ref  /= p_ref.sum()
    p_cur  /= p_cur.sum()
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


# ── public API ───────────────────────────────────────────────────────────────

def fit_psi_reference(
    train: pd.DataFrame,
    sensor_cols: list[str],
    n_bins: int = N_BINS_DEFAULT,
) -> dict[str, np.ndarray]:
    """
    Compute uniform-quantile bin edges from TRAIN data only.

    Why quantile bins instead of uniform-width bins:
        Sensor distributions can be heavily skewed (e.g. s12 temperature spans
        decades of variance depending on op-condition). Quantile bins guarantee
        roughly equal expected counts per bin under the reference distribution,
        which is the assumption underlying PSI's 0.1/0.25 thresholds.

    Returns
    -------
    bin_edges : {sensor: np.ndarray of shape (n_bins+1,)}
                Persist with joblib alongside scalers for inference.
    """
    present    = [s for s in sensor_cols if s in train.columns]
    bin_edges: dict[str, np.ndarray] = {}

    for sensor in present:
        vals = train[sensor].dropna().values
        if len(vals) < n_bins:
            warnings.warn(f"  [PSI] {sensor}: only {len(vals)} non-null values — using fewer bins")
            n_b = max(2, len(vals) // 2)
        else:
            n_b = n_bins
        quantiles  = np.linspace(0, 100, n_b + 1)
        edges      = np.unique(np.percentile(vals, quantiles))
        # ensure at least 2 distinct edges
        if len(edges) < 2:
            edges = np.array([vals.min() - 1e-9, vals.max() + 1e-9])
        bin_edges[sensor] = edges

    print(f"  [PSI] reference fit: {len(bin_edges)} sensors, {n_bins} quantile bins")
    return bin_edges


def compute_psi_sensor(
    ref_vals: np.ndarray,
    cur_vals: np.ndarray,
    bin_edges: np.ndarray,
) -> float:
    """
    PSI for a single sensor.

    Parameters
    ----------
    ref_vals  : 1-D array — reference (train) values for this sensor
    cur_vals  : 1-D array — current (test / production batch) values
    bin_edges : bin edges from fit_psi_reference for this sensor
    """
    ref_counts, _ = np.histogram(ref_vals, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_vals, bins=bin_edges)

    p_ref = ref_counts / max(ref_counts.sum(), 1)
    p_cur = cur_counts / max(cur_counts.sum(), 1)

    return _safe_psi(p_ref, p_cur)


def compute_psi_report(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    bin_edges: dict[str, np.ndarray],
    sensor_cols: list[str] | None = None,
    cluster_col: str | None = "op_cluster",
) -> list[DriftReport]:
    """
    Full drift report across all sensors, optionally per op_cluster.

    Parameters
    ----------
    ref_df      : training DataFrame (reference distribution)
    cur_df      : current batch DataFrame (production / test)
    bin_edges   : output of fit_psi_reference
    sensor_cols : sensors to check; defaults to all keys in bin_edges
    cluster_col : column for per-cluster breakdown. Pass None for global-only.

    Returns
    -------
    reports : list[DriftReport]
              First element is always the global report.
              Subsequent elements are per-cluster (if cluster_col present in both dfs).
    """
    if sensor_cols is None:
        sensor_cols = list(bin_edges.keys())
    present = [s for s in sensor_cols if s in bin_edges and s in ref_df.columns and s in cur_df.columns]

    if not present:
        raise ValueError("No overlapping sensors between bin_edges and DataFrames.")

    reports: list[DriftReport] = []

    def _build_report(ref: pd.DataFrame, cur: pd.DataFrame, cluster: str | None) -> DriftReport:
        sensor_results = []
        for sensor in present:
            ref_vals = ref[sensor].dropna().values
            cur_vals = cur[sensor].dropna().values
            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue
            psi = compute_psi_sensor(ref_vals, cur_vals, bin_edges[sensor])
            sensor_results.append(SensorDrift(
                sensor    = sensor,
                psi       = psi,
                severity  = _severity(psi),
                n_ref     = len(ref_vals),
                n_cur     = len(cur_vals),
                bin_edges = bin_edges[sensor],
            ))
        return DriftReport(sensor_reports=sensor_results, cluster=cluster)

    # Global report
    global_report = _build_report(ref_df, cur_df, cluster=None)
    reports.append(global_report)
    print(f"  [PSI] global: {global_report}")

    # Per-cluster breakdown
    if (cluster_col
            and cluster_col in ref_df.columns
            and cluster_col in cur_df.columns):
        clusters = sorted(set(ref_df[cluster_col].unique()) | set(cur_df[cluster_col].unique()))
        for c in clusters:
            ref_c = ref_df[ref_df[cluster_col] == c]
            cur_c = cur_df[cur_df[cluster_col] == c]
            if len(ref_c) == 0 or len(cur_c) == 0:
                continue
            r = _build_report(ref_c, cur_c, cluster=str(c))
            reports.append(r)
            if r.overall != "stable":
                print(f"  [PSI] {r}")

    return reports


def summarise_drift(reports: list[DriftReport]) -> pd.DataFrame:
    """
    Aggregate all DriftReports into a flat DataFrame for logging / dashboards.

    Columns: cluster, sensor, psi, severity, n_ref, n_cur
    """
    rows = []
    for report in reports:
        cluster_label = report.cluster or "global"
        for s in report.sensor_reports:
            rows.append({
                "cluster":  cluster_label,
                "sensor":   s.sensor,
                "psi":      s.psi,
                "severity": s.severity,
                "n_ref":    s.n_ref,
                "n_cur":    s.n_cur,
            })
    return pd.DataFrame(rows).sort_values(["cluster", "psi"], ascending=[True, False]).reset_index(drop=True)
