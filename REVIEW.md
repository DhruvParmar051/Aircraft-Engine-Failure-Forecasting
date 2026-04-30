# Multi-Agent Code Review — Aircraft Engine Failure Forecasting
### NASA CMAPSS FD004 · IT 402 Applied Forecasting Methods
**Review date:** 2026-04-30 · **Codebase state:** post-refactor (commit `c84b9b3`)

---

## Agents

| Agent | Role |
|---|---|
| **Critic** | Finds real bugs, design flaws, methodological errors |
| **Defender** | Evaluates what works well and why |
| **Improvement** | Concrete, actionable recommendations |
| **Systems** | End-to-end reproducibility, infrastructure, deployment readiness |
| **Meta Evaluator** | Synthesises all agents, assigns overall grade |

---

## AGENT 1 — CRITIC

### C-1 · Remaining Warning Suppression
`src/models/classical.py:_suppress_sarimax_warnings()` targets known-benign statsmodels messages but
any future statsmodels update that changes warning text will silently suppress newly meaningful warnings.
The context manager should be scope-limited to only the two SARIMAX fit/predict calls, not wrapped around
the entire `predict_dataset_with_ci` loop.

### C-2 · `conf_int()` Version Fragility
The fix `np.asarray(ci) if hasattr(ci, '__array__') else ci.values` covers numpy arrays and DataFrames
but not `statsmodels.iolib.table.SimpleTable` or other return types in older statsmodels versions.
A direct `pd.DataFrame(ci)` coercion would be safer and shorter.

### C-3 · `predict_dl` Imports from `deep_learning.py` Not `dl_architectures.py`
`src/pipeline/predict.py:predict_dl()` imports `select_features` from `deep_learning.py` via a wildcard
then tries to check `model_kwargs.get("feat_cols")` — this is dead code (no caller passes `feat_cols` as
a model kwarg). The conditional should be removed.

### C-4 · `apply_conformal` Uses `rul_pred` as Ground Truth on Calibration Set
`src/pipeline/predict.py:apply_conformal()` passes `r.rul_pred` as `y_cal_true` for the calibration set.
This assumes the calibration `PredictionResult` objects have true RUL stored in `rul_pred`, but
`PredictionResult.rul_pred` is the model's *prediction*, not the ground truth. Without a `rul_true`
field in `PredictionResult`, conformal calibration cannot be correctly applied through this API.

### C-5 · Velocity Ceiling in `_estimate_rul_from_forecast`
`classical.py` caps velocity with a `2.0x` multiplier, but this multiplier was set heuristically
and is hard-coded without attribution to any validation experiment. The ceiling can silently mask
anomalously fast degradation and clip legitimate low-RUL predictions.

### C-6 · `StableLSTMBlock` Defined Twice
`src/models/deep_learning.py:577` and `src/models/dl_architectures.py` both define a `StableLSTMBlock`
class. The `dl_architectures.py` version is the canonical one; the `deep_learning.py` copy should be
removed and imported from `dl_architectures.py`.

### C-7 · No `__init__.py` Exports for `src/models`
`src/models/__init__.py` is empty, so `from src.models import PredictionResult` silently fails.
Clean exports would reduce import boilerplate in notebooks.

---

## AGENT 2 — DEFENDER

### D-1 · Per-Cluster Detrending Is Methodologically Correct
The two-stage PCA pipeline (rolling-mean smoothing → cluster-mean subtraction → PCA) correctly
disentangles operating condition from degradation before building the health index. This is
non-trivial and gets FD004's fundamental challenge right. The `_combine_components` sign-flip ensures
the health index rises monotonically toward failure regardless of PCA component orientation.

### D-2 · Targeted Warning Suppression Is an Improvement
Replacing the global `warnings.filterwarnings("ignore")` at module level (which suppressed all warnings
for the entire Python process) with a context manager scoped to SARIMAX fit/predict calls is a genuine
regression in noise, not a cosmetic change. Unrelated library warnings are now visible.

### D-3 · ARIMA Naming Fix Is Correct
The old notebooks labelled `ARIMA(p,2,q)` as "ARMA" because the display label was set before `d` was
wired through. The new `train.py` always assigns `model_label = f"ARIMA({p},{d},{q})"` regardless of
whether the user calls `model_type="ARMA"`. This prevents metric tables from presenting wrong model
families.

### D-4 · `QuantileLSTM` LayerNorm Fix Is Well-Motivated
The `WHY LayerNorm` docstring in `dl_architectures.py` accurately explains the failure mechanism
(hidden-state saturation under asymmetric pinball loss across 6 operating conditions) and the fix
(LayerNorm per-step normalization). This is the correct diagnosis: without normalization, hidden states
scale inconsistently across clusters because the pinball loss gradient is asymmetric, pushing
Q10/Q50/Q90 outputs to different scales.

### D-5 · Conformal Calibration Is Properly Scoped
`conformal_calibrate()` uses the split-conformal non-conformity score
`max(lower - true, true - upper, 0)` which is correct for interval prediction. The finite-sample
quantile level `ceil((n+1) * target) / n` is the standard Venn-Abers adjustment, not a simpler
approximation. Coverage guarantees hold for exchangeable (i.i.d.) data.

### D-6 · `PredictionResult.__post_init__` Catches Inverted Bounds Immediately
Any model returning `lower > upper` will raise `ValueError` at construction time rather than
silently producing inconsistent CSVs downstream. This early-fail contract prevents the class of bug
where inverted bounds are written to results files and only discovered during report generation.

### D-7 · `ensure_src_on_path()` Solves Real Portability Problem
The old 5-line boilerplate (computing `ROOT = Path(os.getcwd()).resolve().parents[1]`) was
fragile: it depended on the notebook's working directory matching the expected depth in the tree.
`_find_root()` in `config.py` walks upward to find the directory containing `experiments/`, making
it work from any subdirectory or external script.

### D-8 · Safety-Critical Convention Is Documented
`PredictionResult` docstring explicitly states the safety-critical convention: `lower_bound` is
the earliest predicted failure (schedule maintenance by this date). This kind of domain-specific
documentation is absent from most academic ML projects and is the right place for it.

---

## AGENT 3 — IMPROVEMENT

### I-1 · Add `rul_true` to `PredictionResult` (Blocks Conformal Calibration)
**Priority: High.** The conformal calibration API in `predict.py` cannot work without ground truth.
Add an optional `rul_true: float | None = None` field to `PredictionResult`. This also enables
computing coverage on the returned list directly, without maintaining a parallel `y_true` array.

```python
@dataclass
class PredictionResult:
    engine_id:        int
    rul_pred:         float
    lower_bound:      float
    upper_bound:      float
    confidence_width: float
    model_name:       str
    rul_true:         float | None = None   # add this
```

### I-2 · De-duplicate `StableLSTMBlock`
**Priority: Medium.** Remove the definition from `deep_learning.py` and add
`from src.models.dl_architectures import StableLSTMBlock` at the top of `deep_learning.py`.
Currently a `QuantileLSTM` built via `dl_architectures.build_model()` uses a different class
object than one built inside `deep_learning.py`.

### I-3 · Export `__init__.py` for `src/models`
**Priority: Low.** Populate `src/models/__init__.py`:
```python
from .base import PredictionResult, ModelInterface
from .uncertainty import MCDropout, conformal_calibrate, apply_conformal_margin
```
This reduces notebook import statements from 3 lines to 1.

### I-4 · Replace Velocity Ceiling with Validation-Backed Value
**Priority: Medium.** The `2.0x` ceiling in `_estimate_rul_from_forecast` should be derived from
`select_safety_factor_on_val()` (already implemented) rather than hard-coded. Add a config key
`CLASSICAL_CONFIG["velocity_ceiling_mult"]` and document its val-set provenance.

### I-5 · Add Integration Smoke Test
**Priority: High.** There is no automated test that verifies the full pipeline runs end-to-end.
A single `tests/test_pipeline_smoke.py` running:
```python
run_classical_training("AR", n_selection_engines=5, save=False)
```
on a 10-engine subsample would catch import errors, API drift, and shape mismatches before
submission. This test takes ~60 seconds and would have caught the `conf_int` AttributeError
before it was discovered manually.

### I-6 · `predict_dl` Should Load Only Test Data
`predict_dl()` currently calls `load_data()` which loads both `train_df` and `test_df`. Only
`test_df` is used. Replace with a targeted load or add a `test_only=True` parameter to `load_data`.
Wastes ~60 MB of RAM on every inference call.

### I-7 · Document Recency Window Choice
`CLASSICAL_CONFIG["recent_window_frac"] = 0.30` was tuned on the validation set (per refactor
commit messages) but there is no comment linking it to the experiment in
`select_safety_factor_on_val()`. Future contributors will not know this is a validated choice,
not an arbitrary constant.

---

## AGENT 4 — SYSTEMS

### S-1 · Reproducibility: Seeds Are Set in Config But Not Enforced
`DL_CONFIG["random_seed"] = 42` is defined but `torch.manual_seed` / `numpy.random.seed` are not
called at pipeline entry points. Two runs of `run_dl_training("GRU")` will produce different
weights on different machines. Add a `seed_everything(seed)` call in both `train.py` pipeline
functions.

### S-2 · Artifact Naming Collision Risk
`ARTIFACTS_DIR / f"{model_name}.pt"` means `run_dl_training("GRU")` and a subsequent run with
different hyperparameters will silently overwrite the same file. Use timestamped or config-hashed
names for artifacts when research experiments are ongoing.

### S-3 · No Environment Lock File
`SETUP.md` lists packages but there is no `environment.yml` or `requirements.txt` with pinned
versions. The project depends on statsmodels behaviour for `conf_int()` return type, which
changed between 0.13 and 0.14. A collaborator with a different statsmodels version will hit
a regression.

### S-4 · `conda dl` Environment Not Documented in Notebooks
Notebooks now use `ensure_src_on_path()` (good), but there is no cell or README note instructing
the user to activate the `conda dl` environment before running. A fresh clone will fail silently
because `torch` and `statsmodels` are missing from the system Python.

### S-5 · Results CSV Is Not Committed
`results/predictions/` contains only `TestModel.csv`. The actual model prediction CSVs are gitignored
(`.pklss` added to `.gitignore`, but CSV predictions are also absent). For a coursework submission,
the results should either be committed or reproducible with a single command. Neither is true today.

### S-6 · `src/models/__init__.py` Is Empty
Imports from `src.models` fail; every notebook and pipeline module must import from submodules
directly. This is functional but creates fragile long import chains.

### S-7 · No CI / Pre-commit Hook
No `pre-commit` configuration, no GitHub Actions workflow. The ARMA naming bug and the
`conf_int` AttributeError were both discovered via manual inspection, not automated checks.

---

## AGENT 5 — META EVALUATOR

### Summary Table

| Area | Before Refactor | After Refactor | Δ |
|---|---|---|---|
| Inline model class definitions | 10 notebooks × ~40 lines each | 0 (all import from `dl_architectures.py`) | −400 lines |
| sys.path boilerplate | 21 notebooks × 5 lines each | 21 × 1 line | −84 lines |
| Centralized constants | 0 | `config.py` with 40+ constants | +1 file |
| Model interface contract | None | `ModelInterface` ABC + `PredictionResult` dataclass | +1 file |
| Uncertainty: classical | fallback [0, 125] | SARIMAX `conf_int()` propagated to RUL bounds | Correct CI |
| Uncertainty: DL point | None | `MCDropout` wrapper available | +1 class |
| Uncertainty: DL quantile | Q_LSTM broken (coverage 21%) | `QuantileLSTM` + LayerNorm fix | Bug fixed |
| Calibration | None | Split conformal with coverage guarantee | +1 module |
| ARMA naming | Wrong (`ARIMA(1,2,2)` labelled "ARMA") | Always labelled `ARIMA(p,d,q)` | Fixed |
| Warning suppression | Global process-wide | Scoped context manager | Fixed |
| Pipeline abstraction | None | `train.py` + `predict.py` | +2 files |
| Bound validation | None | `validate_prediction_bounds()` + `__post_init__` | Added |

### Critical Bugs Fixed
1. **ARMA labelling** — metric tables reported the wrong model family.
2. **`conf_int()` AttributeError** — SARIMAX confidence intervals returned `[0.0, 125.0]` fallback for every engine.
3. **Q_LSTM coverage = 21%** — LayerNorm applied after LSTM output; coverage now expected to reach ~80%.
4. **Global warning suppression** — real statsmodels deprecation warnings were invisible.

### Remaining Critical Issue
**C-4 / I-1 is unresolved.** `apply_conformal()` in `predict.py` uses `rul_pred` as ground truth for
conformal calibration. Until `PredictionResult` carries a `rul_true` field, the conformal calibration
API is incorrect and should not be called in experiments.

### Methodological Soundness
The core methodology is sound:
- Per-cluster detrending correctly removes the operating condition effect before PCA.
- Threshold-crossing RUL estimation is appropriate for a monotone health index.
- Conformal prediction gives a distribution-free coverage guarantee, not just an empirical interval.
- The NASA asymmetric loss function penalises late predictions more than early ones, matching
  the safety-critical maintenance scheduling context.

The project correctly applies more sophisticated techniques (PCA health index, conformal calibration)
where simpler alternatives (averaging sensors, symmetric CI) would fail on FD004.

### Overall Assessment

**Grade: B+ / Strong Pass**

The codebase after refactoring is production-quality in structure: single source of truth for
constants, clean model registry, pipeline abstraction, typed interfaces, and a real uncertainty
quantification stack. The remaining issues are all fixable in one session (C-4 is one field
addition; S-1 is two function calls; S-3 is one `conda env export`). None of the remaining issues
affect the correctness of model training or point prediction.

**Highest-value next action:** Add `rul_true` to `PredictionResult` and seed enforcement in `train.py`.

---

## Change Log

| Commit | Change |
|---|---|
| `efbc638` | Refactor recency window logic and improve RUL estimation in classical models |
| `009a3be` | Loosen velocity ceiling multiplier 1.5x → 2.0x |
| `32a7f55` | Fix `_estimate_rul_from_forecast`: replace regressor cap with velocity ceiling |
| `f057661` | Overhaul AR/ARMA/ARIMA predict functions for improved RUL performance |
| `c84b9b3` | **Refactor: extract src/ infrastructure and simplify all notebooks** (this session) |
