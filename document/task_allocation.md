# NASA CMAPSS Project — Team Execution Plan (2 Weeks)

## Context

Team of 4 students (Dhruv, Aditya, Shrey, Umme) building a predictive maintenance project for IT 402 using NASA CMAPSS dataset. Goal: implement 14+ models, compare performance, and produce a final report in 14 days.

---

## Suggested Unique/Advanced Model: **Temporal Fusion Transformer (TFT)**

Why TFT:

- Combines attention mechanisms with temporal processing

- Handles static covariates (dataset_id), known inputs (operational settings), and observed inputs (sensors)

- Produces interpretable attention weights (shows which time steps and features matter)

- Generates probabilistic forecasts natively (quantile outputs built-in)

- Published by Google (2019) — impressive for an academic project but not overly complex to implement (available in `pytorch-forecasting` library)

- Directly applicable to RUL prediction with mixed operating conditions

---

## Team Roles & Model Allocation

### Dhruv — Team Lead + Classical Models + TFT

| Task                                              | Models                                |
| ------------------------------------------------- | ------------------------------------- |
| Data preprocessing & pipeline (shared foundation) | —                                     |
| Classical models                                  | AR, ARMA, ARIMA                       |
| Advanced model                                    | **Temporal Fusion Transformer (TFT)** |
| Final integration & report coordination           | —                                     |

**Why this allocation:** As team lead, Dhruv owns the data pipeline that everyone depends on. Classical models are quicker to implement, freeing time for the TFT and coordination duties.

### Aditya — Deep Learning (Standard)

| Task                                    | Models                     |
| --------------------------------------- | -------------------------- |
| Feature engineering for sequence models | —                          |
| Standard deep learning                  | MLP, RNN                   |
| Quantile variants                       | Quantile MLP, Quantile RNN |
| Evaluation metrics implementation       | —                          |

### Shrey — Deep Learning (LSTM/GRU)

| Task                               | Models              |
| ---------------------------------- | ------------------- |
| Sequence windowing utility         | —                   |
| LSTM-based models                  | LSTM, Quantile LSTM |
| GRU-based models                   | GRU, Quantile GRU   |
| Visualization of model comparisons | —                   |

### Umme — TCN + Analysis

| Task                          | Models            |
| ----------------------------- | ----------------- |
| EDA & sensor analysis         | —                 |
| TCN models                    | TCN, Quantile TCN |
| Survival/degradation analysis | —                 |
| Final report writing          | —                 |

### Balance Check

| Member | Classical           | Standard DL   | Quantile          | Advanced | Other                       |
| ------ | ------------------- | ------------- | ----------------- | -------- | --------------------------- |
| Dhruv  | AR, ARMA, ARIMA (3) | —             | —                 | TFT (1)  | Data pipeline, coordination |
| Aditya | —                   | MLP, RNN (2)  | Q-MLP, Q-RNN (2)  | —        | Evaluation metrics          |
| Shrey  | —                   | LSTM, GRU (2) | Q-LSTM, Q-GRU (2) | —        | Visualizations              |
| Umme   | —                   | —             | Q-TCN (1)         | TCN (1)  | EDA, report writing         |

**Total models per person:** Dhruv: 4, Aditya: 4, Shrey: 4, Umme: 2 + heavy analysis/reporting

---

## 2-Week Timeline

### Phase 1: Foundation (Days 1–3)

| Day       | Dhruv                                                                                                              | Aditya                                                                                                 | Shrey                                                                                                           | Umme                                                                                                                               |
| --------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Day 1** | Load & combine all 4 datasets into train.csv/test.csv. Compute RUL. Set up shared repo/folder structure.           | Set up Python environment. Install libraries (torch, sklearn, statsmodels, darts/pytorch-forecasting). | Set up environment. Study LSTM/GRU architecture for RUL prediction.                                             | Set up environment. Begin EDA: sensor distributions, lifecycle plots, correlation matrix.                                          |
| **Day 2** | Feature engineering: drop constant sensors, normalize, create rolling features. Share the clean dataset with team. | Study MLP/RNN for time series. Design windowing function for sequence inputs.                          | Build the **sequence windowing utility** (sliding window of W=30 cycles → 3D tensor). Share with Aditya & Umme. | Complete EDA: identify useful vs useless sensors, operating condition clusters, lifetime distributions. Create EDA visualizations. |
| **Day 3** | Implement **AR, ARMA, ARIMA** on per-engine sensor trends. Document results.                                       | Implement shared **evaluation module**: RMSE + NASA scoring function. Share with team.                 | Test windowing utility on FD001. Verify shapes are correct for PyTorch.                                         | Write up EDA findings. Begin studying TCN architecture.                                                                            |

**Day 3 Checkpoint:** Everyone has clean data, shared utilities (windowing + evaluation), and EDA insights.

### Phase 2: Core Modeling (Days 4–8)

| Day       | Dhruv                                                                                       | Aditya                                                              | Shrey                                                   | Umme                                                       |
| --------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------- |
| **Day 4** | Finalize classical model results. Begin studying TFT (read pytorch-forecasting docs).       | Implement **MLP** for RUL prediction. Train, evaluate, log results. | Implement **LSTM** model. Train, evaluate, log results. | Implement **TCN** model. Train, evaluate, log results.     |
| **Day 5** | Begin **TFT** implementation using pytorch-forecasting.                                     | Implement **RNN** model. Train, evaluate, log results.              | Implement **GRU** model. Train, evaluate, log results.  | Tune TCN hyperparameters. Debug if needed.                 |
| **Day 6** | Continue TFT — handle dataset_id as static covariate, operational settings as known inputs. | Implement **Quantile MLP** (modify loss to quantile/pinball loss).  | Implement **Quantile LSTM** (add quantile loss).        | Implement **Quantile TCN** (add quantile loss).            |
| **Day 7** | TFT training & tuning.                                                                      | Implement **Quantile RNN**.                                         | Implement **Quantile GRU**.                             | Refine Quantile TCN. Begin preparing comparison templates. |
| **Day 8** | Finalize TFT. Extract attention weights for interpretability.                               | Tune all 4 models. Re-run final evaluations.                        | Tune all 4 models. Re-run final evaluations.            | Finalize both TCN models.                                  |

**Day 8 Checkpoint:** All 14+ models trained and evaluated. Each person has RMSE + NASA scores logged.

### Phase 3: Evaluation & Comparison (Days 9–11)

| Day        | Dhruv                                                                                                                  | Aditya                                                                                            | Shrey                                                                                                                         | Umme                                                                                |
| ---------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Day 9**  | Collect all model results into a single comparison table. Rank models.                                                 | Create per-model prediction vs actual plots. Residual analysis.                                   | Create **comparison visualizations**: bar charts (RMSE), line plots (predicted vs true RUL), box plots (error distributions). | Begin writing the **final report** structure: intro, dataset, methodology sections. |
| **Day 10** | Analyze: which models work best for which dataset subsets (FD001 vs FD004)? Does TFT interpretability reveal insights? | Error analysis: where do models fail? Short-RUL vs long-RUL engines. Operating condition effects. | Quantile model analysis: prediction intervals, coverage probability, calibration.                                             | Continue report: write EDA section, model description sections.                     |
| **Day 11** | Write analysis section for classical models + TFT. Review all team submissions.                                        | Write analysis section for MLP/RNN + quantile variants.                                           | Write analysis section for LSTM/GRU + quantile variants.                                                                      | Write TCN section. Integrate all sections into report draft.                        |

### Phase 4: Final Assembly (Days 12–14)

| Day        | Dhruv                                                                            | Aditya                                                            | Shrey                                                                          | Umme                                                   |
| ---------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------ |
| **Day 12** | Review full report draft. Fill gaps. Ensure consistency across sections.         | Create final summary table of all models. Proofread own sections. | Create final visualizations package. Ensure all plots are publication-quality. | Finalize report: conclusions, future work, references. |
| **Day 13** | **Buffer day.** Fix any remaining issues. Run final model experiments if needed. | Buffer day. Help with report or fix model issues.                 | Buffer day. Polish visualizations.                                             | Buffer day. Final report formatting and polish.        |
| **Day 14** | Final review. Package everything (code + report + data). **Submit.**             | Final review of own deliverables.                                 | Final review of own deliverables.                                              | Final report compilation. Export to PDF.               |

---

## Dependencies & Workflow

```

Day 1: Dhruv (data pipeline) ──────────────────────┐

                                                     │

Day 2: Dhruv (features) + Shrey (windowing util) ───┤

                                                     │

Day 3: Aditya (eval module) ─────────────────────────┤

                                                     ▼

Day 4+: ALL MODELING IS PARALLEL (everyone has clean data + utils + eval)

                                                     │

Day 9: Results flow TO Dhruv (comparison table)      │

       Results flow TO Umme (report)                 │

       Results flow TO Shrey (visualizations)        ▼

Day 14: Everything assembled

```

### Critical Path (blocking dependencies)

1. **Dhruv's data pipeline (Day 1)** blocks everything — nobody can start without clean data

2. **Shrey's windowing utility (Day 2)** blocks all sequence models (Aditya's RNN, Shrey's LSTM/GRU, Umme's TCN)

3. **Aditya's evaluation module (Day 3)** blocks consistent scoring across all models

### Parallel Opportunities

- Days 4–8: All 4 members work on their models **independently**

- Days 9–11: Dhruv (analysis) + Shrey (viz) + Umme (report) can work in parallel

- Quantile variants reuse 90% of the standard model code (just swap the loss function)

---

## Final Deliverables

### Per-Member Deliverables

| Member | Must Submit                                                                                                                    |
| ------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Dhruv  | Clean dataset (train.csv, test.csv), AR/ARMA/ARIMA notebooks, TFT notebook, final comparison table, project coordination notes |
| Aditya | MLP notebook, RNN notebook, Quantile MLP notebook, Quantile RNN notebook, evaluation metrics module, error analysis            |
| Shrey  | LSTM notebook, GRU notebook, Quantile LSTM notebook, Quantile GRU notebook, windowing utility, all comparison visualizations   |
| Umme   | EDA notebook, TCN notebook, Quantile TCN notebook, degradation analysis, final report (with all sections integrated)           |

### Combined Final Output

1. **Consolidated Jupyter Notebook / set of notebooks** — all models runnable

2. **Final Report (PDF)** containing:
   - Introduction & motivation

   - Dataset description & EDA findings

   - Methodology (all models described)

   - Results comparison table (RMSE + NASA score for all 14+ models)

   - Visualizations (prediction plots, error distributions, quantile intervals)

   - TFT interpretability analysis (attention weights)

   - Conclusions & future work

3. **Model comparison summary table:**

| Model         | Type          | RMSE | NASA Score | Training Time | Notes |
| ------------- | ------------- | ---- | ---------- | ------------- | ----- |
| AR            | Classical     | —    | —          | —             | —     |
| ARMA          | Classical     | —    | —          | —             | —     |
| ARIMA         | Classical     | —    | —          | —             | —     |
| MLP           | Deep Learning | —    | —          | —             | —     |
| RNN           | Deep Learning | —    | —          | —             | —     |
| LSTM          | Deep Learning | —    | —          | —             | —     |
| GRU           | Deep Learning | —    | —          | —             | —     |
| Quantile MLP  | Quantile      | —    | —          | —             | —     |
| Quantile RNN  | Quantile      | —    | —          | —             | —     |
| Quantile LSTM | Quantile      | —    | —          | —             | —     |
| Quantile GRU  | Quantile      | —    | —          | —             | —     |
| TCN           | Advanced      | —    | —          | —             | —     |
| Quantile TCN  | Advanced      | —    | —          | —             | —     |
| TFT           | Advanced      | —    | —          | —             | —     |

---

## Risk Management

| Risk                                    | Likelihood | Impact                      | Mitigation                                                                                                                             |
| --------------------------------------- | ---------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Data pipeline delay (Day 1)**         | Low        | Critical — blocks everyone  | Dhruv prioritizes this above all else on Day 1. If stuck, use the already-existing `final_train.csv` as temporary fallback.            |
| **TFT too complex to implement**        | Medium     | Medium — it's a bonus model | Fallback: use `darts` library which has TFT built-in with minimal code. Or replace with a simpler Transformer encoder.                 |
| **Quantile models don't converge**      | Medium     | Low — they're variants      | Use standard pinball loss. If still failing, use quantile regression on top of standard model predictions.                             |
| **ARIMA doesn't fit well on this data** | High       | Low — it's expected         | Document why (non-stationarity, multivariate nature). This is a valid finding — classical models are expected to underperform DL here. |
| **Team member falls behind**            | Medium     | High                        | Day 8 checkpoint is critical. If someone is behind, redistribute remaining work. Day 13 buffer exists for exactly this reason.         |
| **GPU/compute limitations**             | Medium     | Medium                      | Use Google Colab (free GPU). Keep batch sizes reasonable. LSTM/GRU on this dataset size don't require heavy compute.                   |
| **Inconsistent evaluation**             | Low        | High — ruins comparison     | Aditya's shared evaluation module (Day 3) prevents this. Everyone uses the same function.                                              |

### Backup Strategies

- If a model completely fails: document the failure and analysis (this is still valuable in a report)

- If running out of time: prioritize having all standard models done; quantile variants are the first to cut

- If team coordination breaks down: all shared code goes into a single shared folder/repo by Day 3

---

## Shared Conventions (agree on Day 1)

1. **Column names**: `engine_id`, `cycle`, `op1`, `op2`, `op3`, `s1`–`s21`, `dataset_id`, `RUL`

2. **RUL cap**: 125 cycles

3. **Sequence window**: 30 cycles (for LSTM/GRU/RNN/TCN)

4. **Train/validation split**: 80/20 by engine_id (not by row — don't leak future data)

5. **Random seed**: 42 (for reproducibility)

6. **Evaluation**: always report RMSE and NASA score on test set

7. **Notebook naming**: `model_name.ipynb` (e.g., `lstm.ipynb`, `quantile_gru.ipynb`)
