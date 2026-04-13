# NASA CMAPSS Dataset — Complete Guide for Predictive Maintenance

## Context

Dhruv is building an Applied Forecasting / Predictive Maintenance project for IT 402 using the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. This plan provides a deep, structured explanation of the dataset and a roadmap for building ML models on it.

---

## 1. Dataset Structure Explanation

### Files in your `data/` directory

| File | What it is | Rows | Engines |

|------|-----------|------|---------|

| `train_FD001.txt` | Full run-to-failure trajectories | 20,631 | 100 |

| `train_FD002.txt` | Full run-to-failure trajectories | 53,759 | 260 |

| `train_FD003.txt` | Full run-to-failure trajectories | 24,720 | 100 |

| `train_FD004.txt` | Full run-to-failure trajectories | 61,249 | 248 |

| `test_FD001.txt` | Truncated trajectories (cut before failure) | 13,096 | 100 |

| `test_FD002.txt` | Truncated trajectories | 33,991 | 259 |

| `test_FD003.txt` | Truncated trajectories | 16,596 | 100 |

| `test_FD004.txt` | Truncated trajectories | 41,214 | 249 |

| `RUL_FD001.txt` | True RUL for each test engine (100 values) | 100 | — |

| `RUL_FD002.txt` | True RUL for each test engine (259 values) | 259 | — |

| `RUL_FD003.txt` | True RUL for each test engine (100 values) | 100 | — |

| `RUL_FD004.txt` | True RUL for each test engine (248 values) | 248 | — |

### What makes FD001–FD004 different?

| Dataset | Operating Conditions | Fault Modes | Complexity |

|---------|---------------------|-------------|------------|

| **FD001** | 1 (Sea Level) | 1 (HPC Degradation) | **Easiest** — start here |

| **FD002** | 6 (varying altitude, Mach, throttle) | 1 (HPC Degradation) | Medium — same fault, but operating conditions shift sensor baselines |

| **FD003** | 1 (Sea Level) | 2 (HPC + Fan Degradation) | Medium — two failure modes look different in sensor space |

| **FD004** | 6 | 2 (HPC + Fan Degradation) | **Hardest** — multiple conditions × multiple faults |

**HPC = High Pressure Compressor.** When its blades erode, the compressor loses efficiency, temperatures rise, and the engine eventually fails.

**Fan Degradation** = erosion/damage to the front fan blades, causing vibration and efficiency loss.

---

## 2. Row-Level Understanding — What Does One Row Mean?

Each file has **26 space-separated columns**, no header. The columns are:

| Col # | Name | Description |

|-------|------|-------------|

| 1 | `engine_id` | Which engine (1–100 in FD001). Think of it as a patient ID in a hospital. |

| 2 | `cycle` | The "heartbeat" — one takeoff-cruise-landing cycle. Like odometer clicks. |

| 3 | `op_setting_1` | Operational setting — likely related to **altitude** |

| 4 | `op_setting_2` | Operational setting — likely related to **Mach number** (speed) |

| 5 | `op_setting_3` | Operational setting — likely related to **throttle resolver angle** (TRA) |

| 6–26 | `sensor_1` to `sensor_21` | 21 sensor measurements from various engine locations |

### The 21 Sensors (based on the CMAPSS simulation model)

| Sensor | Symbol | Physical Meaning | Typical Unit |

|--------|--------|-----------------|--------------|

| s1 | T2 | Total temperature at fan inlet | °R |

| s2 | T24 | Total temperature at LPC outlet | °R |

| s3 | T30 | Total temperature at HPC outlet | °R |

| s4 | T50 | Total temperature at LPT outlet | °R |

| s5 | P2 | Pressure at fan inlet | psia |

| s6 | P15 | Total pressure in bypass-duct | psia |

| s7 | P30 | Total pressure at HPC outlet | psia |

| s8 | Nf | Physical fan speed | rpm |

| s9 | Nc | Physical core speed | rpm |

| s10 | epr | Engine pressure ratio (P50/P2) | — |

| s11 | Ps30 | Static pressure at HPC outlet | psia |

| s12 | phi | Ratio of fuel flow to Ps30 | pps/psi |

| s13 | NRf | Corrected fan speed | rpm |

| s14 | NRc | Corrected core speed | rpm |

| s15 | BPR | Bypass ratio | — |

| s16 | farB | Burner fuel-air ratio | — |

| s17 | htBleed | Bleed enthalpy | — |

| s18 | Nf_dmd | Demanded fan speed | rpm |

| s19 | PCNfR_dmd | Demanded corrected fan speed | rpm |

| s20 | W31 | HPT coolant bleed | lbm/s |

| s21 | W32 | LPT coolant bleed | lbm/s |

### Example Row Interpretation

From your data, first row of `train_FD001.txt`:

```

1 1 -0.0007 -0.0004 100.0 518.67 641.82 1589.70 1400.60 14.62 21.61 554.36 2388.06 9046.19 1.30 47.47 521.66 2388.02 8138.62 8.4195 0.03 392 2388 100.00 39.06 23.4190

```

**Translation:** Engine #1, at its very first flight cycle, is operating at sea level (op settings ≈ 0, 0, 100). The fan inlet temperature is 518.67°R (~59°F — standard sea level). HPC outlet temperature is 1589.70°R. Core speed is 9046 rpm. The engine is **brand new and healthy** at this point.

Compare to engine #100's last row (cycle 200): sensors show subtle shifts — HPC outlet temperature has drifted up, corrected fan speed has shifted — the engine is about to fail.

---

## 3. Conceptual Understanding

### What does "machine failure" mean here?

Failure = the engine's health has degraded to the point where it **cannot safely operate**. In this simulation, the C-MAPSS model injects a fault (e.g., HPC blade erosion) that grows progressively until a performance threshold is crossed. The last cycle in training data = the cycle of failure.

### What is Remaining Useful Life (RUL)?

**RUL = number of cycles left before failure.**

If an engine fails at cycle 200 and you're currently at cycle 150:

```

RUL = 200 - 150 = 50 cycles remaining

```

### How is failure simulated?

The NASA simulator:

1. Starts each engine with slightly different initial health (random manufacturing variation)

2. Injects a fault at a random point early in the engine's life

3. The fault **grows monotonically** — blade erosion gets worse, never better

4. Sensors reflect this degradation through gradual drift

5. When degradation crosses a threshold → the engine is "failed"

### Why does RUL decrease over time?

Because each cycle brings the engine one step closer to failure. It's a countdown timer:

```

Cycle 1:   RUL = 191  (engine will fail at cycle 192)

Cycle 2:   RUL = 190

Cycle 50:  RUL = 142

Cycle 192: RUL = 0    ← FAILURE

```

---

## 4. Time Series Nature

### Why is this a time-series dataset?

Each engine produces an **ordered sequence of observations** indexed by cycle number. The order matters — cycle 50's sensor readings are a consequence of what happened in cycles 1–49. You cannot shuffle the rows within an engine.

### How does each engine evolve?

```

Early cycles (healthy):     Sensors are stable, near baseline

Middle cycles (degrading):  Subtle drift begins — temperatures creep up, efficiency drops

Late cycles (near failure): Clear trends — temperatures elevated, pressures shifted

```

### What patterns indicate degradation?

- **Temperature sensors (s2, s3, s4)**: Trend upward — less efficient compression means more heat

- **Pressure sensors (s7, s11)**: May drop — compressor can't maintain pressure

- **Efficiency ratios (s12, s15)**: Shift as the engine compensates

- **Vibration-related (s8, s9)**: Speed fluctuations increase

- **Sensors s1, s5, s6, s10, s16, s18, s19**: Often nearly constant in FD001 (single operating condition) — these are **not useful** for FD001 but become important in FD002/FD004

---

## 5. Train vs Test Logic

### Training Data

Each engine runs **from healthy to failure**. The last row for each engine is its failure point.

- You can compute RUL for every row: `RUL = max_cycle_for_engine - current_cycle`

- This gives you complete labeled data for supervised learning

### Test Data

Each engine's time series is **truncated** — it stops at some random point **before failure**.

- You do NOT know when it will fail

- The `RUL_FD00X.txt` file gives the **true RUL at the last observed cycle** for each test engine

- Your job: predict that RUL value

### Visual:

```

Training Engine:  |====healthy====|===degrading===|==failing==|X (failure)

                  cycle 1                                      cycle 200



Test Engine:      |====healthy====|===degrading===|???

                  cycle 1                          cycle 150  (RUL=50, given in RUL file)

```

---

## 6. Forecasting Perspective

### As Time-Series Forecasting

- Each sensor is a time series per engine

- You could forecast future sensor values and detect when they cross a failure threshold

- Models: ARIMA per sensor, Prophet, LSTM sequence-to-sequence

### As Regression (most common approach)

- Input: a window of recent sensor readings (or engineered features from the full history)

- Output: a single number — predicted RUL

- Models: Random Forest, XGBoost, SVR, Neural Networks (LSTM, CNN-LSTM, Transformer)

- This is how most CMAPSS papers frame it

### As Survival Analysis

- Model the probability of surviving beyond time t given sensor history

- Output: a survival curve per engine

- Models: Cox Proportional Hazards, DeepSurv, Random Survival Forests

- Advantage: gives uncertainty estimates, not just point predictions

### As Classification (simplified)

- Binary: "Will this engine fail within the next N cycles?" (yes/no)

- Multi-class: "Is this engine in healthy / degrading / critical state?"

- Useful for triggering maintenance alerts

---

## 7. Data Challenges

### 1. Sensor Noise

The data is intentionally noisy (simulating real sensor imperfection). Raw sensor values fluctuate cycle-to-cycle even when the engine is healthy. **You must smooth the data** (rolling averages, exponential smoothing) to reveal underlying trends.

### 2. Redundant / Useless Features

In FD001 (single operating condition), sensors s1, s5, s6, s10, s16, s18, s19 are nearly constant — they carry **zero degradation information**. Including them adds noise without signal. Always check variance and correlation before modeling.

### 3. Different Operating Conditions (FD002, FD004)

When the engine operates at different altitudes/speeds, sensor baselines shift dramatically. Temperature at altitude ≠ temperature at sea level. **You must normalize per operating condition** or the model will confuse "operating at high altitude" with "degrading."

### 4. Multiple Failure Modes (FD003, FD004)

HPC degradation and fan degradation produce **different sensor signatures**. A single model must learn both patterns. Some sensors are diagnostic for one mode but not the other. This is why FD003/FD004 are harder.

### 5. Varying Engine Lifespans

Engine lifetimes range from ~130 to ~360+ cycles. This means "cycle 100" means very different things for a short-lived vs long-lived engine. **Raw cycle number is not a good feature** — prefer sensor-derived health indicators.

### 6. RUL Capping (Important Practical Decision)

In practice, predicting RUL = 300 vs RUL = 250 doesn't matter — both mean "the engine is fine." Most papers **cap RUL at 125 or 130 cycles**: any RUL above the cap is set to the cap value. This focuses the model on the critical degradation window.

---

## 8. Intuition Building — Analogies

### The Car Odometer Analogy

- **engine_id** = your car's license plate

- **cycle** = odometer reading (miles driven)

- **sensors** = dashboard gauges (temperature, oil pressure, RPM, fuel efficiency)

- **operational settings** = driving conditions (highway vs city, flat vs mountain, summer vs winter)

- **RUL** = how many more miles before your engine dies

- **Training data** = cars we drove until they broke down (we know exactly when)

- **Test data** = cars still on the road — "how much longer will this one last?"

### The Human Aging Analogy

- A newborn (cycle 1) has healthy vitals

- Over decades, blood pressure creeps up, cholesterol rises, lung capacity drops

- Some people live to 90, some to 70 — just like engines have different lifespans

- A doctor (your ML model) looks at current vitals + trend and predicts remaining lifespan

- The doctor doesn't just look at today's snapshot — they look at **how fast things are changing**

### The Phone Battery Analogy

- A new phone holds 100% capacity

- After 500 charge cycles, it holds 85%

- After 1000 cycles, it holds 70% — and degrades faster from here

- Sensors = battery health metrics (voltage, charge time, discharge rate)

- RUL = "how many more cycles before the battery can't hold enough charge?"

- The degradation is **nonlinear** — slow at first, accelerating near end-of-life (same as CMAPSS)

### Key Insight: The Signal is in the Trend, Not the Snapshot

A single row tells you almost nothing. An engine with T30 = 1590°R could be healthy or dying — it depends on whether T30 was 1580°R last week (normal fluctuation) or 1560°R last month (ominous upward trend). **Your model must capture temporal patterns.**

---

## 9. Next Steps — Implementation Roadmap

### Step 1: Data Loading & Exploration

```python

import pandas as pd

import numpy as np



cols = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

train = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=cols)

test = pd.read_csv('data/test_FD001.txt', sep=r'\s+', header=None, names=cols)

rul = pd.read_csv('data/RUL_FD001.txt', header=None, names=['RUL'])

```

- Plot sensor traces for a few engines (engine 1, 50, 100)

- Check which sensors have near-zero variance → drop them

- Visualize distributions of engine lifetimes

### Step 2: Compute RUL for Training Data

```python

# For each engine, max cycle = failure point

max_cycles = train.groupby('engine_id')['cycle'].max().reset_index()

max_cycles.columns = ['engine_id', 'max_cycle']

train = train.merge(max_cycles, on='engine_id')

train['RUL'] = train['max_cycle'] - train['cycle']

train.drop('max_cycle', axis=1, inplace=True)



# Cap RUL at 125 (piecewise linear)

train['RUL'] = train['RUL'].clip(upper=125)

```

### Step 3: Feature Engineering

- **Drop constant sensors** (s1, s5, s6, s10, s16, s18, s19 for FD001)

- **Normalize** remaining sensors (MinMax or StandardScaler per sensor)

- **Rolling statistics**: rolling mean, std, slope over last 5/10/20 cycles per sensor

- **For FD002/FD004**: cluster operating conditions first, then normalize within each cluster

### Step 4: Structure for ML Models

**For classical ML (XGBoost, RF, SVR):**

- Each row = one sample with features = current sensor values + rolling features

- Target = RUL (capped)

**For sequence models (LSTM, Transformer):**

- Each sample = a **window** of the last W cycles (e.g., W=30)

- Shape: `(num_samples, window_size, num_features)`

- Target = RUL at the last cycle in the window

### Step 5: Baseline Models (in order of complexity)

1. **Linear Regression** — sanity check baseline

2. **Random Forest / XGBoost** — strong tabular baseline, use rolling features

3. **LSTM** — captures sequential patterns, often the go-to for CMAPSS

4. **CNN-LSTM** or **Transformer** — if you want state-of-the-art

### Step 6: Evaluation Metrics

- **RMSE** — standard regression metric

- **NASA Scoring Function** (asymmetric):

  ```

  If predicted RUL < actual (late prediction):  score = e^(-d/13) - 1

  If predicted RUL > actual (early prediction): score = e^(d/10) - 1

  ```

  Late predictions are penalized MORE (predicting failure too late is dangerous). This is the official competition metric.

### Step 7: For Test Set Evaluation

- For each test engine, predict RUL at its last observed cycle

- Compare against `RUL_FD001.txt` ground truth

- Report RMSE and NASA score

---

## Key Files to Work With

- `data/train_FD001.txt` — start here (simplest subset)

- `data/test_FD001.txt` — evaluation data

- `data/RUL_FD001.txt` — ground truth for test

- `data/readme.txt` — official dataset description

- `data/Damage Propagation Modeling.pdf` — the original research paper (worth reading for deep understanding)

## Verification

- After computing training RUL: the last row of each engine should have RUL=0

- After capping: no RUL value should exceed 125

- Sensor variance check: in FD001, sensors s1, s5, s6, s10, s16, s18, s19 should have near-zero variance

- Test RUL file should have exactly as many entries as unique engines in test data (100 for FD001)
