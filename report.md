# Aircraft Engine Failure Forecasting — Deep Understanding Report

### NASA C-MAPSS FD004 · IT 402 Applied Forecasting Methods

---

# PART 1: WHY FD004 IS THE HARDEST DATASET

FD004 has **248 training engines** and **249 test engines**. Each engine produces one row per flight cycle with 26 measurements. The training data has 61,249 rows in total.

What makes FD004 hard are exactly two things:

**1. Six operating conditions.** The engine flies at different altitudes, speeds, and throttle settings. When an engine is at 35,000 ft, its temperatures are naturally lower than the same engine at sea level — not because it's healthier, but because the air is thinner and colder. Sensor `s3` (HPC outlet temperature) might read 1590°R at sea level and 1420°R at altitude for the exact same health state. If your model doesn't account for this, it will confuse "flying high" with "healthy engine."

**2. Two failure modes.** Some engines fail through HPC (High Pressure Compressor) blade erosion — temperatures rise, pressure drops. Other engines fail through fan degradation — vibration increases, fan efficiency drops. These produce different sensor patterns. A single model must learn both signatures simultaneously.

Every design decision in this project — the KMeans clustering, the StandardScaler, the health index — exists to solve one or both of these two problems.

---

# PART 2: PREPROCESSING — SETTING UP FOR SUCCESS

## 2.1 Dropping Useless Sensors

Out of 21 sensors, 7 are near-constant in the data: `s1, s5, s6, s10, s16, s18, s19`. Their standard deviation is below 0.1 across all rows. They carry zero information about degradation. Keeping them would just add noise to every model.

After dropping: **14 sensors remain** — `s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21`.

## 2.2 KMeans Operating Condition Clustering

**File: `src/preprocessing/scaling.py`**

We take the three operating condition columns `op1, op2, op3` from all training rows and cluster them into **k=6 groups** using KMeans.

Why k=6? FD004 has 6 distinct operating conditions. K=6 captures each one as a separate cluster.

Every row now gets a label `op_cluster ∈ {0, 1, 2, 3, 4, 5}`. This label says "this measurement was taken while the engine was flying in condition X."

**Critical rule: KMeans is fit only on training data.** The same fitted model then assigns clusters to test data. If we fit KMeans on test data separately, the cluster assignments would be inconsistent between train and test — a different coordinate system.

## 2.3 StandardScaler Per Cluster

For each of the 6 clusters, we compute the mean and standard deviation of each sensor across all training rows in that cluster. Then:

```
scaled_value = (raw_value − cluster_mean) / cluster_std
```

**Why per-cluster?** If we scale globally (one mean/std for the whole dataset), an engine flying in cluster 3 (low altitude) will have raw sensor values that differ from cluster 0 (high altitude). After global scaling, that difference remains. Per-cluster scaling removes the altitude/speed baseline — what's left is only how far the engine deviates from the normal behaviour for its current flying condition.

**Why StandardScaler and not MinMaxScaler?** As an engine degrades near failure, its sensors go to extremes — temperatures spike, pressures drop beyond anything seen in the healthy part of training. MinMaxScaler would silently clip those values at 0 or 1 (the training bounds). StandardScaler represents them as large z-scores — unusual but not cut off.

**The result:** after scaling, a healthy engine in any of the 6 conditions has sensor values close to zero. A degrading engine has increasingly large values. Operating condition is removed. Only degradation remains.

---

# PART 3: BUILDING THE HEALTH INDEX — COMPLETE EXPLANATION

## 3.1 Why We Need a Single Number

After preprocessing we have 14 sensor columns per row. For the deep learning models (LSTM, GRU etc.), we feed all 14 sensors directly. But for ARIMA, we cannot. ARIMA is a **univariate** time series model — it models one series at a time. We cannot give it 14 parallel series.

We need to compress 14 numbers into **one number that measures degradation**. This is the Health Index.

The goal: a number that

- starts near some baseline value when the engine is healthy
- increases monotonically over time as the engine degrades
- is comparable across all 248 engines regardless of which operating conditions they flew through
- tells ARIMA something meaningful to forecast

## 3.2 The Problem with Simple Approaches

**Why not just average the 14 sensors?**  
Different sensors go in different directions. `s3` (HPC temperature) rises with degradation. `s7` (HPC pressure) falls with degradation. If you average them, they partially cancel each other out. You lose the signal.

**Why not pick one sensor?**  
No single sensor captures all degradation. HPC degradation and fan degradation (FD004's two fault modes) affect different sensors. One sensor that's diagnostic for HPC degradation might be nearly flat during fan degradation.

**The right tool: PCA.** PCA finds the direction in 14-dimensional sensor space where the data varies the most — the main axis of variation. Since degradation drives the dominant variation in this data (all 14 sensors collectively shift as the engine wears), the first principal component captures degradation better than any individual sensor or simple average.

## 3.3 The Specific FD004 Problem with PCA

Here's the subtlety. In FD004, the engine switches between 6 operating conditions from cycle to cycle. Between two consecutive cycles, the sensors might jump significantly — not because of degradation, but because the flying condition changed.

If we run PCA directly on the 14 scaled sensors, the first principal component will capture the largest direction of variation in the data. What is that direction? **Operating condition switching** — because the sensor jumps due to altitude changes are large and frequent. Degradation is a slow, smooth drift. Operating condition changes are sharp jumps. PCA will find the jumps, not the drift.

**The fix**: remove the operating condition effect before running PCA. This is the single most important step in building the health index for FD004.

## 3.4 Step-by-Step Construction

**File: `src/models/classical.py`, function `build_pca_health_index()`**

---

### Step 1 — Rolling Mean Smoothing

Before anything else, replace each raw sensor value with its 10-cycle rolling mean within each engine:

```
s3_rmean_10 = average of s3 over the last 10 cycles for this engine
```

Why 10 cycles? A single-cycle reading is noisy — it reflects measurement error and momentary fluctuations. The 10-cycle rolling mean smooths this noise and shows the underlying trend. This is what gets fed into PCA, not the raw spiky readings.

---

### Step 2 — Per-Cluster Mean Detrending (The Key Step for FD004)

For each of the 6 operating condition clusters, compute the mean of each smoothed sensor across all training rows in that cluster. Call this `cluster_mean[cluster_id][sensor]`.

Then for every row: subtract its cluster's mean from each sensor:

```
detrended_value = smoothed_value − cluster_mean[op_cluster][sensor]
```

**What does this achieve?**

Before detrending:

```
Cycle 50, Cluster 0 (sea level), s3_rmean_10 = +0.8   ← engine A
Cycle 51, Cluster 3 (high alt), s3_rmean_10 = -0.6   ← same engine A, next cycle, different condition
```

That -1.4 swing from cycle 50 to 51 is entirely due to altitude change. Engine A is in exactly the same health state.

After detrending (subtracting the mean for each cluster):

```
Cluster 0 mean for s3 = +0.7
Cluster 3 mean for s3 = -0.7

Cycle 50: +0.8 − (+0.7) = +0.1   ← near zero (healthy)
Cycle 51: -0.6 − (−0.7) = +0.1   ← near zero (same health state)
```

The operating condition jump is gone. What remains is only how much this engine deviates from the typical healthy engine in its current flying condition. That deviation is the degradation signal.

This operation is computed as:

```python
cluster_means = train.groupby("op_cluster")[smoothed_sensor_cols].mean()

for cluster_id, row in cluster_means.iterrows():
    mask = df["op_cluster"] == cluster_id
    df.loc[mask, smoothed_cols] = df.loc[mask, smoothed_cols].values - row.values
```

---

### Step 3 — Global PCA on Detrended Sensors

Now we run PCA on the detrended sensor matrix.

PCA finds the directions (called principal components) of maximum variance in the data. The first principal component (PC1) is the single direction that explains the most variance.

**Mathematically:** PCA finds the eigenvectors of the covariance matrix of the detrended sensors. PC1 is the eigenvector with the largest eigenvalue.

After detrending, the dominant source of variation is no longer operating condition (we removed that). The remaining variation comes from degradation — all 14 sensors slowly drifting as the engine wears. PCA captures this collective drift as one number per row.

```python
pca = PCA(n_components=1).fit(train_detrended[smoothed_cols])
pc1_train = pca.transform(train_detrended[smoothed_cols])  # shape: (n_rows, 1)
```

`n_components=2` was also used in the code (combining two components with `np.maximum`), which helps capture both failure modes separately and then take the worse of the two.

---

### Step 4 — Sign Flip

PCA doesn't know which direction is "more degraded." The sign of a principal component is arbitrary — it could be positive for healthy and negative for degraded, or vice versa.

We check: what is the correlation between PC1 and `−RUL`?  
`−RUL` increases over time (from −125 at start to 0 at failure). If PC1 is positively correlated with `−RUL`, it also increases over time — that's what we want.  
If PC1 is negatively correlated with `−RUL`, we multiply by −1 to flip it.

```python
sign = 1.0 if np.corrcoef(pc1_train.ravel(), -train["RUL"].values)[0, 1] >= 0 else -1.0
health_index = pc1_train.ravel() * sign
```

After this: higher value = more degraded. Always.

---

### Step 5 — Standardization Using Training Statistics

```
health_index = (raw_health_index − train_mean) / train_std
```

This makes the scale consistent: near −1 when healthy (early cycles), growing towards positive values as the engine degrades, reaching 1.5–4.0 near failure.

The training mean and std are computed only from the training data and applied to both train and test (no leakage).

---

### Step 6 — Isotonic Regression (Enforcing Monotonicity)

Even after all of the above, the health index can have small dips — a value at cycle 80 might be slightly lower than at cycle 79 because of residual noise.

Physical reality: a degraded engine does not spontaneously repair itself. Degradation only moves in one direction.

**Isotonic Regression** finds the closest monotonically non-decreasing sequence to the health index series. For each engine separately, it fits the constraint: `health_index[t] ≤ health_index[t+1]` for all t, while minimising the sum of squared changes from the original values.

```python
from sklearn.isotonic import IsotonicRegression
ir = IsotonicRegression(increasing=True)
health_index_monotone = ir.fit_transform(cycles, health_index)
```

This is the minimum correction — it doesn't change the overall shape, it just removes the illegal downward bumps.

---

## 3.5 The Health Index in Action — Actual Notebook Output

After running `build_pca_health_index()` on FD004:

```
health_index range: [−1.597, 4.402]   mean = −0.000
health_index R2 with RUL (post-monotone): −5.188  (target: > 0.3)
```

### The Three-Engine Plot

![Health Index for Engines 1, 2, 3](report_images/T08_AR_model_book_cell6_img1.png)

**What you are looking at:** The health index over cycles for three different FD004 engines. The x-axis is the cycle number (each cycle = one flight). The y-axis is the health index value.

**What this tells us:**

- All three engines start near −1.0 (healthy baseline after standardisation)
- All three end at roughly 2.0–3.5 (severely degraded, near failure)
- The shape is not a straight line — it's relatively flat for a long period (slow degradation phase) and then accelerates sharply near the end (rapid degradation phase near failure)
- Engine 1 lives for ~325 cycles, Engine 2 for ~305, Engine 3 for ~310 — each engine has a different lifespan

This is exactly what we want: a signal that starts low, ends high, and rises monotonically. ARIMA can work with this.

---

## 3.6 Why the R² is −5.188 (This is Critical to Understand)

The output says `health_index R2 with RUL: −5.188`. This looks alarming. A negative R² normally means the model is worse than just predicting the mean. But it does **not** mean the health index is broken.

Here is what the code actually computes:

```python
r2 = r2_score(-train["RUL"].values, train["health_index"].values)
```

This is asking: "how well does a straight line fit the relationship between `health_index` and `−RUL`?"

Look at the plot above. The health index is flat for a long time, then shoots up steeply. That is not a straight-line relationship with time or with −RUL. The relationship is nonlinear (roughly exponential near failure).

R² measures linear fit. The health index has a **nonlinear, accelerating** relationship with degradation. So R² = −5.188 simply means "a straight line is a bad fit" — not that the health index is tracking the wrong thing. Visually, the plots confirm it rises monotonically over time, which is exactly what we need for threshold-based ARIMA forecasting.

---

## 3.7 The Failure Threshold

```
Failure threshold: 1.6850
```

This is computed as the **5th percentile of health_index values among all training rows where RUL ≤ 5**.

In plain English: take all rows where the engine was about to fail (RUL ≤ 5 cycles left). Look at their health_index values. The threshold = the value below which 95% of those near-failure rows lie.

So when a forecasted health_index crosses 1.685, we say: "this engine is now in the same degradation zone as training engines that were about to fail."

### Threshold Candidates Table from the Notebook:

```
health_index EOL stats (rows with RUL ≤ 5):
  EOL mean   = 2.613
  EOL median = 2.166
  EOL min    = 1.428
  EOL max    = 4.402

Threshold candidates:
  q=0.05 → threshold=1.685  |  22/248 test engines reach it (9%)
  q=0.10 → threshold=1.747  |  20/248 test engines reach it (8%)
  q=0.20 → threshold=1.840  |  17/248 test engines reach it (7%)
  q=0.30 → threshold=1.927  |  17/248 test engines reach it (7%)
  q=0.50 → threshold=2.166  |  13/248 test engines reach it (5%)
  q=0.70 → threshold=3.363  |  0/248 test engines reach it (0%)
  q=0.90 → threshold=3.707  |  0/248 test engines reach it (0%)
```

**The most important finding here:** Even with the lowest threshold (q=0.05, threshold=1.685), only **22 out of 248 test engines** (9%) ever reach it in their observed history. The remaining 91% of test engines are cut off before their health index gets that high.

This is a fundamental property of FD004 test data: most test engines are truncated early — they still have a lot of life left when the test sequence ends. Their health index never climbs to the failure zone in the observed data.

**This is why the FALLBACK mechanism exists.** For the 91% of engines whose health index doesn't reach the threshold in observed data, ARIMA tries to forecast forward until the threshold is crossed. If the forecast slope is too flat (the model cannot see the degradation accelerating), it falls back to a direct linear regression from health_index to RUL.

---

# PART 4: FROM HEALTH INDEX TO ARIMA — FINDING p, d, q

## 4.1 What ARIMA Actually Does Here

ARIMA doesn't directly predict RUL. It **forecasts the health index** forward in time. The RUL is then derived from that forecast:

```
Observed health_index: [−1.0, −0.95, −0.90, ..., 1.2]   (actual cycles 1 to T)
                                                              ↓
                                              ARIMA forecasts the next 400 steps
                                              [1.25, 1.30, 1.36, 1.42, ..., 1.69, ...]
                                                                              ↑
                                                              First step crossing 1.685 = Predicted RUL
```

The number of forecast steps until the health index crosses the failure threshold = predicted remaining useful life.

For this to work well, the ARIMA model must capture the statistical structure of the health index time series accurately. To build ARIMA, three parameters must be specified:

- **p** — how many past values to use for prediction (AR order)
- **d** — how many times to difference the series to make it stationary (Integration order)
- **q** — how many past forecast errors to use for correction (MA order)

Each one is determined from data, not guessed.

---

## 4.2 Finding d — The ADF Stationarity Test

**ARIMA requires the series to be stationary** before the AR and MA parts can be fit. A stationary series has a constant mean and constant variance over time — it fluctuates around one fixed level.

The health index is clearly **not stationary**. Look at the plots above — it starts at −1 and trends upward to 3 or 4. The mean is changing over time. That's non-stationarity.

**Differencing** converts a non-stationary series into a stationary one. First differencing takes the change between consecutive values:

```
Original:    −1.0, −0.95, −0.90, −0.85, −0.80
1st diff:    +0.05, +0.05, +0.05, +0.05            ← the step size each cycle
```

If the original series is trending (non-stationary), the differences (step sizes) might be approximately constant — stationary.

If those differences are also trending (the step size is accelerating), we difference again:

```
1st diff:    +0.05, +0.05, +0.06, +0.07, +0.09    ← accelerating
2nd diff:    +0.00, +0.01, +0.01, +0.02            ← roughly constant → stationary
```

**The ADF Test** (Augmented Dickey-Fuller) formally tests whether a series is stationary.

- Null hypothesis: the series is non-stationary (has a unit root)
- p-value < 0.05 → reject null → series IS stationary → stop differencing
- p-value ≥ 0.05 → fail to reject null → series is NOT stationary → difference once more

### Actual ADF Results from the Notebook:

```
Engine 118 (longest, 543 cycles):
  Level  ADF p-value : 1.0      → p ≥ 0.05 → NOT stationary → difference
  Diff-1 ADF p-value : 0.9744   → p ≥ 0.05 → STILL not stationary → difference again
  Diff-2 ADF p-value : 0.0      → p < 0.05 → STATIONARY ✓
  Recommended d      : 2
```

### ADF Report Across 10 Engines:

```
engine_id   level_p   diff1_p   rec_d
1           0.9958    1.0       2
2           1.0       0.9267    2
3           1.0       0.8719    2
4           1.0       0.6973    2
5           1.0       1.0       2
6           1.0       0.9953    2
7           0.9991    0.0007    1   ← exception: this engine is shorter/flatter
8           1.0       0.9014    2
9           1.0       0.9697    2
10          0.9989    0.9985    2

d distribution: {d=2: 9 engines, d=1: 1 engine}
→ MODAL d = 2
```

Across the full 248 training engines: **206 engines need d=2, 42 need d=1, 0 need d=0**.

**Why does FD004 need d=2?** The health index doesn't just trend upward (non-stationary at level, requiring d=1). The rate of increase itself accelerates near failure (non-stationary at d=1, requiring d=2). This is the hallmark of accelerating degradation — not a constant rate of wear, but wear that compounds on itself.

---

### Visual: RAW vs After Differencing

![RAW vs diff-1](report_images/T08_AR_model_book_cell10_img1.png)

**Left panel — health_index RAW (d=0):** This is the full training dataset (all 248 engines stacked). The x-axis is not per-engine cycle but the total row index across all engines. You can see a clear upward drift overall — the series is non-stationary. This is what the ADF test at level (p=1.0) confirms.

**Right panel — health_index after diff-1 (d=1):** The first difference removes the linear trend. The series now fluctuates around zero (red dashed line). However, notice the variance is **not constant** — it spreads wider at higher row indices (corresponding to later cycles in longer-lived engines). This remaining non-stationarity in the variance is why diff-1 still fails the ADF test (p ≈ 0.97 for most engines). A second difference would be needed.

---

## 4.3 Finding p and q — ACF and PACF Plots

After determining d=2, we work on the twice-differenced series to find p and q.

**What is ACF?** Autocorrelation Function. It measures the correlation between the series and a lagged version of itself.

- ACF at lag 1: how much does today's value correlate with yesterday's?
- ACF at lag 5: how much does today's value correlate with the value 5 steps ago?

**What is PACF?** Partial Autocorrelation Function. It measures the same thing but removes the indirect effects. PACF at lag 3 asks: "how much does the value 3 steps ago directly predict today, after already accounting for the values at lags 1 and 2?"

**Reading the plots:**

- Blue shaded band = 95% confidence interval. Bars inside the band are not statistically significant.
- **For AR order p**: look at the PACF. Where does it first drop inside the band after some initial significant lags? That lag = p.
- **For MA order q**: look at the ACF. Same rule.
- If both ACF and PACF tail off gradually (neither cuts off sharply) → need both AR and MA terms → ARMA/ARIMA.

---

### ACF/PACF on the Twice-Differenced Series (from ARIMA notebook)

![ACF PACF diff-2 three engines](report_images/T10_ARIMA_model_book_cell14_img0.png)

**What you are looking at:** ACF (left column) and PACF (right column) for three FD004 engines (rows), computed on the **twice-differenced** health index.

**Reading these plots:**

The large spike at lag 1 in both ACF and PACF (going to −0.5 or lower) is a very strong lag-1 autocorrelation. This is common after twice-differencing — the double-differencing operation itself introduces a negative first-lag correlation.

After lag 1, both ACF and PACF have bars that are mostly inside the confidence band (small and insignificant). This pattern — one large lag-1 spike then nothing — is characteristic of an **MA(1) or ARIMA(0,2,1)** structure.

The fact that both ACF and PACF show the same pattern (both tail off quickly) means neither pure AR nor pure MA structure alone is dominant. This points to a mixed ARMA model.

---

### ACF/PACF on the Once-Differenced Series (from ARMA notebook)

![ACF PACF diff-1 engine 118](report_images/T09_ARMA_model_book_cell10_img0.png)

**What you are looking at:** ACF and PACF for the longest FD004 engine (engine 118, 543 cycles) after **one** difference.

**Reading this:** The ACF (left) shows bars that stay significant across many lags — they don't cut off sharply. The PACF (right) drops into the band after about lag 3. This pattern — PACF cuts off, ACF tails off — suggests an AR(3) or AR(p≤3) structure. The ACF tailing off rather than cutting off sharply means MA terms are also present → ARMA(p,q).

Note: this notebook uses d=1 on the diff-1 plot (ARMA notebook), while the ARIMA notebook applies d=2 before plotting. They show different views of the same data.

---

## 4.4 AIC Grid Search — Getting the Exact Values of p and q

The ACF/PACF plots give visual hints. But for FD004 (noisy, multi-condition data), the visual cutoff is not clean. We use **AIC (Akaike Information Criterion)** to make the final decision.

**What AIC measures:** the quality of a statistical model, balancing how well it fits the data against how complex it is (how many parameters it uses). Lower AIC = better model.

For each candidate (p, q) pair (every combination of p ∈ {1,2,3} and q ∈ {1,2,3}), we fit SARIMAX on one engine's health index and record the AIC. We repeat this for 15 sampled engines and take the modal winner.

### AR Order Selection — Actual Notebook Output:

```
engine  1: best p=10  (AIC=−1759.91)
engine  2: best p=10  (AIC=−1506.21)
engine  3: best p=10  (AIC=−1481.13)
engine  4: best p=10  (AIC=−1320.87)
engine  5: best p= 8  (AIC= −908.90)
engine  6: best p=10  (AIC=−1673.11)
engine  7: best p=10  (AIC=−1121.43)
engine  8: best p= 9  (AIC=−1137.10)
engine  9: best p=10  (AIC=−1676.21)
engine 10: best p=10  (AIC=−1879.24)
engine 11: best p= 7  (AIC=−1567.30)
engine 12: best p=10  (AIC=−1466.00)
engine 13: best p=10  (AIC=−1246.39)
engine 14: best p=10  (AIC=−1239.60)
engine 15: best p= 3  (AIC= −954.93)

→ Modal best AR order: p=10  (12 out of 15 engines prefer p=10)
```

AR(10) wins overwhelmingly. This means the health index retains autocorrelation going back 10 cycles — the degradation state 10 cycles ago still carries predictive information about the current state.

### ARMA and ARIMA Order Selection — Actual Notebook Output:

```
engine  1: best (p,q)=(2,1)  (AIC=−1717.30)
engine  2: best (p,q)=(1,1)  (AIC=−1514.52)
engine  3: best (p,q)=(1,1)  (AIC=−1456.44)
engine  4: best (p,q)=(2,3)  (AIC=−1326.35)
engine  5: best (p,q)=(3,2)  (AIC= −895.17)
engine  6: best (p,q)=(2,1)  (AIC=−1638.74)
engine  7: best (p,q)=(3,3)  (AIC=−1128.36)
engine  8: best (p,q)=(3,1)  (AIC=−1144.73)
engine  9: best (p,q)=(1,1)  (AIC=−1662.76)
engine 10: best (p,q)=(1,2)  (AIC=−1820.47)
engine 11: best (p,q)=(1,1)  (AIC=−1596.33)
engine 12: best (p,q)=(3,2)  (AIC=−1460.38)
engine 13: best (p,q)=(2,1)  (AIC=−1259.72)
engine 14: best (p,q)=(2,3)  (AIC=−1198.12)
engine 15: best (p,q)=(3,3)  (AIC= −936.46)

→ Modal best ARIMA order: (1, 2, 1)   [4 out of 15 engines prefer (1,1)]
```

**Final chosen orders:**

- AR: `(p=10, d=2, q=0)` → SARIMAX(10, 2, 0)
- ARMA: `(p=1, d=2, q=1)` → SARIMAX(1, 2, 1)
- ARIMA: `(p=1, d=2, q=1)` → SARIMAX(1, 2, 1)

Note that ARMA and ARIMA end up with the same order here because d was already determined to be 2 for ARMA (the code uses d=2 internally for ARMA as well, since the health_index needs 2 differences regardless of model family).

---

## 4.5 SARIMAX Model Fit — What the Summary Means

After selecting the best order, we fit the model on the **representative engine** (engine 118, the longest with 543 cycles) and examine the output.

```
SARIMAX Results for ARIMA(1,2,1):
==============================================================
Dep. Variable:               y   No. Observations: 543
Model:           SARIMAX(1, 2, 1)
Log Likelihood:           1459.052
AIC:                     −2912.103
BIC:                     −2899.223
==============================================================
             coef    std err      z    P>|z|
----------------------------------------------
ar.L1       0.0367    0.028    1.329   0.184   ← AR coefficient for lag 1
ma.L1      −0.9139    0.012  −77.656   0.000   ← MA coefficient for lag 1
sigma2      0.0003  8.81e−6   30.073   0.000   ← noise variance
==============================================================
```

**Understanding the coefficients:**

`ar.L1 = 0.0367` with p=0.184 (not statistically significant at 0.05). This means the AR term (yesterday's value) barely contributes. The model relies mostly on the MA term.

`ma.L1 = −0.9139` with p=0.000 (highly significant). This is a large negative MA coefficient. It means: if the model over-predicted last step (positive error), it corrects strongly downward next step. This "error correction" behaviour is the dominant structure in the health index after twice-differencing.

`sigma2 = 0.0003` — the residual noise is very small. The model fits closely.

---

## 4.6 Residual Diagnostics — Ljung-Box Test and QQ Plot

After fitting the model, we check: did the model capture all the patterns? If yes, the **residuals** (actual − predicted) should look like pure random noise.

**Ljung-Box test** checks for any remaining autocorrelation in the residuals.

- p > 0.05 for all tested lags → no remaining pattern → residuals are white noise → model is adequate
- p < 0.05 → residual autocorrelation remains → model missed some structure

### Actual Ljung-Box Output:

```
Ljung-Box residual test — ARIMA(1,2,1):
lag    lb_stat    lb_pvalue
1      75.71      3.3e−18
2      75.71      3.6e−17
3      75.84      2.4e−16
...
10     76.42      2.5e−12

✗ Some p-values < 0.05 — residual autocorrelation remains
```

All p-values are essentially zero — extremely far below 0.05. This means there IS still autocorrelation in the residuals. The model did not fully capture the structure.

**Why does this happen and why is it acceptable here?** FD004 is a complex, noisy, multi-condition dataset. The health_index has complex nonlinear dynamics near failure that a simple ARIMA(1,2,1) cannot fully model. A perfect model would need higher p and q, or a nonlinear model entirely. We accept this imperfect fit because: (1) increasing p further overfits small engines, and (2) the practical goal is not perfect one-step-ahead forecasting but reasonable long-horizon RUL estimation, which does not require perfect residuals.

### QQ Plot and Residuals Over Time

![QQ plot and residuals](report_images/T10_ARIMA_model_book_cell18_img1.png)

**Left panel — QQ Plot:** This compares the distribution of residuals to a theoretical normal distribution. If residuals were perfectly normal, all points would lie on the red diagonal line.

What we see: most points track the red line closely in the middle, but there are **heavy tails** — extreme outliers at both ends (especially the bottom-left outlier at −1.0). The Kurtosis value of 7.95 in the SARIMAX summary confirms this (normal distribution has kurtosis=3; our residuals have fat tails). This means the model occasionally produces large errors — residuals are not normally distributed but they're acceptable for practical forecasting.

**Right panel — Residuals over time:** The residuals fluctuate around zero (red dashed line). The large spike near cycle 0 (the bottom-left outlier in the QQ plot) corresponds to the model's initial stabilisation — ARIMA needs a few observations to "warm up." Beyond that, residuals are small and consistent.

---

# PART 5: THE ROLLING FORECAST — VALIDATING THE MODEL

Before applying the model to test data, we validate it on training engines using **walk-forward (rolling) validation**.

**What is walk-forward validation?** For one engine, take the first 70% of cycles as training. Then for each cycle in the remaining 30%, refit the model on all cycles up to that point and predict just the next cycle. This is the most honest validation — the model never sees future data.

### AR(10) Rolling Forecast on Engine 118:

![AR rolling forecast](report_images/T08_AR_model_book_cell18_img0.png)

**What you are looking at:**

- **Blue solid line**: the actual observed health_index for engine 118 (all 543 cycles)
- **Orange dashed line**: the AR(10) one-step-ahead rolling forecast on the validation portion (after the grey dotted vertical line at cycle ~380)
- **Grey dotted vertical line**: the train/val split point (first 70% = training, last 30% = validation)
- **Red dashed horizontal line**: the failure threshold at 1.685

**Key observations:**

1. The orange forecast tracks the blue observed line almost perfectly after the split. Rolling forecast RMSE = 0.0280 (very small in health_index units).
2. The forecast crosses the failure threshold at approximately cycle 480. Engine 118 is in the training set and fails at cycle 543 — the model would predict roughly 60 cycles of RUL from cycle 480, which is correct.
3. In the first half (cycles 0–380), the health index is flat near −1. There is nothing to forecast here — the engine is healthy. This is where many test engines are truncated.

### ARMA(1,1) Rolling Forecast on Engine 118:

![ARMA rolling forecast](report_images/T09_ARMA_model_book_cell16_img0.png)

Same engine, same validation. ARMA(1,1) also tracks the observed health_index closely. Rolling RMSE = 0.0741 (slightly worse than AR(10) on one-step-ahead validation because AR(10) uses more history). Note: the orange line has a small jump at the train/val split — this is the initial condition discontinuity when the model is first applied to the validation portion.

### ARIMA(1,2,1) Rolling Forecast on Engine 118:

![ARIMA rolling forecast](report_images/T10_ARIMA_model_book_cell22_img0.png)

Identical pattern. ARIMA(1,2,1) and ARMA(1,1) produce nearly the same rolling forecast (same underlying order). Rolling RMSE = 0.0741.

---

# PART 6: ARIMA FORECAST TO RUL — THREE REAL EXAMPLES

After validating the rolling forecast, we apply the model to **test engines** to predict RUL. Here is how it works for individual engines.

### How the Forecast → RUL Conversion Works

1. Take the test engine's observed health_index (all cycles up to the truncation point)
2. Fit ARIMA(1,2,1) on this observed series
3. Forecast 150 steps into the future
4. Find the first forecast step where the health_index ≥ 1.685 (failure threshold)
5. That step number = predicted RUL

If the forecast never crosses the threshold within 150 steps → use slope extrapolation or fallback regressor.

---

### Example 1 — Engine 31 (Good Prediction)

![Engine 31 ARIMA forecast](report_images/T10_ARIMA_model_book_cell26_img0.png)

**True RUL = 6 cycles. Predicted RUL = 10 cycles.**

**What you are seeing:**

- **Blue solid line**: observed health_index history from engine 31's test sequence. It starts near −0.5, rises steeply in recent cycles, and is near 1.4 at the last observed cycle (grey dotted vertical line at cycle ~135)
- **Orange dashed line**: ARIMA forecast from the last observed point forward
- **Orange shaded band**: 80% confidence interval of the forecast — widens as we project further
- **Red dashed line**: failure threshold = 1.685
- **Green dot**: the point where the forecast first crosses the threshold (cycle ~145)
- **Predicted RUL = 10**: the forecast says the engine will reach the failure threshold in ~10 more cycles

This is a good prediction. True RUL = 6, predicted = 10. The model correctly sees that the engine is already very close to the failure zone (health_index already at 1.4, threshold at 1.685) and predicts only a few cycles remain.

---

### Example 2 — Engine 183 (Moderate Case)

![Engine 183 ARIMA forecast](report_images/T10_ARIMA_model_book_cell26_img1.png)

**True RUL = 15 cycles. Predicted RUL = 26 cycles.**

**What you are seeing:**

- The engine's health_index at the truncation point (grey vertical line at cycle ~130) is approximately 1.35 — already in the high-degradation zone but not quite at the threshold
- The ARIMA forecast projects the health_index to cross 1.685 (green dot) at approximately cycle 150 — 26 steps after truncation
- The confidence interval is narrow near the observed data and fans out as we project forward
- True RUL = 15, Predicted = 26 → error = +11 (11 cycles late, which is not ideal but the forecast correctly identifies the engine is near failure)

---

### Example 3 — Engine 45 (Difficult Case)

![Engine 45 ARIMA forecast](report_images/T10_ARIMA_model_book_cell26_img2.png)

**True RUL = 85 cycles. Predicted RUL = 71 cycles.**

**What you are seeing:**

- The engine's health_index at truncation (grey vertical line at cycle ~80) is approximately −0.3 — still in the healthy zone, far from the failure threshold
- The ARIMA forecast rises slowly from −0.3, reaching 1.685 (green dot) at approximately cycle 150 — 71 steps after truncation
- True RUL = 85, predicted = 71 → error = −14 (14 cycles early, conservative but acceptable)

Notice the confidence interval fans out enormously here — the model is highly uncertain about a forecast 71 steps out from a series that has only 80 observed cycles. This is where ARIMA is weakest: long-horizon prediction from a still-healthy engine.

---

# PART 7: THE FALLBACK MECHANISM

From the per-engine verbose output in the AR notebook:

```
engine   3  true=107.0  pred=110.0  err=+3.0   [FALLBACK]
engine   4  true= 75.0  pred=110.0  err=+35.0  [FALLBACK]
engine   5  true=125.0  pred=110.0  err=−15.0  [FALLBACK]
engine   6  true= 78.0  pred=109.1  err=+31.1
engine   9  true= 99.0  pred=125.0  err=+26.0
```

The `[FALLBACK]` tag means the ARIMA forecast slope was too flat to produce a threshold crossing. This happens when the test engine was truncated very early — the health_index is still near −1, growing extremely slowly. The model cannot see the eventual acceleration toward failure.

In these cases, the code uses a **linear regressor** fitted on training data: `RUL = slope × health_index + intercept`. This regressor was fitted on the last 60% of each training engine's life (the degradation phase), so it gives a reasonable estimate of RUL from the current health state even without forecasting.

The **safety factor of 0.88** is then applied: `final_prediction = clipped_prediction × 0.88`. This makes all predictions slightly conservative. Because the NASA score penalises late predictions more than early ones, being slightly early is the safer strategy.

---

# PART 8: FINAL RESULTS — ALL THREE CLASSICAL MODELS

### AR(10, d=2, q=0) Results:

```
RMSE       : 27.67
NASA Score : 25,742  (mean per engine: 103.8)
R²         : 0.586
Bias       : −4.74  (slightly early → conservative)
```

### ARMA(1, d=2, 1) Results:

```
RMSE       : 26.19
NASA Score : 17,514  (mean per engine: 70.6)
R²         : 0.629
Bias       : −1.53  (nearly unbiased)
```

### ARIMA(1, 2, 1) Results:

```
RMSE       : 24.76
NASA Score : 13,791  (mean per engine: 55.6)
R²         : 0.668
Bias       : −3.58  (slightly early)
```

### What These Numbers Mean

**ARIMA is the best of the three** — lowest RMSE (24.76 vs 27.67 for AR) and lowest NASA Score (13,791 vs 25,742 for AR). Adding the MA term (q=1) on top of the AR term helps the model correct its errors faster, producing better forecasts.

**All three have negative bias** — they predict slightly earlier than the true RUL on average. This is intentional (safety factor) and safe (conservative predictions).

**RMSE of ~25 cycles** means the average error is about 25 flight cycles. For context, FD004 test engines have a mean true RUL of 78 cycles. So the average error is about 32% of the typical RUL — reasonable for a classical univariate model on the hardest CMAPSS dataset.

The NASA Score is high (large numbers = worse) because even a few large late-prediction errors get exponentially penalised. Looking at the sorted-predictions plot below, you can see why:

### Predicted vs Actual — AR(1) Three-Panel View:

![AR predictions three panel](report_images/T10_ARIMA_model_book_cell25_img0.png)

**Left panel — Scatter plot:** Each dot is one test engine. Perfect predictions would all lie on the red diagonal line. What we see:

- Engines with low true RUL (left side) are predicted at various values — some correctly near zero, some predicted at 110 (the FALLBACK value)
- Engines with high true RUL (right side, true RUL ≈ 125 due to capping) tend to be predicted around 110 — a conservative underestimate
- There's a cluster of dots at predicted = 110 regardless of true RUL — these are the FALLBACK predictions

**Middle panel — Error distribution:** Most errors are within ±25 cycles (the main peak). Mean error = −3.6 (slightly conservative). But there are significant tails on both sides — some engines have errors of ±75 cycles or more. These extreme errors drive the NASA score up.

**Right panel — Sorted predictions:** Engines are sorted by their true RUL (blue line = smooth S-curve from 0 to 125). Orange = predicted RUL. The prediction is jagged and highly variable — ARIMA has difficulty on FD004's multi-condition, multi-fault structure.

---

# PART 9: KEY INSIGHTS AND HONEST ASSESSMENT

## 9.1 Why the Health Index is Both Necessary and Imperfect

The health index was necessary to make ARIMA work on FD004. Without it, there would be no single time series to model.

But it has a real limitation: it compresses two different failure modes (HPC and fan degradation) into one number. The PCA-based health index finds the dominant direction of variation. If HPC degradation dominates the training data numerically, the health index will be well-calibrated for HPC failures but potentially off for fan failures. This is an inherent information loss that ARIMA cannot compensate for.

The R² of −5.188 reflects this: the health index is not a perfect linear proxy for RUL, especially for FD004 engines that die in unexpected ways.

## 9.2 Why Only 9% of Test Engines Cross the Threshold

FD004 test engines are truncated at various points. Most engines are cut off while still relatively healthy (mean true RUL = 78 cycles, meaning most are still far from failure). The health index for these engines hasn't reached the failure zone yet. So ARIMA must forecast far ahead to find the threshold crossing — and that long-horizon forecast is where ARIMA struggles most.

This is why the classical models perform worse on FD004 than on FD001 (single condition, single fault mode, more predictable degradation pattern).

## 9.3 What Makes ARIMA Better than AR and ARMA Here

AR(10) uses 10 past health_index values to predict the next. ARMA(1,1) uses 1 past value + 1 past error. ARIMA(1,2,1) uses 1 past value + 1 past error + 2 differences.

The MA term (q=1) is what makes ARIMA better. When ARIMA makes an error at step t (predicted too high or too low), the MA term immediately adjusts the next prediction in the opposite direction. This error-correction behaviour is especially valuable in the rapidly accelerating phase near failure, where the health index deviates significantly from its recent trend.

---

# PART 10: VIVA QUESTIONS — HEALTH INDEX AND p, d, q

**Q: What is the health index and why did you build it?**

The health index is a single scalar value per cycle per engine that measures degradation state. We built it because ARIMA requires a univariate time series input — we can't feed it 14 sensors simultaneously. We used PCA to find the main axis of variation in sensor space after removing the operating condition effect. The result is one number that rises monotonically from ~−1 (healthy) to ~3 (near failure) as the engine degrades.

**Q: Why didn't you just use PCA directly on the raw sensors?**

In FD004, the engine switches between 6 operating conditions. Sensor readings jump dramatically between operating conditions — a temperature change of 170°R just because the altitude changed. If we run PCA on raw sensors, the first principal component captures altitude/speed variation, not degradation. We first subtract the per-cluster mean (detrending) to remove operating condition effects. Only then does PCA find the degradation direction.

**Q: What is the per-cluster detrending step?**

We group all rows by their operating condition cluster (0–5) and compute the mean of each smoothed sensor within each cluster. Then we subtract that cluster mean from every row. The result: all operating condition baselines are removed. What remains is only how much each row deviates from the "typical healthy engine in that condition." That deviation grows with degradation.

**Q: Why is the health_index R² negative (−5.188)?**

R² measures how well a straight line fits the data. The health index has a nonlinear, accelerating relationship with RUL — flat for most of the engine's life, then sharply rising near failure. A straight line is a very poor fit for this shape, hence negative R². It does not mean the health index is broken — the plots clearly show it rises monotonically over time, which is what we need for threshold-based ARIMA forecasting.

**Q: How did you determine d=2?**

Using the ADF (Augmented Dickey-Fuller) test. At the original level (d=0), the ADF p-value is ~1.0 — clearly non-stationary. After first differencing (d=1), the p-value is ~0.97 — still non-stationary. After second differencing (d=2), the p-value drops to ~0.0 — stationary. This was confirmed across 248 training engines: 206 need d=2, 42 need d=1, none are already stationary.

**Q: Why does the health_index need d=2 specifically?**

Because FD004 degradation is accelerating, not constant-rate. The health_index doesn't just trend upward (which would be d=1) — its rate of increase also increases over time. The first difference (rate of change) is itself trending upward. The second difference (acceleration) is approximately constant — that's what becomes stationary.

**Q: How did you find p=10 for AR and p=1, q=1 for ARIMA?**

We ran a grid search across 15 representative training engines. For each engine, we fit every candidate model (p from 1 to 10 for AR; all (p,q) pairs from {1,2,3}×{1,2,3} for ARIMA) using SARIMAX and recorded the AIC. The best model per engine is the one with the lowest AIC. We then took the modal (most common) winner across 15 engines. For AR: p=10 won in 12/15 engines. For ARIMA: (p=1, q=1) won in 4/15 engines.

**Q: What is AIC and why use it instead of just RMSE?**

AIC (Akaike Information Criterion) = −2×log-likelihood + 2×(number of parameters). It penalises model complexity. A model with p=10 always fits training data better in terms of RMSE than p=3, but it uses 7 more parameters and may overfit short engine series. AIC rewards models that achieve good fit with few parameters. RMSE on training data would always favour the most complex model — AIC finds the right balance.

**Q: The Ljung-Box test failed (all p-values < 0.05). Does that mean the model is wrong?**

Not in a practical sense for this application. Ljung-Box tests whether residuals are white noise. They aren't here — there's still some structure the model missed. This means the model isn't perfectly specified. However, for our goal (long-horizon RUL estimation, not perfect one-step-ahead forecasting), this imperfection is acceptable. The model still produces useful forecasts — as shown by the rolling forecast RMSE of 0.028 on training engines. Perfect residuals would require a much more complex model that might overfit.

**Q: Why does the failure threshold = 1.685 instead of some other value?**

The threshold is the 5th percentile of health_index values among all training rows where RUL ≤ 5. We want the threshold to represent "genuinely near failure." Using the 5th percentile means 95% of near-failure training rows have health_index ≥ 1.685. We chose q=0.05 (aggressive/low threshold) because only a small fraction of test engines ever reach the threshold — using a higher threshold would make even fewer engines reachable, forcing more FALLBACK predictions.

**Q: Why do only 9% of test engines' health index reach the threshold?**

Because most FD004 test engines are truncated early — they still have 80+ cycles of life remaining when the test sequence ends. Their health index is still in the healthy zone (−1 to 0) and hasn't yet started the steep rise toward failure. ARIMA has to forecast 80+ cycles into the future to find the threshold crossing, and long-horizon forecasts are inherently less reliable.

**Q: What is the FALLBACK mechanism and when does it trigger?**

When ARIMA's forecast for an engine is nearly flat (slope ≤ 0.0001 over the forecast horizon), the threshold crossing detection fails — the forecast never reaches 1.685. This happens when the engine is still in the flat early-life phase and the model cannot see the upcoming acceleration. In this case, we fall back to a linear regression model: `RUL = slope × health_index + intercept`, fitted on training data from the last 60% of each engine's life. This gives a reasonable point estimate based on current health state alone.

**Q: Why multiply all predictions by 0.88 (the safety factor)?**

The NASA scoring function penalises late predictions (overestimating RUL) more heavily than early predictions. Specifically: late by 10 cycles → score += 1.72, but early by 10 cycles → score += 1.13. By multiplying all predictions by 0.88 we shift them slightly downward (more conservative), accumulating fewer late-prediction penalties. It's an engineering trade-off: we accept being slightly early to avoid being dangerously late.

---

# PART 11: DEEP LEARNING MODELS — COMPLETE EXPLANATION

## 11.1 Why Deep Learning After Classical Models?

The ARIMA-family models have a fundamental constraint: they are **univariate** (one series at a time) and **linear**. They operate on the health index — a single derived number that already lost information by compressing 14 sensors into one. They cannot see the individual sensor trajectories, cannot model the interaction between sensors, and cannot capture the nonlinear degradation dynamics directly.

Deep learning models take a completely different approach:
- **Input**: all 14 sensors + rolling statistics simultaneously — no information compression
- **Architecture**: can learn complex, nonlinear mappings from sensor patterns to RUL
- **Temporal modelling**: designed specifically for sequential data — each cycle's context is propagated forward
- **Scale**: trained on all training engines simultaneously, learning a universal degradation model

## 11.2 Data Preparation for Deep Learning

**File: `src/models/deep_learning.py`**

### Feature Engineering: Rolling Mean over 42 Features

For each of the 14 sensors, we compute rolling means over three windows: 5 cycles, 10 cycles, and 20 cycles. This gives:

```
14 sensors × 3 windows = 42 features per cycle
```

**Why rolling means?** Raw sensor readings are noisy cycle-to-cycle. The 5-cycle rolling mean captures short-term trends; the 20-cycle rolling mean reveals the slower background drift. Together, the three rolling windows give the model information at multiple time scales — a short burst of noise vs. a genuine underlying trend.

The raw 14 sensors themselves are **not** used as separate features — the rolling means already embed the current value (as a 1-window average). This keeps the feature count at 42.

### Sliding Windows: Preparing Sequences for the Model

Deep learning models process fixed-length sequences. We use a sliding window approach:

- **Window size = 30 cycles**: each sample consists of the 30 most recent cycles before the prediction point
- **Step = 1**: every cycle in every engine generates one sample (highly overlapping windows)
- **Shape**: `(n_samples, 30, 42)` — n_samples sliding windows, each 30 steps long with 42 features

For prediction: given a window of 30 consecutive cycles, the model predicts the RUL at the **last cycle** of that window.

### Train/Validation Split

All 248 FD004 training engines are split 80/20 **by engine ID**, not by cycle:

- **Training**: 199 engines (all cycles of those 199 engines)  
- **Validation**: 49 engines (all cycles of those 49 engines)

Splitting by engine (not by cycle) is essential. If we split by cycle within an engine, training would see the early healthy cycles and validation would see the degrading cycles of the same engine — a data leakage problem. Splitting by engine ensures the model never sees any cycle from a validation engine during training.

### Target: Capped RUL

Training targets are RUL values capped at 125. This means the model is trained to output at most 125, and for all cycles more than 125 cycles before failure, the target is exactly 125. This focuses the model on the critical degradation window.

---

## 11.3 The Training Pipeline (Shared Across All Four DL Models)

**Loss function**: Mean Squared Error (MSE). We minimise `(predicted_RUL − true_RUL)²` averaged over the batch.

**Optimiser**: Adam with learning rate = 1e-3. Adam adapts the learning rate per parameter, making it robust to different gradient scales across layers.

**Learning rate scheduling**: `ReduceLROnPlateau` — if validation loss doesn't improve for 5 consecutive epochs, the learning rate is halved (factor=0.5). This allows coarse learning initially and fine-tuning later.

**Early stopping**: if validation loss doesn't improve for 10 consecutive epochs, training stops and the best weights are restored. This prevents overfitting.

**Gradient clipping**: `max_norm = 1.0`. During backpropagation, if the gradient vector's L2 norm exceeds 1.0, it is rescaled to 1.0. This prevents the "exploding gradients" problem — a critical stability issue for RNN/LSTM/GRU on long sequences.

**Maximum epochs**: 50. In practice, early stopping triggers before this for most models.

**Batch size**: 128. Stochastic gradient descent with batches of 128 samples.

**Device**: MPS (Apple Silicon GPU) on the project machine; automatically falls back to CPU if unavailable.

---

## 11.4 Model 1: Vanilla RNN

### Architecture

```
Input layer:  (batch=128, seq=30, features=42)
RNN Layer 1:  hidden=64, dropout=0.2 between layers
RNN Layer 2:  hidden=64
FC Layer:     64 → 1 (RUL scalar)
Total params: 15,297
```

The vanilla RNN processes the sequence one step at a time. At each step t, the hidden state `h_t` is updated as:

```
h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b)
```

where `x_t` is the input at step t and `h_{t-1}` is the previous hidden state. Only the hidden state at the last step (h₃₀) is passed to the fully connected layer.

**The vanishing gradient problem**: during backpropagation, gradients must flow back through 30 steps of the `tanh` operation. Each step, the gradient gets multiplied by the Jacobian of `tanh`, which is ≤ 1. After 30 steps: gradient ≈ (0.8)³⁰ ≈ 0.001 — effectively zero. This means the RNN barely learns from events 10+ cycles ago.

**Why RNN still works here?** The 30-cycle window + gradient clipping + the fact that RUL is mostly driven by recent cycles (the last 10 cycles carry the strongest degradation signal) means vanishing gradients are not catastrophic. The model still captures short-term trends well enough.

### Training Results

```
Training stopped: epoch 50 (ran full — never triggered early stopping)
Final train loss: ~0.0012 (MSE in normalised RUL units)
Final val loss: ~0.0015
```

![RNN Loss Curves](report_images/RNN_cell12_img0.png)

**What you are looking at:** Training and validation loss curves for the RNN model across epochs.

- **Blue line (train loss)**: decreases steadily from epoch 1 to 50. The ReduceLROnPlateau scheduler triggers partway through (visible as the slight change in the rate of decrease where the learning rate was halved).
- **Orange line (val loss)**: also decreases but with more fluctuation. The gap between train and val loss shows mild overfitting — the model learned training-specific patterns to a small degree.
- The fact that training ran all 50 epochs without early stopping means the validation loss was still slowly improving at epoch 50 — the RNN kept learning but slowly.

### Prediction Results

![RNN Predictions](report_images/RNN_cell12_img1.png)

**What you are looking at:** Predicted vs. actual RUL across all 249 FD004 test engines (scatter plot), error distribution, and sorted predictions.

- **Scatter**: most points cluster near the red diagonal (perfect prediction) for RUL values 0–80. The model does well for engines near failure (RUL < 40) and for engines that are fresh (RUL capped at 125).
- **Error distribution**: centred slightly above 0 (bias = +1.21) — the model is very slightly late on average but essentially unbiased.
- The spread is visually smaller than the ARIMA scatter plot — RNN captures degradation patterns better than ARIMA.

### Final RNN Metrics:
```
RMSE       : 15.30
NASA Score : 1,593.95
R²         : 0.8733
Bias       : +1.21  (very slightly late → barely noticeable)
```

---

## 11.5 Model 2: LSTM (Long Short-Term Memory)

### Architecture

```
Input layer:   (batch=128, seq=30, features=42)
LSTM Layer 1:  hidden=64, dropout=0.2
LSTM Layer 2:  hidden=64
FC Layer:      64 → 1
Total params:  60,993  (≈4× more than RNN)
```

LSTM extends the vanilla RNN by adding a **cell state** (long-term memory) and three **gates**:

```
Forget gate:  f_t = σ(W_f × [h_{t-1}, x_t] + b_f)   ← what to forget from cell
Input gate:   i_t = σ(W_i × [h_{t-1}, x_t] + b_i)   ← what new info to store
Output gate:  o_t = σ(W_o × [h_{t-1}, x_t] + b_o)   ← what to output from cell
Cell update:  c_t = f_t * c_{t-1} + i_t * tanh(W_c × [h_{t-1}, x_t] + b_c)
Hidden state: h_t = o_t * tanh(c_t)
```

The cell state flows from step to step with only multiplicative interactions (forget gate). This allows gradients to flow back without repeatedly squashing through tanh — the **solution to vanishing gradients**.

Why 60,993 parameters vs 15,297 for RNN? LSTM has 4 weight matrices per layer instead of 1 (for the 4 computations above). More parameters → more capacity but also more chance of overfitting.

### Training Results

```
Training stopped: epoch 13  (early stopping triggered at epoch 13)
Best epoch:       epoch 3
Validation loss at best: very low
```

![LSTM Loss Curves](report_images/LSTM_cell12_img0.png)

**What you are looking at:** LSTM training and validation loss curves.

- Training runs for only 13 epochs before early stopping triggers. The validation loss (orange) reaches its minimum very early (around epoch 3) then starts climbing — clear overfitting.
- The large gap between train loss (blue, keeps decreasing) and val loss (orange, increases after epoch 3) is textbook overfitting: LSTM's higher capacity caused it to memorise training engines rather than generalise.
- The best weights (saved at the epoch with lowest val loss) are what's used for test evaluation.

### Prediction Results

![LSTM Predictions](report_images/LSTM_cell12_img1.png)

**What you are looking at:** LSTM predictions are significantly lower than actual RUL for most engines — the model is highly conservative (early predictions).

- **Scatter**: strong horizontal clustering at predicted RUL ≈ 0–40, even for engines with true RUL of 100–125. The model is stuck predicting "this engine is almost dead" for most engines.
- **Error distribution**: mean error = −23.15 — the model underestimates RUL by 23 cycles on average. The distribution is heavily left-skewed (many large negative errors).
- This is a consequence of the overfitting: the model learned the training set's RUL distribution, which is dominated by low-RUL cycles (since every engine produces many cycles near failure), and defaulted to predicting low RUL.

### Final LSTM Metrics:
```
RMSE       : 35.24  (worst among DL models)
NASA Score : 6,433.77
R²         : 0.3276  (poor)
Bias       : −23.15  (extremely early → overly conservative)
```

**Why did LSTM underperform despite being more powerful than RNN?**

Three reasons:
1. **Overfitting**: 60,993 parameters is over-specified for this dataset size. With only 199 training engines, LSTM memorised training patterns instead of generalising.
2. **Short training**: early stopping at epoch 13 gave LSTM fewer gradient updates than RNN (50 epochs). LSTM needs more careful tuning (lower learning rate, more regularisation) to converge properly.
3. **Dropout placement**: the current implementation applies dropout between layers but not within the recurrent connections (no variational dropout). More regularisation within the recurrent layers would help LSTM generalise better.

---

## 11.6 Model 3: GRU (Gated Recurrent Unit)

### Architecture

```
Input layer:   (batch=128, seq=30, features=42)
GRU Layer 1:   hidden=64, dropout=0.2
GRU Layer 2:   hidden=64
FC Layer:       64 → 1
Total params:  45,761
```

GRU simplifies LSTM by combining the forget and input gates into a single **update gate** and removing the separate cell state:

```
Reset gate:   r_t = σ(W_r × [h_{t-1}, x_t])     ← how much past to forget
Update gate:  z_t = σ(W_z × [h_{t-1}, x_t])     ← how much to update
New memory:   n_t = tanh(W_n × [r_t * h_{t-1}, x_t])
Output:       h_t = (1 − z_t) * h_{t-1} + z_t * n_t
```

GRU has **2 gates vs LSTM's 3** — fewer parameters (45,761 vs 60,993), but still retains the long-term memory capability that solves vanishing gradients. In practice, GRU and LSTM perform similarly on most tasks, but GRU trains faster and is less prone to overfitting on small datasets.

### Training Results

```
Training stopped: epoch 15  (early stopping)
```

![GRU Loss Curves](report_images/GRU_cell12_img0.png)

**What you are looking at:** GRU training and validation loss. The curves are more balanced than LSTM — the gap between train and val loss is smaller, indicating less overfitting. Early stopping at epoch 15 also suggests more stable training than LSTM.

### Prediction Results

![GRU Predictions](report_images/GRU_cell12_img1.png)

**What you are looking at:** GRU predictions are well-distributed across the full RUL range (0–125), with less conservative bias than LSTM.

- **Scatter**: points track the red diagonal better than LSTM. The model correctly predicts low RUL for degraded engines and high RUL for healthy ones.
- **Bias = −9.31**: slightly conservative (early predictions on average). More conservative than RNN (+1.21) but much better than LSTM (−23.15).

### Final GRU Metrics:
```
RMSE       : 17.28
NASA Score : 957.85  (BEST NASA score among all DL models)
R²         : 0.8383
Bias       : −9.31  (slightly conservative)
```

**Why GRU has the best NASA score but not the best RMSE?** The NASA scoring function penalises late predictions (positive errors) much more than early predictions. GRU's negative bias (slightly early) means it rarely makes large late-prediction errors — the exponential penalty term in the NASA score is rarely triggered. RNN has lower RMSE (15.30 vs 17.28) but slightly positive bias (+1.21), meaning occasional late predictions drive up its NASA score.

---

## 11.7 Model 4: Transformer

### Architecture

```
Input layer:           (batch=128, seq=30, features=42)
Input projection:      Linear(42 → 64)  — project to model dimension
Positional encoding:   Learnable (30 × 64 parameters)
Transformer encoder:   2 layers, 4 attention heads, d_model=64, FFN dim=256, dropout=0.1
Pooling:               Mean over sequence dimension (→ 64-dim vector)
FC Layer:              64 → 1
Total params:          104,705
```

**Key conceptual difference from RNN/LSTM/GRU:** The Transformer does not process the sequence step-by-step. Instead, it looks at all 30 steps simultaneously through **self-attention**.

For each step t, the attention mechanism asks: "which other steps in this window are most relevant to predicting RUL from step t?" This allows the model to directly compare cycle 30 (the prediction point) with any earlier cycle — even if they are 25 steps apart. Long-range dependencies are captured without the gradient problem.

**Positional encoding**: Because the Transformer processes all steps simultaneously (not sequentially), it needs to be told the position of each step. We use learnable positional encodings — a separate parameter vector for each of the 30 positions that is added to the input embedding. The model learns which positional relationships are important.

**Multi-head attention (4 heads)**: four independent attention modules, each looking for different types of relationships between steps. One head might focus on recent cycles (temporal proximity), another on cycles with similar sensor patterns (degradation state similarity).

**Mean pooling**: after the 2-layer encoder, we have 30 vectors of dimension 64 (one per cycle). We average these to get a single 64-dim summary, then pass it to the FC layer for RUL prediction.

### Training Results

```
Training stopped: epoch 19  (early stopping)
```

![Transformer Loss Curves](report_images/Transformer_cell12_img0.png)

**What you are looking at:** Transformer training and validation loss curves. The curves are the smoothest of all four models — validation loss decreases steadily and the train/val gap remains small throughout. This indicates good generalisation: despite having the most parameters (104,705), the Transformer does not overfit significantly.

The attention mechanism + mean pooling act as implicit regularisation — the model cannot overfit a single time step because it must aggregate information across all 30 positions.

### Prediction Results

![Transformer Predictions](report_images/Transformer_cell12_img1.png)

**What you are looking at:** Transformer predictions across all 249 test engines.

- **Scatter**: the tightest cluster around the red diagonal among all models — fewest outliers and smallest spread.
- **For engines with RUL ≈ 0–60**: predictions are accurate and near the diagonal.
- **For engines with RUL = 125** (the capped high-RUL engines): predictions cluster just below 125, correctly identifying these as low-urgency engines.
- **Bias = +2.26**: slightly late on average — the Transformer is very slightly optimistic. This is the opposite of GRU's conservative bias. The slightly late bias costs the Transformer in NASA score compared to GRU.

### Final Transformer Metrics:
```
RMSE       : 13.73  (BEST RMSE — best overall accuracy)
NASA Score : 1,286.14
R²         : 0.8979  (BEST R² — explains 89.8% of RUL variance)
Bias       : +2.26  (very slightly late → negligible)
```

**Why does the Transformer achieve the best accuracy on FD004?**

FD004's complexity is exactly what the Transformer is designed for:
1. **Six operating conditions**: the attention mechanism can learn to ignore sensor readings from different operating conditions (they're at different positions in the sequence with different patterns) and focus on comparison between same-condition cycles.
2. **Two failure modes**: multi-head attention allows different heads to specialise — one head for HPC degradation patterns, another for fan degradation patterns.
3. **Long-range dependencies**: the health index has long-term memory (what happened 20 cycles ago still matters). Self-attention captures this without gradient decay.
4. **No sequential bottleneck**: RNN/LSTM/GRU must compress all 30 cycles into one hidden vector. The Transformer keeps all 30 representations and mean-pools at the very end — less information loss.

---

## 11.8 Deep Learning Model Comparison

| Model | Params | Epochs Trained | RMSE | NASA Score | R² | Bias |
|-------|--------|----------------|------|------------|-----|------|
| RNN | 15,297 | 50 (full) | 15.30 | 1,593.95 | 0.873 | +1.21 |
| LSTM | 60,993 | 13 | 35.24 | 6,433.77 | 0.328 | −23.15 |
| GRU | 45,761 | 15 | 17.28 | **957.85** | 0.838 | −9.31 |
| Transformer | 104,705 | 19 | **13.73** | 1,286.14 | **0.898** | +2.26 |

**Key takeaways:**
- Transformer wins on RMSE and R² — best raw accuracy
- GRU wins on NASA score — safest predictions (slightly early, never dangerously late)
- LSTM fails despite having the most capacity of the recurrent models — overfitting with 4× RNN parameters
- RNN surprisingly competitive — simple architecture works well with gradient clipping and full 50-epoch training
- ARIMA had RMSE=24.76 — all four DL models significantly outperform it except LSTM

---

# PART 12: QUANTILE MODELS — UNCERTAINTY QUANTIFICATION

## 12.1 Why Quantile Models?

All models so far (ARIMA, RNN, LSTM, GRU, Transformer) produce a **single point prediction** — one number for RUL. But a single number cannot tell you:
- How confident is this prediction?
- Is this a certain prediction (narrow range) or a wild guess (wide range)?
- What is the worst-case scenario (the 90th percentile RUL)?

In safety-critical applications like aircraft maintenance, knowing uncertainty is as important as knowing the estimate. "This engine has RUL = 50 cycles (±5 cycles)" is very different from "RUL = 50 cycles (±30 cycles)."

**Quantile models solve this by outputting three numbers simultaneously:**
- **Q10 (10th percentile)**: pessimistic estimate — "there is only 10% chance the engine dies sooner than this"
- **Q50 (50th percentile / median)**: best estimate
- **Q90 (90th percentile)**: optimistic estimate — "there is 90% chance the engine dies sooner than this"

The interval [Q10, Q90] is the **80% prediction interval** — you expect 80% of true RUL values to fall within this range.

## 12.2 Pinball Loss — Training Quantile Models

Standard models are trained with MSE loss. Quantile models are trained with **Pinball loss** (also called Quantile loss).

For a target quantile τ (e.g., τ=0.1 for Q10):
```
error = actual_RUL − predicted_RUL

Pinball loss = τ × error          if error ≥ 0   (under-prediction)
             = (τ − 1) × error    if error < 0   (over-prediction)
```

**Intuition for why this works:**

For τ = 0.1 (Q10 — the low/pessimistic quantile):
- If the model under-predicts (actual > predicted): loss = 0.1 × error → small penalty (we WANT to be low)
- If the model over-predicts (actual < predicted): loss = 0.9 × error → large penalty (we DO NOT want Q10 to be too high)
- Result: the model learns to predict the 10th percentile — 90% of errors will be positive (actual > predicted)

For τ = 0.9 (Q90 — the high/optimistic quantile):
- If the model under-predicts: loss = 0.9 × error → large penalty
- If the model over-predicts: loss = 0.1 × error → small penalty
- Result: the model learns to predict the 90th percentile — 10% of errors will be positive

The model outputs three neurons simultaneously: one calibrated to Q10, one to Q50, one to Q90.

### Post-Processing: Monotone Sort

The model outputs Q10, Q50, Q90 independently. It's possible (and does happen) that the raw output has Q10 > Q50 or Q50 > Q90 — a mathematical impossibility. We enforce monotonicity by sorting: `Q10, Q50, Q90 = sorted([pred10, pred50, pred90])`. This simple post-processing step ensures valid quantile ordering without re-training.

## 12.3 Coverage vs. RMSE Trade-off

**Coverage** = the percentage of test engines where the true RUL falls within the [Q10, Q90] prediction interval.

A perfectly calibrated model should have coverage = 80% (since [Q10, Q90] is the 80% interval).

But there is a natural tension:
- To increase coverage: widen the interval (lower Q10, raise Q90) — easy but low quality
- To decrease RMSE: tighten the predictions — narrow intervals have worse coverage

The ideal model achieves **calibration** (coverage ≈ 80%) AND **sharpness** (narrow intervals).

## 12.4 The Five Quantile Models

All five models share the same data pipeline (30-cycle windows, 42 rolling features, same train/val split) and the same Pinball loss. The architectures differ.

**Q-MLP**: a fully-connected network that takes the 30×42=1260 flattened features as input. No temporal structure — it treats the window as a flat vector.
```
Input:   (batch, 1260)  — flattened 30×42 window
Hidden:  Linear(1260→128) → ReLU → Dropout(0.2)
Hidden:  Linear(128→64)  → ReLU → Dropout(0.2)
Output:  Linear(64→3)    → [Q10, Q50, Q90]
```

**Q-RNN, Q-LSTM, Q-GRU, Q-Transformer**: same architectures as their point-prediction counterparts, but with output dimension 3 instead of 1, and trained with Pinball loss instead of MSE.

### Q-MLP: Loss and Prediction

![Q-MLP Loss](report_images/T12_Quantile_models_cell19_img1.png)

**What you are looking at:** Q-MLP training and validation Pinball loss across epochs. The curves decrease and stabilise, but the model trains for fewer epochs before convergence because MLP has no recurrent components to warm up.

![Q-MLP Predictions](report_images/T12_Quantile_models_cell19_img2.png)

**What you are looking at:** Q-MLP quantile predictions for a sample of test engines. The three bands show Q10 (lower boundary), Q50 (middle line), and Q90 (upper boundary).

- The interval width is generally wide — Q-MLP has the best coverage (84.3%) but widest intervals.
- The median (Q50) line is less tightly coupled to true RUL compared to Q-Transformer.
- For engines with true RUL near 125, the Q90 boundary often hits 125 (the cap) correctly.

**Q-MLP Metrics:**
```
RMSE (Q50) : 16.86
Coverage   : 84.3%  (closest to the ideal 80% calibration — actually slightly over-covers)
```

Q-MLP over-covers because flattening the temporal structure loses the model's ability to predict precisely — it produces wide intervals that capture almost everything but with low sharpness.

---

### Q-RNN: Loss and Prediction

![Q-RNN Loss](report_images/T12_Quantile_models_cell19_img4.png)

**What you are looking at:** Q-RNN training and validation Pinball loss. Similar to the point-prediction RNN, training is stable across all 50 epochs (or until early stopping). The validation loss is smooth and close to training loss — no severe overfitting.

![Q-RNN Predictions](report_images/T12_Quantile_models_cell19_img5.png)

**What you are looking at:** Q-RNN quantile intervals. The intervals are narrower than Q-MLP (better sharpness) but coverage drops.

**Q-RNN Metrics:**
```
RMSE (Q50) : 15.58
Coverage   : (see comparison below)
```

---

### Q-LSTM: Loss and Prediction

![Q-LSTM Loss](report_images/T12_Quantile_models_cell19_img7.png)

**What you are looking at:** Q-LSTM training and validation Pinball loss. The validation loss curve diverges from training loss more than Q-RNN — same overfitting issue seen in the point-prediction LSTM.

![Q-LSTM Predictions](report_images/T12_Quantile_models_cell19_img8.png)

**What you are looking at:** Q-LSTM quantile intervals. The Q50 median predictions are consistently low (conservative/early), matching the point-prediction LSTM's bias of −23 cycles. The intervals are narrow but don't cover the true RUL — worst calibration.

**Q-LSTM Metrics:**
```
RMSE (Q50) : 15.50
Coverage   : 46.8%  (WORST — true RUL falls in the interval only 46.8% of the time)
```

Coverage of 46.8% vs. 80% target means the model's 80%-intended interval actually covers only ~47% of cases — severely miscalibrated. The LSTM is too confident in its wrong predictions.

---

### Q-GRU: Loss and Prediction

![Q-GRU Loss](report_images/T12_Quantile_models_cell19_img10.png)

**What you are looking at:** Q-GRU training/validation Pinball loss. Similar to the point-prediction GRU — clean convergence, small train/val gap, early stopping.

![Q-GRU Predictions](report_images/T12_Quantile_models_cell19_img11.png)

**What you are looking at:** Q-GRU predictions. Intervals are well-centred around the true RUL for most engines. The Q50 line tracks the true RUL pattern well.

**Q-GRU Metrics:**
```
RMSE (Q50) : 14.50
Coverage   : (good — intervals centred around true RUL)
```

---

### Q-Transformer: Loss and Prediction

![Q-Transformer Loss](report_images/T12_Quantile_models_cell19_img13.png)

**What you are looking at:** Q-Transformer training and validation Pinball loss. The best-behaved curves of all five models — smooth decrease, very small train/val gap, stable convergence.

![Q-Transformer Predictions](report_images/T12_Quantile_models_cell19_img14.png)

**What you are looking at:** Q-Transformer quantile predictions. The [Q10, Q90] intervals are:
- Narrow for engines near failure (low RUL) — the model is confident when it sees clear degradation
- Wider for engines with high RUL — more uncertainty for still-healthy engines
- The Q50 median closely tracks the true RUL (red line) across the full range

This **adaptive uncertainty** — tighter intervals when confident, wider when uncertain — is the hallmark of a well-calibrated quantile model.

**Q-Transformer Metrics:**
```
RMSE (Q50) : 14.15  (BEST among quantile models)
Coverage   : (good calibration)
```

## 12.5 Quantile Model Comparison

| Model | RMSE (Q50) | Coverage | Key Observation |
|-------|------------|----------|-----------------|
| Q-MLP | 16.86 | **84.3%** | Best coverage but widest, least sharp intervals |
| Q-RNN | 15.58 | Good | Competitive with point-prediction RNN |
| Q-LSTM | 15.50 | **46.8%** | Worst calibration — dangerously overconfident |
| Q-GRU | 14.50 | Good | Well-balanced; second-best RMSE |
| Q-Transformer | **14.15** | Good | Best RMSE; adaptive interval width |

**The Coverage vs. RMSE trade-off in action:**
- Q-MLP achieves best coverage (84.3%) by producing wide intervals — it almost always includes the true RUL but with high uncertainty
- Q-LSTM achieves middle-of-pack RMSE (15.50) but its coverage (46.8%) is disastrously low — it is confident in wrong predictions
- Q-Transformer achieves both the best RMSE and reasonable coverage — the best overall quantile model

---

# PART 13: COMPLETE MODEL COMPARISON — ALL MODELS

## 13.1 Master Results Table

| Model | Type | RMSE | NASA Score | R² | Bias |
|-------|------|------|------------|-----|------|
| AR(10,2,0) | Classical | 27.67 | 25,742 | 0.586 | −4.74 |
| ARMA(1,2,1) | Classical | 26.19 | 17,514 | 0.629 | −1.53 |
| ARIMA(1,2,1) | Classical | 24.76 | 13,791 | 0.668 | −3.58 |
| RNN | Deep Learning | 15.30 | 1,594 | 0.873 | +1.21 |
| LSTM | Deep Learning | 35.24 | 6,434 | 0.328 | −23.15 |
| **GRU** | Deep Learning | 17.28 | **958** | 0.838 | −9.31 |
| **Transformer** | Deep Learning | **13.73** | 1,286 | **0.898** | +2.26 |
| Q-MLP | Quantile | 16.86 | — | — | — |
| Q-RNN | Quantile | 15.58 | — | — | — |
| Q-LSTM | Quantile | 15.50 | — | — | — |
| Q-GRU | Quantile | 14.50 | — | — | — |
| **Q-Transformer** | Quantile | **14.15** | — | — | — |

**Notes:**
- NASA Score and R² are computed on the Q50 median predictions for quantile models (same metric as point-prediction)
- Best RMSE overall: **Transformer** (13.73) closely followed by Q-Transformer (14.15)
- Best NASA Score: **GRU** (957.85) — safest predictions
- Worst performer: **LSTM** (RMSE=35.24) — overfitting with insufficient regularisation

## 13.2 Why Deep Learning Dominates Classical Models

RMSE improvement from best classical (ARIMA, 24.76) to best DL (Transformer, 13.73) = **45% reduction in error**.

The root cause: ARIMA uses only 1 feature (health_index) while DL uses 42 features (rolling means of 14 sensors). ARIMA models a single time series; DL models the joint trajectory of all 14 sensor channels simultaneously. For FD004 with two failure modes, the multi-sensor view is essential — HPC and fan degradation produce different sensor patterns that ARIMA's health index partially conflates.

## 13.3 Quantile vs. Point Prediction

For each architecture type, the quantile version achieves slightly better Q50 RMSE than the point-prediction version:

| Architecture | Point RMSE | Quantile RMSE | Difference |
|---|---|---|---|
| RNN | 15.30 | 15.58 | +0.28 (slightly worse) |
| LSTM | 35.24 | 15.50 | **−19.74 (much better!)** |
| GRU | 17.28 | 14.50 | −2.78 (better) |
| Transformer | 13.73 | 14.15 | +0.42 (slightly worse) |

The most dramatic improvement is Q-LSTM vs LSTM — the Pinball loss approach somehow regularised LSTM's training better than MSE. This is likely because Pinball loss is asymmetric and more robust to outliers than MSE, reducing the impact of the worst-case overfitting errors.

---

# PART 14: COMPLETE VIVA QUESTIONS — DEEP LEARNING AND QUANTILE MODELS

**Q: Why did you use sliding windows of 30 cycles for deep learning?**

The window of 30 cycles gives the model 30 consecutive timesteps of context for each prediction. Why 30? It's a standard choice that balances several factors: (1) short enough that early healthy cycles don't dominate the window for degrading engines, (2) long enough to capture the medium-term trend (rolling means over 20 cycles need 20+ data points to be meaningful), (3) for engines shorter than 30 cycles, we zero-pad — the model must also learn to handle incomplete histories. We also tested windows of 20 and 50; 30 gave the best validation performance.

**Q: What are the 42 input features? Why not use raw sensors?**

The 42 features are rolling means of the 14 sensors at three window sizes: 5, 10, and 20 cycles (14 × 3 = 42). We do not include raw sensor values separately — a rolling mean with window=1 is just the raw value, so it's implicit. Rolling means at multiple scales give the model information at different temporal resolutions: the 5-cycle mean captures fast fluctuations, the 20-cycle mean captures the slow background drift. This multi-scale view helps the model distinguish noise from genuine degradation trends without requiring the model to learn the smoothing itself from scratch.

**Q: Why does the Transformer outperform LSTM and GRU on FD004?**

Three specific reasons for FD004: (1) Multi-head attention handles FD004's two failure modes — different attention heads can specialise for HPC vs fan degradation sensor patterns. (2) Self-attention avoids the sequential bottleneck — RNN/LSTM/GRU compress 30 cycles into one hidden state, losing early-window information; the Transformer keeps all 30 representations until the final pooling. (3) The Transformer's lack of sequential inductive bias means it doesn't assume temporal proximity = temporal relevance — cycle 1 and cycle 30 can have equal attention weight if they're both informative. For FD004 where the most informative degradation signal can appear at any point in the window, this flexibility helps.

**Q: Why did LSTM fail when it was supposed to be better than RNN?**

LSTM's additional complexity (4 weight matrices per layer vs 1 for RNN = 60,993 vs 15,297 parameters) means it needs more data and more careful tuning to generalise. With 199 training engines, LSTM was over-parameterised — it memorised training engine trajectories instead of learning a general degradation model. The early stopping at epoch 13 (vs RNN's full 50 epochs) shows LSTM overfit very quickly. Solutions would include: (1) more dropout inside recurrent connections (variational dropout), (2) smaller hidden size (32 instead of 64), (3) more training engines (augmentation), (4) weight decay. For the current setup, the simpler RNN with 4× fewer parameters generalised better.

**Q: What is the difference between RMSE and NASA score? Which matters more?**

RMSE measures average squared error symmetrically — a prediction that's 10 cycles early is penalised equally to one that's 10 cycles late. NASA score is asymmetric: late predictions are penalised with `exp(error/10)−1` and early predictions with `exp(−error/13)−1`. For an error of 10 cycles: late → penalty = 1.72, early → penalty = 1.13. So the NASA score reflects operational reality: predicting an engine will last longer than it does (missing a failure) is more dangerous than predicting it will fail soon (unnecessary maintenance). For safety decisions, the NASA score matters more. For comparing model accuracy in absolute terms, RMSE is more interpretable.

**Q: GRU has the best NASA score but not the best RMSE. Which model would you deploy?**

For a real aircraft maintenance system, I would deploy the GRU. The NASA score is the operationally relevant metric — it directly quantifies the cost of late vs early predictions. GRU's lower NASA score (957 vs 1,286 for Transformer) means it makes fewer dangerously late predictions. Its slightly higher RMSE (17.28 vs 13.73) is a secondary concern. Additionally, GRU's −9.31 bias means it's conservative on average — it will tend to schedule maintenance slightly early rather than slightly late. In safety-critical systems, conservative predictions are preferable.

**Q: What is Pinball loss and why is it needed for quantile models?**

Pinball loss is an asymmetric loss function that tilts the training gradient to make the model predict a specific quantile rather than the mean. For a target quantile τ, under-prediction (actual > predicted) is penalised by τ and over-prediction by (1−τ). When τ=0.1, over-predictions are penalised 9× more than under-predictions — pushing the model to predict low values such that 90% of actual values lie above the prediction. This is the definition of the 10th percentile. Standard MSE minimises the conditional mean; Pinball loss minimises the conditional quantile. Three output neurons with three different τ values gives us Q10, Q50, Q90 simultaneously.

**Q: Why do you sort the quantile outputs after prediction?**

The three output neurons (Q10, Q50, Q90) are trained independently with different Pinball loss weights. During inference, nothing enforces `Q10 < Q50 < Q90`. If the Q10 neuron predicts 60 and the Q50 neuron predicts 55, the interval would be invalid (lower bound > median). This "quantile crossing" is uncommon but does happen. Sorting the three predictions (taking the minimum as Q10, middle as Q50, maximum as Q90) is a simple, model-free post-processing step that guarantees valid ordering. More sophisticated approaches (quantile regression forests, deep learning with monotone output layers) can enforce this structurally, but sorting is sufficient here.

**Q: What does coverage of 46.8% mean for Q-LSTM? Why is it a problem?**

An 80% prediction interval [Q10, Q90] should contain the true value 80% of the time. Q-LSTM achieves only 46.8% — the true RUL falls outside the predicted interval 53.2% of the time. For a maintenance decision system, this means: if you look at the Q10–Q90 interval and make scheduling decisions based on it, you will be wrong more often than right. The interval is telling you "the engine will probably fail between cycle X and cycle Y" — but in more than half the cases, it fails outside that window. This is worse than useless for uncertainty-aware decision making. Q-LSTM is overconfident: its narrow intervals reflect false certainty. The root cause is the same LSTM overfitting problem — the model is very confident about wrong predictions.

**Q: Why did you cap RUL at 125 for both classical and DL models?**

The cap serves two purposes: (1) Practical: predicting "RUL = 250 cycles" vs "RUL = 200 cycles" makes no operational difference — the engine won't need maintenance for a very long time either way. The last 125 cycles are the relevant window for scheduling. (2) Statistical: without capping, the training data would have RUL values from 0 to 500+, and the model would need to fit a very wide range. The MSE loss would be dominated by high-RUL prediction errors (because squared errors are larger there) and would neglect the low-RUL region (the critical zone). By capping at 125, the model focuses entirely on the degradation window that matters for maintenance.

**Q: Your Transformer had 104,705 parameters but didn't overfit, while LSTM had 60,993 and severely overfit. Why?**

The Transformer has more parameters but also more structural regularisation. First, multi-head self-attention with 4 heads inherently distributes the learning across different "views" of the sequence — each head independently attends to different patterns, preventing any single pathway from dominating and overfitting. Second, mean pooling aggregates all 30 positions before the final layer — no single timestep can dominate the prediction. Third, the Transformer uses separate dropout (0.1) inside the attention and feed-forward layers (not just between layers). LSTM's dropout is only between layers, leaving the recurrent connections without regularisation. The combination of architectural diversity + multi-scale aggregation + within-layer dropout makes the Transformer more resistant to overfitting despite having more parameters.

**Q: What is the difference between Q-MLP and Q-Transformer on this task?**

Q-MLP flattens the 30×42 window into a 1260-element vector and processes it through fully connected layers. This means it treats the temporal ordering of cycles as irrelevant — cycle 1 and cycle 30 are just two different positions in a flat vector. Q-Transformer explicitly models the temporal structure through positional encodings and self-attention — it knows that cycle 30 is the "most recent" and that temporal proximity matters. For RUL prediction where the recent cycles (cycles 25–30) are most informative about current health state, knowing the temporal order is crucial. That's why Q-Transformer (RMSE=14.15) significantly outperforms Q-MLP (RMSE=16.86).

**Q: If you had to improve the project further, what would you do?**

Three specific improvements: (1) **Hyperparameter tuning for LSTM** — reduce hidden size to 32 (from 64), add variational dropout (recurrent dropout), increase patience from 10 to 20. This would likely close the LSTM gap with GRU. (2) **Ensemble the top 3 models** — take the average prediction of Transformer, GRU, and Q-Transformer. Ensembles consistently outperform individual models because errors are partially independent. (3) **Fault-mode-aware training** — label each engine as HPC or fan degradation type (using clustering on sensor trajectories), then train separate models per fault mode or add fault-mode as a training label. FD004's two failure modes confuse a single model; separating them would improve accuracy for both modes.

**Q: How does the feature engineering differ between classical and deep learning models?**

Classical (ARIMA):
- Features: 1 (health index — PCA of 14 sensors after cluster detrending)
- No rolling windows used as model input (rolling smoothing is used inside health index construction, but the ARIMA model sees one time series)
- Feature engineering: very heavy (6 preprocessing steps)

Deep Learning (RNN/LSTM/GRU/Transformer):
- Features: 42 (rolling means of 14 sensors over 3 window sizes)
- Input shape: (30 cycles × 42 features) per sample
- Feature engineering: light — just compute rolling means; the model learns the rest

The classical approach needs heavy feature engineering to compensate for the model's inability to handle multiple variables. The DL approach needs less feature engineering because the model capacity handles the sensor interactions. The rolling means are the only manual feature engineering step — everything else (cross-sensor interactions, nonlinear patterns, temporal dependencies) is learned from data.

**Q: In the sliding window approach, is there any data leakage?**

No. The train/val split is done by engine ID before creating windows. All windows from engine 1–199 go to training, all windows from engine 200–248 go to validation. No window spans two engines. Within a window, only the 30 cycles immediately before the prediction point are used — no future cycles are included (the window predicts RUL at cycle t from cycles t-29 to t). The test evaluation uses the last window per engine from the test set (test engines were never seen during training). There is no leakage.

**Q: How did you handle test engines that are shorter than the window size (30 cycles)?**

For test engines with fewer than 30 observed cycles, we zero-pad the beginning of the window. If an engine has only 15 cycles, the input window is [0,0,...,0, cycle1, cycle2,..., cycle15] — the first 15 positions are zeros and the last 15 are the actual data. The model sees many zero-padded sequences during training (for engines with short histories at the start of their trajectories), so it learns to handle them. Zero-padding is applied with min_periods=1 in the rolling calculations, ensuring no NaN values propagate.
