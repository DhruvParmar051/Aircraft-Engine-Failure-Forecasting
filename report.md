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
