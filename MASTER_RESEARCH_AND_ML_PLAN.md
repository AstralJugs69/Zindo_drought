# A Step Ahead of Drought — Master Research, Validation, and ML Development Plan

**Competition:** Zindi — *A Step Ahead of Drought: Forecasting Global Water Storage Challenge*  
**Organizer:** ITU / AI for Good, with Copernicus/JRC context  
**Competition close:** 2026-09-13  
**Document role:** Canonical research ledger, modeling blueprint, experiment discipline, leakage specification, compute plan, and living decision record for the entire challenge.  
**First compiled:** 2026-09-07  
**Current phase:** Causal validation infrastructure is complete; EXP005 hydrologic-gap LightGBM is the current dev incumbent; the recent lockbox has been opened once for frozen EXP003 only; subsequent tuning remains dev-fold-only.  
**Local project path:** `C:\dev\zindi\drought`  
**Git status at first compilation:** Folder exists, but it is not yet initialized as a Git repository and has no remote configured.

---

## 0. How this document must be used

This file is the project’s source of truth. We should update it whenever we:

- discover a new competition rule or organizer clarification;
- modify the definition of a feature, label, horizon, or validation fold;
- introduce a new data source;
- add or kill a model family;
- change an experiment promotion/kill threshold;
- observe a new failure mode or leakage risk;
- make a leaderboard submission;
- learn that CV and leaderboard disagree;
- change the compute environment;
- create a new reproducibility or trustworthiness artifact.

The purpose is not to preserve every idea forever. The purpose is to preserve **why** we accepted, rejected, delayed, or modified each idea so that we do not drift back into previously falsified paths during the final competition sprint.

Every substantial experiment should eventually add a short entry to the experiment ledger in this file or to a linked machine-readable run registry.

---

# 1. Executive decision summary

The strongest current strategy is:

> **competition-faithful causal rolling-mask validation → explicit trend/seasonal baselines → global pooled direct horizon-aware residual/ΔTWS LightGBM → causal TWS/SPEI/soil histories and hydrologic gap summaries → low-rank EOF/PCA spatial branch → small state-space/residual corrections → optional sparse same-month TWS assimilation only after legality/value checks → tiny TCN/GRU only if residual diagnostics justify it → provenance-safe external data only after the provided-data pipeline plateaus → small robust ensemble.**

The most important finding from the research phase is that this challenge is **not** an ordinary one-step tabular regression problem.

It is better understood as:

> **forecasting a persistent, seasonal, spatially correlated global water-storage state when the key autoregressive state observation, TWS, becomes stale for 1–7 months, while hydrometeorological exogenous variables remain fresh.**

The main competition risk is therefore not choosing the wrong booster depth or neural architecture. It is building an invalid validator or allowing a subtle TWS leak that makes local CV unrealistically easy.

The current research phase is considered saturated enough to begin implementation because the independent research branches—competition rules, hydrology, rolling validation, tabular formulation, low-rank spatial modeling, state-space methods, deep-model triage, external data, sparse observation assimilation, leaderboard discipline, and an adversarial falsification pass—converged on the same practical hierarchy.

---

# 2. Official competition specification

## 2.1 Core dataset

Official Zindi Data page reports:

- **Train rows:** 2,154,021
- **Test rows:** 280,961
- each record uniquely identified by latitude, longitude, and date;
- provided features include:
  - `TWS_t` — Total Water Storage at source month `t`;
  - `SPEI_01` — 1-month Standardized Precipitation-Evapotranspiration Index;
  - `SPEI_03` — 3-month SPEI;
  - `SPEI_06` — 6-month SPEI;
  - `SPEI_12` — 12-month SPEI;
  - `SOIL_MOISTURE_t` — near-surface soil moisture at source month `t`;
- training target is TWS at the next **calendar** month, `t+1`, where available.

Zindi explicitly encourages approaches that model both:

- temporal dynamics;
- global spatial patterns.

Source:  
`https://zindi.world/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge/data`

## 2.2 Target definition

Organizer clarification:

> `target(t)` corresponds to `TWS_t` at calendar month `t+1`, where available.

This wording is critical.

It means **label construction must use exact calendar joins**, not row-order operations such as `groupby(...).shift(-1)` unless the code first proves that the next row is the next calendar month.

GRACE has real missing monthly solutions and the GRACE/GRACE-FO mission gap. Therefore:

```text
next row != necessarily next calendar month
```

Any pipeline that blindly shifts rows can silently create incorrect labels or incorrect effective horizons.

## 2.3 Test TWS masking

Zindi states that approximately **66.5%** of test `TWS_t` is masked.

Because the test retains repeated months with stale TWS, the effective TWS forecast horizon ranges from:

```text
h = 1, 2, 3, 4, 5, 6, 7 months
```

The prediction at source month `t` is still for target month `t+1`. What changes is the age of the most recent legal TWS observation.

This distinction must stay explicit:

```text
forecast lead of target: always next calendar month
age of latest TWS state: 1–7 months relative to target
```

## 2.4 Strict causality rule

Organizer clarification:

> For predictions at month `t`, only information available at or before `t` may be used. Future observed values must not be used to fill or infer masked TWS values.

Therefore the following are fundamentally invalid for prediction features:

- future TWS;
- backward filling from a future TWS observation;
- two-sided TWS interpolation;
- fixed-interval Kalman smoothing;
- centered rolling operations involving future months;
- future-aware STL/decomposition;
- full-series target-derived preprocessing reused in earlier folds;
- any product containing future GRACE/TWS information.

Organizer/rules source:  
`https://zindi.world/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge/discussions/34093`

## 2.5 External-data rule

Zindi allows relevant external data only when it is:

- freely available;
- operationally available;
- available within one month of acquisition;
- available at the stated prediction time;
- not directly or indirectly contaminated by future GRACE/TWS information;
- fully documented for reproducibility.

The current rules explicitly mention Copernicus resources such as:

- European Drought Observatory;
- Global Drought Observatory;
- Copernicus Climate Data Store.

## 2.6 Evaluation

### Phase One — predictive score

- metric: **Root Mean Squared Error (RMSE)**;
- predictive leaderboard component: **50%** of final evaluation.

### Phase Two — report evaluation for top performers

- AI Trustworthiness: **30%**;
- Innovation and practicality: **20%**.

Trustworthiness topics explicitly include:

- data/model bias;
- transparency;
- reusability;
- sustainability and efficiency.

Evaluation source:  
`https://zindi.world/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge/discussions/34542`

## 2.7 Leaderboard mechanics

- public leaderboard: approximately **30%** of test rows;
- private leaderboard: remaining approximately **70%**;
- maximum submissions: **5 per day**;
- maximum total submissions: **200**;
- before close, participants select **2** submissions for private scoring;
- if no two are selected, Zindi uses the two best public submissions;
- top 10 must submit reproducible model/code/report within 72 hours of request;
- seeds must be fixed and reruns must reproduce the score;
- AutoML is not permitted;
- only publicly available/open-source tools are permitted;
- competition documentation states paid services or free trials requiring a credit card are not accepted for reproducibility.

This makes public-leaderboard hill-climbing especially dangerous.

---

# 3. Dataset forensics performed during research

The research phase included **read-only structural analysis** of the provided CSVs. No model was trained and the starter notebook was not run as a training experiment.

## 3.1 Test mask geometry

Observed local structure:

- Test contains 18 source months;
- approximately 186,913 test rows have missing `TWS_t`;
- measured missing rate ≈ **66.526%**;
- missingness is overwhelmingly **calendar-block structured**, not IID per-row missingness.

Approximate sequence of month-level TWS visibility includes patterns resembling:

```text
O
O M M
O M M M
O M M M M M M
O
O M
```

where:

- `O` = current TWS substantially observed;
- `M` = current TWS mostly masked.

This is why random Bernoulli masking is not a faithful validator.

## 3.2 Exact row-level effective horizon distribution

Using the actual row-level visibility pattern and each location’s last legal TWS anchor, the exact test counts found were:

| Effective TWS horizon `h` | Rows | Share |
|---:|---:|---:|
| 1 | 94,048 | 0.334737 |
| 2 | 62,576 | 0.222721 |
| 3 | 46,777 | 0.166489 |
| 4 | 31,076 | 0.110606 |
| 5 | 15,560 | 0.055381 |
| 6 | 15,479 | 0.055093 |
| 7 | 15,445 | 0.054972 |

Weighted mean effective horizon:

```text
2.714341 months
```

This is very close to the intuitive month-block ratio:

```text
6 : 4 : 3 : 2 : 1 : 1 : 1
```

These exact frequencies should be used for:

- training-horizon sampling;
- proxy-CV weighting;
- full-CV aggregate scoring;
- horizon-level diagnostics.

## 3.3 Same-month sparse visible TWS observations

In months that are mostly masked, a small number of current-month TWS values remain visible across the global grid.

Observed order of magnitude:

```text
~4 to ~65 visible cells out of ~15,600–15,700 locations
```

This corresponds to only approximately:

```text
0.026% to 0.417% observed
```

or:

```text
99.58% to 99.97% missing
```

This fact motivated the optional low-rank sparse-assimilation research branch described later.

## 3.4 Location count and panel structure

Read-only profiling indicated approximately:

```text
15,715 unique spatial locations
```

Train spans approximately:

```text
138 source months
```

Therefore the data should not be mentally treated as 2.15 million independent observations.

It is closer to:

```text
~138 global monthly TWS fields × ~15.7k strongly spatially dependent cells
```

That pseudo-sample-size issue strongly affects neural-model risk.

## 3.5 Target identity finding

For source months where the next calendar TWS row exists, read-only comparison found that:

```text
Target(cell, t) == TWS(cell, t+1)
```

exactly for the compared matched pairs.

This is scientifically expected and matches the official target definition.

It is also the single most dangerous leakage path in the challenge.

For example:

```text
Target(cell, September) == TWS(cell, October)
```

Therefore, if October `TWS_t` is synthetically masked in validation but the feature-generation code can inspect September’s label, it can silently reconstruct the hidden October state.

### Non-negotiable design rule

> **Feature-generation functions must never receive the target column.**

The label table must be isolated and joined only at the final supervised-example assembly step.

## 3.6 GRACE mission gap

The GRACE → GRACE-FO transition includes a real mission-level gap.

No-row calendar gaps must not be interpreted as extra test forecast horizons.

Effective horizon is determined by the intended competition sequence of available source months and legal TWS state, not simply by row distance or naive calendar elapsed time across absent mission data.

---

# 4. Persistence and difficulty diagnostics

Read-only diagnostics were used to calibrate the problem before model training.

## 4.1 One-step persistence

Using current `TWS_t` as the next-month prediction:

```text
RMSE ≈ 0.5724
Corr(TWS_t, Target_t) ≈ 0.8033
```

This confirms that storage persistence is a very strong base signal.

## 4.2 Persistence degradation with stale TWS

Synthetic horizon diagnostics showed approximately:

| TWS age horizon | Persistence RMSE, broad history | Approx. later/post-2012 RMSE |
|---:|---:|---:|
| 1 | 0.572 | 0.655 |
| 2 | 0.674 | 0.774 |
| 3 | 0.726 | 0.831 |
| 4 | 0.765 | 0.855 |
| 5 | 0.798 | 0.850 |
| 6 | 0.827 | 0.846 |
| 7 | 0.866 | 0.883 |

Key interpretation:

- h=1 is substantially easier than the competition-wide problem;
- performance degrades monotonically or near-monotonically as TWS becomes stale;
- later historical periods appear harder/nonstationary;
- a validation design dominated by h=1 can dramatically overstate competition skill.

Using the actual test horizon mix, a later-period persistence baseline would be materially worse than the starter notebook’s h=1-style baseline.

---

# 5. Scientific structure of Total Water Storage

## 5.1 What TWS represents

GRACE/GRACE-FO terrestrial water storage anomaly reflects integrated water storage including major contributions from:

- groundwater;
- soil moisture;
- surface water;
- snow/ice storage components, depending on region/product.

TWS is therefore a **state variable with memory**, not simply an instantaneous response to precipitation.

## 5.2 Expected temporal behavior

Research reviewed during the saturation phase consistently indicates:

- strong persistence;
- strong annual seasonality;
- meaningful semiannual structure in some regions;
- varying lag between precipitation/climate forcing and TWS response;
- longer hydrological memory in groundwater-dominated/drier/high-latitude regimes;
- shorter response in some tropical/wetter regimes;
- spatially heterogeneous trends that may reflect climate, hydrology, anthropogenic extraction, or long-term variability.

Important consequence:

> We should not blindly detrend every cell and assume the trend is nuisance.

Trend is potentially predictive signal, but it must be estimated causally.

## 5.3 Annual/semiannual baseline importance

The final adversarial review highlighted that trees cannot extrapolate a secular time trend beyond the maximum training time in the same way a parametric trend branch can.

Therefore an explicit structural baseline should be tested early:

\[
B_i(t) = \alpha_i + \beta_i t
       + a_i\sin(2\pi t/12) + b_i\cos(2\pi t/12)
       + c_i\sin(4\pi t/12) + d_i\cos(4\pi t/12)
\]

where parameters are estimated only from fold-past data.

A useful forecast formulation is:

\[
\widehat{TWS}_{i,t+1}
= B_i(t+1)
+ \rho_h\,[TWS_{i,s}-B_i(s)]
\]

where `s` is the last legally observed TWS month.

This should be tested before broad tree tuning because it costs almost nothing and gives us a principled extrapolating seasonal/trend reference.

## 5.4 Spatial support is coarser than apparent grid size

GRACE products are often distributed on grids finer than the native geophysical information support.

Research reviewed from JPL/CSR documentation emphasizes that:

- gridded mascon products can be displayed around 0.5°;
- underlying/native mascon or effective resolution is substantially coarser;
- neighboring grid cells are strongly dependent;
- basin/regional interpretation is often more scientifically meaningful than treating every fine-grid cell as independent.

Consequence:

> Extremely high-capacity models can appear to have millions of training samples when the true number of independent temporal/spatial degrees of freedom is much smaller.

---

# 6. Correct forecasting formulation

## 6.1 Notation

Let:

- `t` = current/source month;
- target month = `t+1`;
- `s` = latest month at or before `t` for which the location’s TWS is legally visible;
- effective horizon:

\[
h = \text{months}(t+1) - \text{months}(s)
\]

For the competition:

```text
h ∈ {1,2,3,4,5,6,7}
```

## 6.2 Preferred target

Rather than forcing the main model to learn the entire absolute spatial storage field, use the last legal TWS anchor and predict the accumulated change:

\[
\Delta_{i,t,h}
= TWS_{i,t+1} - TWS_{i,s}
\]

Then reconstruct:

\[
\widehat{TWS}_{i,t+1}
= TWS_{i,s} + \widehat{\Delta}_{i,t,h}
\]

Benefits:

- directly generalizes persistence;
- makes `h` explicit;
- reduces burden of learning static location level;
- aligns with storage-state dynamics;
- allows fresh exogenous variables through source month `t`;
- avoids recursive prediction through hidden TWS months.

## 6.3 Critical fresh-exogenous rule

The adversarial review identified a common pseudo-horizon construction error:

> Do **not** build an h-month example using only features from the stale anchor month `s`.

At test prediction month `t`, TWS may be stale, but current exogenous variables are available through `t`.

Correct pseudo-example:

```text
last legal TWS: from s
TWS-derived lags/rolls: must obey mask and stop at/before s
SPEI/soil/current calendar information: may use legal information through source month t
target: TWS at calendar t+1
```

This distinction is essential.

---

# 7. Validation system — highest priority implementation

## 7.1 Why starter-style temporal holdout is insufficient

A normal train/validation split that gives the model true `TWS_t` in every validation month measures mostly h=1 forecasting.

The test set frequently withholds `TWS_t`, so the actual task distribution is h=1…7.

Therefore the first serious engineering task is **not model tuning**. It is a competition-faithful online validator.

## 7.2 Online simulation requirement

For every validation block:

1. freeze all fold-fitted artifacts at the training cutoff;
2. transplant a historical equivalent of the real test TWS-visibility pattern;
3. iterate source months chronologically;
4. at each month, expose only TWS values that the simulated mask allows;
5. make predictions using the stale TWS state + legal exogenous variables;
6. keep hidden validation TWS exclusively in the scorer;
7. do **not** feed hidden truth into next month’s feature state;
8. if a later month is designated as legally observed by the transplanted mask, reveal it only at that later month;
9. never let future visibility geometry enter features.

## 7.3 Target-time fold boundary

Fold inclusion must be based on target timestamp, not merely source timestamp.

For a training cutoff `C`, a pair `(source=t, target=t+1)` can be used to fit only if:

```text
target_date <= C
```

This prevents the last source row before a cutoff from carrying a target that lies inside the validation future.

## 7.4 Exact horizon-weighted fold metric

Let `q_h` be the exact test row share for horizon `h`.

For validation fold `f`:

\[
MSE_f = \sum_{h=1}^{7} q_h MSE_{f,h}
\]

Overall headline CV:

\[
CV\_RMSE = \sqrt{\frac{1}{F}\sum_f MSE_f}
\]

Do not average horizon RMSEs directly.

Do not use inverse-variance horizon weights.

The leaderboard is plain rowwise RMSE.

## 7.5 Area weighting caution

Geophysical studies often use latitude/cosine area weighting.

Competition loss is plain rowwise RMSE.

Therefore:

- area-weighted EOF/PCA can be an auxiliary scientific branch;
- **unweighted score-aligned EOF/PCA must also be tested**;
- do not make cos(latitude) weighting the only spatial representation.

## 7.6 Recommended two-speed CV

### Tier A — proxy CV

Purpose: rapidly kill weak ideas.

Suggested design:

- 2 recent pseudo-test temporal blocks;
- fixed 20–30% location subset;
- spatial subset selected once with a stable hash and, preferably, stratified geographic coverage;
- preserve complete temporal trajectories for selected locations;
- fixed cheap model parameters;
- one factor changed per experiment.

Kill rule:

```text
two proxy failures -> stop spending compute on the idea
```

### Tier B — full rolling CV

Use several non-overlapping approximately competition-length blocks.

A preliminary architecture from the research pass was:

```text
development fold 1 ≈ source-month indices 67–84
development fold 2 ≈ 85–102
development fold 3 ≈ 103–120
lockbox          ≈ 121–138
```

Exact indices must be aligned to the real sorted calendar-month sequence during implementation.

The newest block should remain untouched during ordinary development.

Promotion rule:

- improvement in at least 2/3 development folds;
- no material degradation in the newest dev fold;
- lockbox inspected only for finalists.

## 7.7 Block bootstrap / uncertainty

Rows are strongly spatially and temporally dependent.

Do not estimate experiment significance from IID row-level errors.

For uncertainty around model deltas:

- aggregate loss at target-month level;
- use moving-block bootstrap, approximately 7-month blocks as a starting point;
- inspect paired per-month squared-error deltas between challenger and incumbent.

---

# 8. Leakage and causality specification

## 8.1 Allowed

Provided all artifacts are fitted on fold-past only:

- last TWS observation at or before current source month;
- months since last observed TWS;
- current/source-month SPEI;
- current/source-month provided soil moisture;
- one-sided historical rolling features;
- causal TWS lags derived after masking;
- deterministic month/year features;
- latitude/longitude;
- deterministic geographic IDs derived only from coordinates;
- past-only EOF/PCA/climatology/trend models;
- causal Kalman **filter** state;
- same-month sparse visible TWS across other cells appears allowed under literal time wording, but see unresolved organizer-risk note below.

## 8.2 Questionable / default-deny until provenance or organizer clarification

- final consolidated ERA5 used retrospectively as a proxy for historical ERA5T;
- complete source-month ERA5 monthly means if operational issue time is ambiguous;
- same-month cross-location test TWS assimilation, because temporal legality appears clear but row-independence expectations are not explicitly documented;
- GDO historical anomaly archives that may have been reprocessed with later climatological baselines.

## 8.3 Forbidden

- `Target` anywhere inside feature-generation code;
- `Target(t-1)` used to recreate masked `TWS_t`;
- future TWS;
- future test rows used to infer current TWS;
- backward fill;
- centered rolling windows;
- future-aware interpolation;
- Kalman/RTS smoothing as online features;
- bidirectional RNN/BRITS-style inference using future time points;
- full-series target-derived PCA/EOF reused in earlier validation folds;
- full-series target-derived normalization/decomposition reused in earlier folds;
- remaining gap length / next observation date / total future mask-block length;
- GLDAS-2.2 GRACE data-assimilation products;
- GDO/other GRACE TWS products that encode the target family;
- any external variable released later than allowed operational window;
- AutoML.

## 8.4 Required unit tests

### Target identity test

For every train `(cell,t)` where the next calendar TWS row exists:

```python
assert abs(Target[cell,t] - TWS[cell,t+1]) < tolerance
```

This catches calendar alignment bugs.

### Target-blind feature builder

Feature functions must reject or never receive the target table.

### Prefix/suffix invariance

Generate features through cutoff `C`.

Append arbitrary future rows or perturb future observations.

Features at dates `<= C` must remain exactly unchanged.

### Streaming == batch

For the same visibility mask, chronological streaming feature generation and batch causal generation must be identical.

### Synthetic horizon state assertion

For an h-horizon example, assert that every TWS-derived feature comes only from permitted observed TWS dates.

### Future mask geometry invariance

Change the visibility pattern of future validation months.

Current features must not change.

### Hidden-truth unreachable test

Replace hidden validation TWS with random noise.

Predictions/features for that month and all later masked months must remain unchanged until the simulated mask legally reveals a new TWS observation.

### Fold label-time test

Every supervised training row must satisfy:

```text
target_timestamp <= training_cutoff
```

### EOF/climatology suffix invariance

Past EOF/climatology outputs must remain unchanged if future target data is altered.

### External provenance gate

Every external feature should carry metadata:

```text
source
variable
valid_time
release_time / operational latency
archive/product version
whether product contains/assimilates GRACE
license
download/retrieval recipe
```

Unknown provenance => reject feature.

---

# 9. Mandatory baseline ladder

Before training complex ML, implement and score these in exact rolling-mask CV.

## B0 — persistence

```text
prediction = last legal observed TWS
```

This must be reported by h=1…7.

## B1 — 12-month seasonal naive

Where legally available:

```text
prediction = TWS from same calendar month one year earlier
```

Variants:

- exact lag-12;
- lag-12 blended with persistence;
- lag-12 + shrunk linear trend correction.

## B2 — harmonic + trend structural baseline

Fit fold-past only:

- intercept;
- linear trend;
- annual sine/cosine;
- semiannual sine/cosine.

Optionally shrink location-level parameters toward regional/global values if unstable.

## B3 — seasonal-anomaly persistence

Forecast structural baseline at `t+1`, then persist the most recent observed anomaly with an h-dependent shrinkage `rho_h`.

## B4 — simple hydrologic-bridge ridge/ARX

Predict change since stale TWS from:

- `h`;
- `Δsoil` from stale observation month to current source month;
- current vs stale SPEI;
- `SPEI_01 - SPEI_03`;
- `SPEI_01 - SPEI_06`;
- `SPEI_01 - SPEI_12`;
- season;
- optional simple latitude/regime terms.

This near-zero-cost model is an important falsifier for whether complex GBDT nonlinearities are truly necessary.

---

# 10. Main model — global pooled direct residual LightGBM

## 10.1 Primary target

```text
delta = target_TWS(t+1) - last_legal_TWS(s)
prediction = last_legal_TWS(s) + delta_hat
```

## 10.2 Why pooled global training

Each location has only around 138 monthly fields.

Separate complex local models would have too little temporal data.

Global pooling lets the model share:

- seasonal response;
- SPEI/soil response;
- h-dependent change dynamics;
- regional similarities;
- spatial level/structure.

## 10.3 Feature groups, ordered by priority

### P0 — state and horizon

- `last_observed_TWS`;
- `h` / age;
- source month;
- target month;
- elapsed time index;
- latitude;
- longitude.

### P1 — causal TWS history

All derived **after** applying the simulated TWS visibility mask:

- prior observed TWS lags around 1,2,3,6,12 months where legal;
- same-season/lag-12 TWS anchor;
- differences between recent legal TWS observations;
- 3/6/12 historical one-sided TWS means;
- historical standard deviations;
- robust recent slope;
- anomaly relative to past-only seasonal baseline.

No feature may read a hidden TWS inside the simulated gap.

### P2 — provided hydrometeorology

- current `SPEI_01`;
- current `SPEI_03`;
- current `SPEI_06`;
- current `SPEI_12`;
- current `SOIL_MOISTURE_t`;
- causal lagged values;
- first differences;
- 3/6/12-month one-sided summaries where useful.

### P3 — hydrologic gap summaries

These are especially important because they describe what happened while TWS was stale.

For source `t` and last TWS month `s`:

- soil current minus soil at `s`;
- mean soil over `s..t`;
- min/max soil over gap;
- soil slope over gap;
- current SPEI minus SPEI at `s`;
- mean/min/max SPEI over gap;
- gap precipitation-stress trend proxies;
- interactions between `h` and SPEI timescale;
- interactions between `h` and soil change.

### P4 — season and spatial regime

- `sin(month)`;
- `cos(month)`;
- optional second harmonic;
- hemisphere;
- absolute latitude;
- coarse geographic bins;
- deterministic location ID if useful;
- optional climate-regime cluster fit without future target leakage.

## 10.4 Pooled vs horizon-specific models

Start with:

```text
one pooled LightGBM with h as a feature
```

Then compare:

```text
7 horizon-specific LightGBMs
```

Then test:

```text
pooled + direct-h specialist blend
```

Do not assume one architecture wins every horizon.

## 10.5 Efficient synthetic horizon sampling

Full seven-way expansion would produce roughly 15M training examples.

Do **not** start there.

Initial strategy:

- assign one synthetic horizon to each eligible target;
- sample h according to exact test frequencies;
- keep dataset near original ~2.15M-row scale;
- optionally repeat with 2–3 independent deterministic seeds if this becomes useful.

Alternative later strategy:

- train h-specific models serially;
- never materialize all 7 horizons simultaneously.

## 10.6 LightGBM initial search region

Research recommendation:

```text
objective: L2 / regression
learning_rate: ~0.03–0.08
num_leaves: {31, 63, 127}
max_depth: roughly 8–12 or leaf-limited
min_data_in_leaf: ~500–5000
feature_fraction: ~0.75–0.95
bagging_fraction: ~0.75–0.95
bagging_freq: 1
lambda_l2: ~1–20
lambda_l1: ~0–2
max_bin: {127, 255}
boosting rounds: ~1000–4000 with temporal early stopping
```

Tuning priority:

1. leaf complexity / min leaf size;
2. learning-rate + rounds;
3. sampling;
4. regularization;
5. broad HPO only after formulation/features are stable.

## 10.7 Alternative tree families

After LightGBM stabilizes:

- XGBoost histogram method can provide model-family diversity;
- CatBoost is interesting mainly for high-cardinality location/category treatment;
- HistGradientBoosting remains a useful simple control.

### CatBoost caution

If duplicated synthetic horizons share the same target, target-statistic/CTR encodings can leak labels across duplicates.

Safer CatBoost experiments:

- one sampled horizon per original target;
- or separate horizon models;
- chronological/cross-fitted target statistics only.

---

# 11. Spatial and low-rank branch

## 11.1 Motivation

The global TWS field is highly spatially structured and low dimensional relative to 15.7k cells.

Direct TWS forecasting literature has used PCA/EOF decomposition successfully.

Represent monthly field:

\[
TWS_t \approx \mu + U z_t
\]

where:

- `U` = past-only spatial basis;
- `z_t` = low-dimensional coefficient vector.

## 11.2 Fold fitting rule

EOF/PCA must be re-fit inside every training prefix.

Never fit a global basis on all 2002–2015 targets and then score earlier folds.

## 11.3 Candidate ranks

Start with:

```text
k ∈ {4, 8, 12}
```

Assimilation-specific branch can explore:

```text
k ∈ {2,4,8,12,16}
```

Choose rank by causal target RMSE, not only by explained variance.

## 11.4 Unweighted vs area-weighted EOF

Because leaderboard loss is rowwise RMSE:

- unweighted EOF is score-aligned and mandatory;
- area-weighted/cos(latitude) EOF may be a scientific diversity branch;
- do not rely solely on area weighting.

## 11.5 Coefficient forecast

Cheap initial factor models:

- diagonal AR(1);
- ridge ARX using projected current exogenous variables;
- annual/semiannual harmonic terms;
- avoid unrestricted VAR with many modes under ~138 months.

Trigger to keep factor modeling:

- top few factors explain a meaningful share of OOF residual field;
- factor autocorrelation is stable;
- relationships with exogenous drivers persist across folds.

---

# 12. State-space complement

## 12.1 Residual Kalman correction

Let `base` be the causal LightGBM prediction.

When actual TWS is legally observed, form base residual and track a latent persistent bias:

\[
s_{t+1}=\phi s_t + \beta'z_t + w_t
\]

\[
r_t=s_t+v_t
\]

At masked months:

- skip the measurement update;
- propagate state forward;
- add predicted latent bias to the base forecast.

Trigger:

- OOF residual ACF persists;
- prior residual predicts next error;
- base error grows systematically with TWS age.

Kill if:

- `phi ≈ 0`;
- process/measurement variance degenerates;
- no h≥2 gain;
- <~0.2% rolling-CV benefit and no ensemble diversity.

## 12.2 EOF-factor ARX/DLM

For each low-rank factor:

- AR(1) dynamics;
- ridge on 2–6 projected exogenous scores;
- optional deterministic annual/semiannual terms.

Avoid high-order AR and unrestricted large VAR due short temporal sample.

## 12.3 Regional structural residual correction

If OOF errors show coherent geographic seasonal bias, fit partially pooled regional correction:

- local level;
- optional damped/fixed slope;
- annual harmonic;
- semiannual harmonic;
- noise.

Do not fit separate unconstrained state-space models for all 15.7k cells.

## 12.4 Filter vs smoother

**Allowed for online features:** Kalman filtering using observations `<= t`.  
**Forbidden:** fixed-interval/RTS smoothing that uses observations after `t`.

---

# 13. Sparse same-month TWS assimilation research branch

## 13.1 Why ordinary gap filling is a bad idea

Mostly masked months retain only approximately 4–65 visible TWS cells out of ~15.7k.

Plain gappy POD/DINEOF attempts to infer a full field from an almost completely missing snapshot.

At 99.6–99.97% missing:

- `m < rank` can be underdetermined;
- `m ≈ rank` can be badly conditioned;
- local/clumped visible cells may have low leverage on global modes;
- DINEOF becomes leakage-prone if later months participate.

The reviewed DINEOF literature typically operates at missingness levels dramatically less extreme than this challenge; some applications discard exceptionally sparse snapshots.

## 13.2 Better formulation — forecast-error assimilation

Do not reconstruct raw TWS from scratch.

Instead:

1. produce a strong causal prior current field `p_t` from the baseline/GBDT;
2. form rolling-OOF historical current-field errors:

\[
e_t = TWS_t - p_t
\]

3. fit a past-only EOF basis `U_r` to forecast errors;
4. for current visible cells `O`, compute innovations:

\[
d = y_O - p_O
\]

5. perform a strongly regularized low-rank update:

\[
a^+ = (U_O^T R^{-1}U_O + \Lambda^{-1})^{-1}U_O^T R^{-1}d
\]

6. updated current field:

\[
p_t + U a^+
\]

7. propagate the correction one month using factor persistence or a tiny learned transition;
8. add this correction to the next-month base forecast.

## 13.3 Candidate regularization

Ranks:

```text
r = {2,4,8,12,16}
```

Prior/noise scale multipliers:

```text
{0.25, 1, 4, 16}
```

Optional global correction shrinkage:

```text
beta = {0.25, 0.5, 0.75, 1}
```

## 13.4 Minimal assimilation ladder

```text
E0: base forecast only
E1: raw gappy POD negative control
E2: prior + error-EOF ridge update
E3: Bayesian Lambda/R update
E4: diagonal factor persistence rho, then small transition G
E5: posterior correction/features supplied to pooled model only if robust
```

## 13.5 Even cheaper pre-test

Before full EOF assimilation, try a shrunk common-mode or regional residual offset derived from the legal same-month visible TWS anchors.

If this simple calibration does not help in faithful CV, kill elaborate assimilation early.

## 13.6 Rule status

Using current-month observations across locations appears temporally legal under the stated `<= t` rule.

However, no explicit organizer statement was found that specifically endorses using one test row’s visible TWS to modify another row’s same-month forecast.

Therefore:

```text
status = HOLD / organizer clarification preferred before final reliance
```

---

# 14. Deep-learning triage

The research conclusion is deliberately conservative.

The 15.7k spatial rows per month are not 15.7k independent long sequences. We have only ~138 global monthly fields.

High-capacity deep sequence/spatial models therefore face severe pseudo-replication and overfitting risk.

## 14.1 Rank order if we later test neural models

### 1. Small causal TCN

Best first neural experiment.

Reasons:

- causal convolutions;
- stable long-ish receptive fields;
- relevant groundwater forecasting evidence;
- lower complexity than transformers;
- straightforward mask/age inputs.

### 2. Small mask-aware GRU / GRU-D-inspired model

Use:

- unidirectional recurrent processing;
- TWS observation mask;
- time-since-last-TWS;
- fresh exogenous inputs.

Literal GRU-D’s decay-to-global-mean assumption may be inappropriate for strongly seasonal/persistent TWS, so adapt cautiously.

### 3. N-HiTS

Potential one-shot conditional baseline, but h≤7 is not where its long-horizon advantage is strongest.

### 4. Tiny spatial CNN

Only if OOF residual maps show spatial autocorrelation not captured by geography/EOF.

## 14.2 Low priority / likely kill

- TFT;
- PatchTST;
- TimesNet;
- generic Transformer;
- heavy CNN/ConvLSTM;
- GNN/Graph WaveNet without a clearly defensible graph and residual trigger.

## 14.3 Neural promotion threshold

Do not spend multiple days on architecture search.

Require roughly:

- ≥0.5% stable standalone RMSE gain across at least 2 temporal folds, **or**
- ≥0.2–0.3% stable ensemble gain with real residual diversity.

Kill after one compact config plus one extra seed if promising.

Kill immediately if:

- gain is h=1-only;
- result is seed unstable;
- compute is >2–3× TCN with no better blend gain.

---

# 15. External-data audit

## 15.1 General rule

External data enters only after provided-data modeling plateaus.

Every variable needs a provenance manifest proving it could have existed operationally at prediction time.

## 15.2 ERA5 / ERA5T

ECMWF documents ERA5T as the near-real-time preliminary stream, roughly ~5 days behind real time, while consolidated ERA5 is produced later and can overwrite preliminary values.

Important unresolved issue:

> Today’s historical final ERA5 archive is not automatically equivalent to an archived snapshot of what ERA5T looked like at the historical prediction time.

Therefore:

```text
historical final ERA5 as ERA5T proxy = default-deny unless provenance/organizer ruling is defensible
```

Lagged variables with clearly compliant release timing are safer than same-month complete monthly aggregates.

## 15.3 ERA5-Land

Scientifically promising variables:

- deeper soil water layers, especially ~28–100 cm and ~100–289 cm;
- precipitation accumulations;
- snow/SWE/snowfall/melt where climate makes them relevant;
- actual ET;
- runoff.

But the same historical-vintage/latency issue must be resolved.

## 15.4 GLDAS

Research found:

- GLDAS-2.1 main production latency around several months;
- Early Product approximately ~1.5 months;
- both exceed or threaten the one-month operational requirement;
- GLDAS-2.2 GRACE-DA assimilates GRACE TWS and is target contaminated.

Current verdict:

```text
GLDAS-2.1 = kill for this competition unless organizer explicitly rules otherwise
GLDAS-2.2 GRACE-DA = forbidden
```

## 15.5 GDO/EDO

Potential products reviewed:

- SPI;
- soil-moisture anomaly;
- fAPAR anomaly;
- drought composites;
- GRACE TWS anomaly.

Problems:

- some anomaly archives use climatological baselines that can include future years relative to early historical dates;
- archives may be reprocessed;
- several signals duplicate provided SPEI/soil information;
- GRACE TWS anomaly is direct target-family leakage.

Current priority:

```text
raw/mechanistically new variables > derived composite drought indicators
```

## 15.6 External-feature ranking if provenance is approved

1. raw precipitation and causal 1/2/3/6/12-month accumulations;
2. deeper soil water/storage layers;
3. snow/SWE in cold/high-elevation regions;
4. ET + runoff/subsurface runoff;
5. lagged climate modes such as ENSO/SOI/PDO/NAO/IOD with region interactions;
6. raw/causally derived vegetation signal such as fAPAR.

Avoid external-data fishing before the internal causal pipeline is strong.

---

# 16. Experiment ladder — ordered by information gained per compute

## Stage 0 — infrastructure, no serious model training

- exact calendar label join;
- availability ledger;
- exact test row-level h derivation;
- rolling-mask simulator;
- target isolation;
- leakage unit tests;
- metric implementation;
- run logger;
- deterministic seeds/config hashes.

Exit condition:

> all baseline sanity checks and leakage tests pass.

## Stage 1 — cheap baselines

- persistence;
- lag-12 seasonal naive;
- trend-adjusted lag-12;
- harmonic + trend;
- seasonal-anomaly persistence;
- hydrologic-bridge ridge.

Goal:

> establish a trustworthy performance floor and expose validation bugs.

## Stage 2 — core LightGBM formulation

Run one-factor ablations:

1. absolute target vs ΔTWS target;
2. last-TWS + h;
3. causal TWS history;
4. provided current SPEI/soil;
5. SPEI/soil historical summaries;
6. hydrologic gap summaries;
7. seasonal features;
8. geography/location features;
9. pooled h vs h-specific;
10. modest model-complexity tuning.

## Stage 3 — spatial low-rank branch

- unweighted EOF rank 4/8/12;
- factor AR(1);
- factor ARX;
- optional area-weighted EOF diversity branch;
- evaluate standalone and blend residual correlation.

## Stage 4 — simple statistical corrections

- residual Kalman filter;
- regional seasonal residual model;
- simple sparse-anchor common-mode calibration.

## Stage 5 — optional sparse EOF assimilation

Only if:

- simple anchor calibration shows signal;
- legality is confirmed sufficiently;
- forecast-error EOF basis is stable.

## Stage 6 — small neural branch

Only if core residuals indicate remaining nonlinear temporal sequence structure.

Order:

```text
TCN -> mask-aware GRU -> N-HiTS
```

## Stage 7 — external data

Only provenance-safe, mechanistically new variables.

## Stage 8 — final ensemble

Prefer 2–3 genuinely different models.

Likely candidates:

```text
GBDT + EOF/state-space + optional TCN/assimilation
```

Try equal weights before fitting weights.

---

# 17. Experiment discipline and stopping rules

## 17.1 Change one factor at a time

Do not simultaneously change:

- target formulation;
- feature set;
- model family;
- CV folds;
- hyperparameters.

Otherwise we learn nothing from the run.

## 17.2 Before each run write expected effect

Examples:

```text
Hypothesis: gap soil-change feature should mostly improve h>=3.
Hypothesis: lag-12 seasonal anchor should help strong-seasonality regions.
Hypothesis: coordinate removal will reveal whether location memorization is carrying recent-fold skill.
```

Then compare diagnostics against the prediction.

## 17.3 Promotion criteria

General default:

- positive full-dev gain in at least 2/3 folds;
- no strong worsening in newest fold;
- improvement bigger than temporal/bootstrap noise;
- or clear complementary residual pattern that improves a robust blend.

## 17.4 Kill criteria

- two proxy failures;
- three adjacent HPO settings below material improvement;
- expensive family with no gain and residual correlation ≳0.995 to incumbent;
- gain exists only in h=1 while test difficulty is h-mixed;
- gain reverses badly in later folds;
- suspiciously huge gain without a scientifically plausible mechanism => leakage audit first.

## 17.5 Research saturation stop criterion

The final adversarial worker proposed a useful project-level stop rule.

Research/idea hunting is saturated when:

- calendar/causality tests pass;
- exact rolling-mask CV covers:
  - persistence;
  - seasonal-anomaly+trend;
  - lag-12 naive;
  - hydrologic-bridge ridge;
  - core direct residual tree model;
  - simple anchor calibration;
- no remaining cheap method improves core/blend by roughly **0.003–0.005 absolute RMSE** across at least two outer temporal cutoffs, or by roughly ≥1% in a horizon bucket with ≥5% test weight;
- no legal ≤1-month external source adds a mechanistically new signal;
- two additional independent research passes yield only tuning variants rather than new model classes or causality corrections.

At that point, spend time on robustness/reproducibility rather than literature fishing.

---

# 18. OOF diagnostics required for every serious model

Store at minimum:

```text
run_id
fold
location_id / lat / lon
source_date
target_date
h
y_true
y_pred
residual
squared_error
region/regime tags
```

Report:

- overall CV RMSE;
- RMSE by h;
- bias by h;
- RMSE by fold;
- bias by fold;
- RMSE by target month/season;
- RMSE by region/climate regime;
- worst target months;
- worst locations/regions;
- largest squared-error contributors;
- paired monthly loss difference vs incumbent;
- residual correlation vs other candidate ensemble members;
- residual autocorrelation;
- spatial residual autocorrelation/map where feasible.

Important diagnostic questions:

- Does improvement persist toward recent time?
- Does it help longer horizons or merely h=1?
- Does the model introduce geographic bias?
- Does it improve drought/extreme states?
- Are gains concentrated in a tiny region?
- Is the challenger genuinely different enough to ensemble?

---

# 19. Ensembling rules

## 19.1 Candidate qualification

Do not ensemble models merely because they exist.

A component must have:

- strong standalone CV, or
- clear stable complementarity.

## 19.2 First combination

Try equal-weight 2-model and 3-model combinations.

Simple combinations are robust under small temporal sample and nonstationarity.

## 19.3 Learned weights

If later justified:

- nonnegative convex weights only;
- fit on earlier OOF;
- validate on later untouched fold;
- per-horizon weights only if stable across time;
- never fit blend weights on public leaderboard scores.

## 19.4 Meta-model leakage rule

Any stacker must be trained strictly on causal OOF predictions.

If a base model’s OOF is contaminated, the ensemble is contaminated.

---

# 20. Public leaderboard discipline

Public LB is only ~30% of test and private is ~70%.

Treat public LB as a weak external diagnostic, not an optimization dataset.

Typical daily submission budget should be well below the allowed five.

Purposeful submissions:

1. stable incumbent;
2. genuinely orthogonal challenger;
3. prevalidated blend.

Avoid:

- tiny hyperparameter probes;
- 0.05 weight sweeps;
- five nearly identical models/day;
- selecting final two solely because they are top two public scores.

Interpretation rule:

```text
one CV/LB disagreement -> likely leaderboard sampling noise
repeated coherent disagreement -> audit CV distribution and leakage assumptions
```

Final two private selections should likely be:

- strongest stable model;
- strong diverse hedge/blend.

---

# 21. Trustworthiness, report, and sustainability plan

Because 30% of final evaluation is trustworthiness, evidence must be generated during development.

## 21.1 Bias

Track performance by:

- latitude band;
- hemisphere;
- aridity/climate regime;
- geographic region/continent;
- season;
- TWS horizon;
- TWS state/extreme bins.

Investigate whether one region dominates overall RMSE gain.

## 21.2 Transparency

For tree models:

- gain/split importance;
- permutation importance where computationally feasible;
- SHAP on a representative sample;
- dependence plots for `h`, last TWS, soil change, SPEI scales;
- error examples.

For EOF/state-space:

- visualize leading spatial modes;
- show coefficient dynamics;
- explain filter update equations;
- show how missing TWS is handled.

## 21.3 Reusability

Keep:

- deterministic pipeline;
- environment manifest;
- data download instructions;
- one-command training where possible;
- clean separation between feature building, validation, training, inference, and submission;
- no notebook-only hidden state for final reproduction.

## 21.4 Sustainability

Record:

- CPU/GPU type;
- wall-clock time;
- peak RAM;
- model size;
- number of training examples;
- unnecessary compute avoided by proxy CV/kill rules;
- optional CodeCarbon emissions estimate.

This aligns model-selection discipline with the final report rubric.

---

# 22. Reproducibility and run logging

Every serious run should save:

```text
run_id
timestamp
git commit
config hash
data hash/version
mask pattern hash
fold cutoffs
spatial sample hash
random seed
feature groups
model family
hyperparameters
training rows
feature count
wall time
CPU/GPU
peak RAM
artifact size
overall CV
CV by horizon
CV by fold
regional metrics
bootstrap interval / paired monthly deltas
OOF residual correlations
public LB score if submitted
keep/kill decision
reason
```

Create a score-vs-compute Pareto frontier rather than rewarding tiny score gains achieved with massive compute.

---

# 23. Approaches currently killed or deferred

| Approach | Status | Reason |
|---|---|---|
| Starter notebook CV as main model selector | KILL | Does not reproduce h=1…7 stale-TWS mask geometry |
| Random KFold | KILL | Temporal leakage / invalid dependence assumptions |
| IID random 66.5% TWS masking | KILL | Wrong mask geometry |
| Row-order `shift(-1)` labels | KILL | Next row may not be next calendar month |
| Target column in feature builder | KILL | Can recreate masked TWS exactly |
| Future TWS/backfill/two-sided interpolation | KILL | Explicit rule violation |
| Full-series STL/PCA/normalization on target state | KILL | Future leakage |
| Recursive one-step GBDT as primary strategy | DEFER | Error propagation/exposure mismatch |
| Per-location complex models | KILL | Only ~138 months/location |
| Immediate 15M-row full h expansion | KILL | High compute with little initial information gain |
| Broad HPO before formulation ablations | KILL | Optimizes wrong problem |
| TFT/PatchTST/TimesNet/large Transformer early | KILL/DEFER | Pseudo-sample-size/overfit risk |
| GNN immediately | KILL/DEFER | No demonstrated graph/residual need |
| BiLSTM/BRITS | KILL | Bidirectional future information |
| Kalman smoother features | KILL | Uses future observations |
| Plain DINEOF over >99% missing test fields | KILL | Ill-conditioned / extreme missingness / leakage risk |
| GLDAS-2.1 | KILL for now | Operational latency > allowed window |
| GLDAS-2.2 GRACE-DA | FORBIDDEN | Assimilates target-family data |
| GDO GRACE TWS anomaly | FORBIDDEN | Target-family leakage |
| Final ERA5 archive as historical ERA5T | HOLD | Vintage/provenance unresolved |
| Same-month sparse cross-location TWS assimilation | HOLD | Potentially legal and useful, but organizer confirmation preferred |
| 10-model stacked ensemble | KILL for now | Unnecessary instability/compute |
| Public-LB weight optimization | KILL | 30% public overfit |

---

# 24. Research branches completed during saturation phase

The research process intentionally used repeated parallel workers with non-overlapping tasks. Worker launch failures caused by desktop/browser bootstrap issues were retried where necessary; substantive scopes were still covered.

Completed substantive branches:

1. competition validation and leakage analysis;
2. TWS/GRACE hydrology and temporal memory;
3. forecasting literature and simple baselines;
4. external-data operational-latency audit;
5. masked-state/missing-observation forecasting methods;
6. spatial structure, mascons, EOF/PCA, CNN/GNN triage;
7. global pooled LightGBM/XGBoost/CatBoost formulation;
8. experiment ladder, proxy CV, full CV, ensemble discipline;
9. deep-model ranking and compute triage;
10. sparse current-month TWS assimilation;
11. leakage/rules adversarial red-team;
12. state-space / dynamic-factor complement;
13. final adversarial falsification of the full strategy.

The final falsification pass found **no fatal flaw** in the LGBM+EOF strategy provided calendar alignment and causality are exact.

It did identify two potentially fatal implementation traps:

1. creating labels/horizons with next-row shift instead of next-calendar-month joins;
2. constructing long-horizon pseudo-examples using only stale-anchor exogenous variables instead of preserving fresh exogenous information through the current source month.

It also added the explicit trend/harmonic branch and score-aligned unweighted EOF requirement.

---

# 25. Key literature/evidence ledger

This section records major evidence streams used to shape the plan. It is not a complete bibliographic review, but it preserves the research findings that materially affected decisions.

## Competition / operational rules

- Zindi challenge Data page: dataset sizes, features, spatial/temporal encouragement.  
  `https://zindi.world/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge/data`
- Zindi challenge rules/discussion: target is next calendar month, 66.5% TWS masking, 1–7 effective horizon, causality and external-data rule.  
  `https://zindi.world/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge/discussions/34093`
- Zindi evaluation: RMSE 50%, trustworthiness 30%, innovation/practicality 20%, submission rules.  
  `https://zindi.world/competitions/one-step-ahead-of-drought-forecasting-global-water-storage-challenge/discussions/34542`

## Forecast validation

- Hyndman/FPP3 rolling-origin time-series cross-validation.  
  `https://otexts.com/fpp3/tscv.html`
- Cerqueira et al. 2020: repeated out-of-sample evaluation under nonstationarity.
- Tashman 2000: rolling-origin forecast evaluation review.
- Roberts et al. 2017: structured cross-validation under spatial/temporal dependence.
- Bergmeir et al. 2018: restrictions under which ordinary K-fold may be valid for autoregressive settings.

## Direct TWS / GRACE forecasting

- Li et al., WRR 2020: PCA/decomposition + regression robustness; more complex ANN/ARX can overfit relative to simpler robust models.
- Li et al., GRL 2024, DOI `10.1029/2024GL109101`: global/regional GRACE TWS forecasting using PCA and lagged hydrometeorological/climate variables, with skill decline as lead increases.
- Ahmed et al., Remote Sensing 2019: NARX + hydrometeorological variables for African basin TWS forecasting.
- 2024 systematic review of GRACE+ML literature: majority of papers are reconstruction/downscaling rather than true forward forecasting; reconstruction accuracy must not be treated as forecasting evidence.

## Groundwater / sequence-model evidence

- Wunsch et al., HESS 2021: shallow/simple approaches can outperform LSTM/CNN under limited historical groundwater records; short record length is a real deep-learning constraint.
- Haider et al., Water 2023: TCN showed useful multi-month groundwater performance relative to LSTM/ANN in the compared setting.
- Peng et al., WRR 2025: modern temporal model can outperform simpler sequence models on longer well records, but at significantly greater training cost.
- Bai/Kolter/Koltun TCN work: causal temporal convolutions as strong sequence baseline.
- Che et al. 2018 GRU-D: explicit missingness/time-since modeling, though assumptions need adaptation for seasonal TWS.

## Spatial / GRACE support

- NASA/JPL/CSR GRACE mascon documentation: effective spatial support is coarser than fine output grid; spatial cells are strongly dependent.
- GRACE PCA/EOF literature: few dominant modes often capture large shares of global/regional TWS variance.

## Data assimilation / EOF

- reduced-rank SEEK/Kalman literature: forecast-error covariance can be represented in a low-rank EOF subspace;
- GRACE data-assimilation literature supports model-prior + TWS observation updates conceptually;
- DINEOF literature reviewed as a negative comparison because this challenge’s current-month missingness is extraordinarily high.

## External data

- ECMWF ERA5/ERA5T documentation: near-real-time ERA5T and later consolidated ERA5;
- NASA GES DISC GLDAS documentation: GLDAS-2.1 latency, Early Product latency, and GRACE-assimilating GLDAS-2.2 caveat;
- JRC GDO documentation: anomaly/climatology definitions and reprocessing concerns.

---

# 26. Free-compute decision — current 2026 recommendation

## 26.1 Primary recommendation: Kaggle Notebooks

For this challenge, **Kaggle is the best default free compute environment** for the repository.

Reasons:

1. Kaggle currently documents free NVIDIA Tesla P100 GPU access with a weekly GPU quota around **30 hours or sometimes more**, depending on demand/resources.
2. The challenge’s initial core path—feature engineering, LightGBM, ridge, EOF/PCA, state-space—depends heavily on CPU/RAM and does not need an accelerator for every run.
3. Kaggle notebooks provide persistent input datasets and saved output versions, which is useful for our run-artifact workflow.
4. Kaggle can import a GitHub repository directly as a Dataset and can automatically sync/update that dataset from GitHub.
5. Popular Python packages are preinstalled, and Kaggle’s Dependency Manager can pin additional packages for reproducibility.
6. Kaggle requires no paid subscription or credit-card trial for ordinary free notebook compute, which aligns cleanly with Zindi reproducibility rules.
7. A 12-hour notebook session class is suitable for our planned experiments, especially because the entire research strategy intentionally kills any run that needs extremely long uninterrupted training early.

Official Kaggle references:

- GPU usage: `https://www.kaggle.com/docs/efficient-gpu-usage`
- datasets/GitHub connector: `https://www.kaggle.com/docs/datasets`
- packages/dependency manager: `https://www.kaggle.com/docs/packages`
- API/CLI authentication: `https://www.kaggle.com/docs/api`

## 26.2 Why Colab Free is secondary

Google currently states:

- free Colab does provide GPUs/TPUs;
- GPU access is heavily restricted and not guaranteed;
- overall usage limits and GPU types are dynamic and unpublished;
- free notebooks can run for at most roughly **12 hours**, depending on availability/usage;
- managed runtimes may terminate early and prioritize interactive users.

That unpredictability makes Colab less attractive for a deadline-driven competition where we want reproducible repeated CV runs.

Colab remains a useful fallback for:

- quick notebook inspection;
- short experiments;
- emergency extra compute when Kaggle quota is unavailable.

Official source:  
`https://research.google.com/colaboratory/faq.html`

## 26.3 Lightning AI

Lightning currently advertises a free tier and initial GPU credits, including up to ~80 GPU hours depending on machine/credit assumptions.

However:

- some additional free credits involve adding a card;
- Zindi explicitly disallows solutions depending on paid services/free trials that require a credit card;
- reproducibility simplicity is better with Kaggle.

Therefore Lightning is **not** our canonical competition environment.

## 26.4 GitHub Codespaces

Good for:

- repo editing;
- tests;
- lightweight CPU development.

Not ideal as the main ML training host because:

- free usage is monthly core-hour quota;
- no normal free GPU path;
- the included quota is explicitly designed for side projects rather than continuous compute.

## 26.5 Final compute hierarchy

```text
Primary training/CV: Kaggle
Local machine: orchestration, code review, small tests only if desired
Fallback burst notebook: Colab Free
Repo editing/CI fallback: GitHub/Codespaces
Lightning: noncanonical/avoid depending on trial credits
```

---

# 27. Kaggle repository operating model

The intended architecture is:

```text
GitHub repo = source of truth for code/config/docs
Kaggle Dataset = immutable-ish competition input/data mount
Kaggle Notebook = thin execution wrapper
Kaggle output = run artifacts/checkpoints/OOF/submissions
MASTER_RESEARCH_AND_ML_PLAN.md = living project strategy
```

Avoid developing substantial modeling logic only inside the Kaggle UI.

The notebook should import/run repo modules so final code review is reproducible.

Recommended future repo layout:

```text
drought/
├── MASTER_RESEARCH_AND_ML_PLAN.md
├── README.md
├── requirements.txt
├── configs/
│   ├── baseline.yaml
│   ├── lgbm_core.yaml
│   └── ...
├── src/
│   ├── data.py
│   ├── calendar.py
│   ├── availability.py
│   ├── validation.py
│   ├── features.py
│   ├── leakage_tests.py
│   ├── metrics.py
│   ├── baselines.py
│   ├── models/
│   │   ├── lgbm.py
│   │   ├── eof.py
│   │   ├── state_space.py
│   │   └── ...
│   ├── train.py
│   └── infer.py
├── notebooks/
│   └── kaggle_runner.ipynb
├── scripts/
│   ├── run_proxy.py
│   ├── run_full_cv.py
│   └── make_submission.py
├── tests/
│   ├── test_calendar_alignment.py
│   ├── test_causality.py
│   ├── test_mask_streaming.py
│   └── ...
└── artifacts/      # normally gitignored
```

---

# 28. Immediate implementation order from this document

When development begins, do **not** start with model tuning.

Exact first engineering sequence:

1. initialize repository/version control;
2. commit this master plan;
3. add `.gitignore` preventing challenge data/model artifacts from accidental push if needed;
4. implement exact calendar index utilities;
5. implement target identity test;
6. implement TWS availability ledger;
7. implement test horizon derivation and verify exact counts above;
8. implement streaming rolling-mask validator;
9. implement leakage tests;
10. implement exact weighted metric;
11. implement persistence/seasonal/trend baselines;
12. confirm baseline CV behaves rationally by h;
13. only then train the first ΔTWS LightGBM.

---

# 29. Living decision log

## 2026-09-07 — Research saturation checkpoint

**Decision:** Do not run the starter notebook as the primary experiment yet.  
**Reason:** Its ordinary temporal holdout does not replicate test TWS staleness and therefore overrepresents h=1 conditions.

**Decision:** Build the causal validator before serious model training.  
**Reason:** Validation fidelity is the most important competition-specific problem.

**Decision:** Main first model = pooled horizon-aware ΔTWS LightGBM.  
**Reason:** best compute/value match to short temporal panel, nonlinear hydrology, and mixed horizon.

**Decision:** Add explicit harmonic/trend baseline before broad tree tuning.  
**Reason:** annual/semiannual TWS structure is strong and trees do not extrapolate secular time cleanly.

**Decision:** EOF/PCA is the first orthogonal spatial branch.  
**Reason:** GRACE field is strongly spatially dependent/low-rank and literature supports dominant-mode forecasting.

**Decision:** External data delayed.  
**Reason:** internal structure is underused and external historical operational provenance creates rule risk.

**Decision:** Deep learning delayed behind residual diagnostics.  
**Reason:** only ~138 global monthly fields; row count exaggerates temporal sample size.

**Decision:** Kaggle is canonical free compute.  
**Reason:** current free quota, reproducible notebooks/datasets, stable competition workflow, and no dependence on a paid/free-trial service.

## 2026-09-07 — Calendar-safe dataset audit implemented

**Implementation:** Added `scripts/audit_dataset.py` and ran it successfully against the local competition files before pushing it for Kaggle reproduction.

**Verified structural results:**

- Train rows: `2,154,021`
- Test rows: `280,961`
- Train unique locations: `15,715`
- Test unique locations: `15,715`
- Train source months: `138`
- Test source months: `18`
- Train range: `2002-05` to `2015-08`
- Test range: `2015-09` to `2018-12`
- exact next-calendar-month target/TWS matched pairs: `1,977,398`
- target identity maximum absolute error: `0.0`
- target identity RMSE: `0.0`
- exact identity share: `1.0`
- masked Test TWS rows: `186,913`
- Test masked share: `0.6652631504`
- Test TWS NaN rows: `186,913`
- mask-vs-NaN mismatches: `0`

**Important confirmation:** The audit deliberately uses an explicit next-calendar-month join rather than row-order shifting. It confirms that `target(cell,t) == TWS(cell,t+1)` exactly whenever the next calendar TWS row exists, and therefore reinforces the rule that the target table must remain inaccessible to feature-generation code.

**Observed mostly-masked source months retain only a handful of legal current TWS values:** examples include `4` visible cells in 2017-05, `17` in 2016-03, `21` in 2017-04, `31` in 2018-12, and at most `65` among the audited mostly-masked months. This independently reconfirms the sparse-assimilation premise while also demonstrating why plain snapshot gap filling is severely underdetermined.

---

# 30. Next update trigger

Update this document after the following first milestone:

> **calendar-safe data layer + exact test-mask simulator + persistence/seasonal/trend baseline CV are implemented and validated.**

At that point append:

- exact fold calendar dates;
- verified baseline CV table;
- benchmark runtime/memory;
- leakage-test results;
- any discrepancy between expected and observed horizon counts;
- first accepted/rejected feature hypotheses.

---

# 31. 2026-09-07 — First causal baseline scoring milestone

The first actual forecast scores were produced only after the calendar audit,
test availability ledger, exact historical mask simulator, and recent direct-horizon
fold builder all passed locally and on Kaggle.

The newest 18-observed-month block (`2013-11` through `2015-08`) remains an untouched
lockbox. Baseline development used only dev1–dev3.

## 31.1 Baseline scoring protocol

For every direct-horizon validation example:

- target labels stayed in a separate table;
- `h=1..7` anchors were calendar-aligned as `target_month - h`;
- no `h>1` example used source-month TWS;
- fold-fitted harmonic/trend artifacts used only raw TWS observations strictly
  before the validation source block;
- headline score was `sqrt(sum_h q_h * MSE_h)` with exact test row weights;
- the lockbox was not requested or scored.

## 31.2 Weighted RMSE results

| Model | dev1 | dev2 | dev3 | Mean dev RMSE |
|---|---:|---:|---:|---:|
| **Persistence** | **0.628228** | **0.696437** | **0.684559** | **0.669741** |
| Seasonal-anomaly persistence (`rho=1`) | 0.676381 | 0.718577 | 0.710663 | 0.701874 |
| Harmonic + trend, persistence fallback | 0.897400 | 0.990534 | 0.950495 | 0.946143 |
| Lag-12 seasonal naive, persistence fallback | 0.849777 | 1.061902 | 0.946639 | 0.952773 |

Persistence won **all three** development folds.

## 31.3 Persistence horizon behavior

### dev1

```text
h1 0.5162
h2 0.6164
h3 0.6678
h4 0.6916
h5 0.7134
h6 0.7572
h7 0.7855
```

### dev2

```text
h1 0.5385
h2 0.6584
h3 0.7367
h4 0.7946
h5 0.8403
h6 0.8910
h7 0.9380
```

### dev3

```text
h1 0.5563
h2 0.6631
h3 0.7403
h4 0.7539
h5 0.7890
h6 0.8187
h7 0.8697
```

The intended difficulty pattern is clearly visible: error generally rises with TWS
age. This is another validation sanity check and reinforces the need for an explicit
horizon-aware model rather than optimizing only one-step forecasting.

## 31.4 Decisions from the baseline experiment

### KEEP — persistence as the incumbent baseline

Persistence is now the score every serious model must beat, both overall and by
horizon. It is especially strong at h=1 and remains competitive even at h=7.

### KILL as standalone model — lag-12 seasonal naive

Lag-12 was substantially worse in every fold. Do not spend tuning budget on pure
same-month-last-year forecasting.

Its existence as a potential **input feature** is not ruled out; a nonlinear model
may learn to use it only in the locations/seasons where it is genuinely useful.

### KILL as standalone model — raw harmonic + trend forecast

Per-location linear trend + annual + semiannual harmonics were far worse than
persistence. Do not treat the deterministic structural fit as a main predictor.

However, the research reason for retaining trend/seasonal structure still stands:
trees do not extrapolate time trends cleanly. Harmonic/trend outputs or anomalies may
still be tested later as **features/residual references**, but only through controlled
ablation.

### KILL as standalone model — unshrunk seasonal-anomaly persistence

Persisting the latest anomaly around the structural harmonic/trend baseline (`rho=1`)
was consistently worse than raw TWS persistence. Do not add horizon-specific `rho_h`
tuning yet; the base formulation has not earned extra degrees of freedom.

## 31.5 Important interpretation

These results strongly support the original direct-residual plan:

> The problem is dominated by a strong stale TWS state anchor, and useful models
> should learn the **change away from persistence** using horizon, recent legal TWS
> history, and fresh hydrometeorological forcing rather than replacing the anchor
> with a standalone seasonal model.

The next promoted experiment is therefore the first pooled horizon-aware residual
model, beginning with a deliberately small feature set before causal-history and
hydrologic-gap features are added.

---

# 31. Implementation progress — recent direct-horizon validation folds

## 2026-09-07 — Recent h=1…7 fold builder validated

The exact test-calendar replay is faithful but cannot be shifted beyond 2012-04 because later GRACE source-month presence differs from the test calendar. To preserve recent nonstationarity coverage, a second validation mode was implemented: **direct-horizon folds**.

For every validation source row at month `t` and every `h in {1,…,7}` where a legal historical TWS anchor exists, the builder constructs:

```text
target month      = t + 1 calendar month
legal TWS anchor  = target month - h calendar months
current exogenous = source month t (joined later from an explicit allow-list)
raw source TWS    = unavailable for h > 1
```

The label table remains physically separate from the feature/availability ledger.

### Fixed observed-source-month blocks

The most recent 72 observed train source months were divided into four non-overlapping 18-observed-month blocks:

| Fold | Source start | Source end | First target month | Max fit-label month | Role |
|---|---|---|---|---|---|
| dev1 | 2008-04 | 2009-09 | 2008-05 | 2008-04 | development |
| dev2 | 2009-10 | 2011-05 | 2009-11 | 2009-10 | development |
| dev3 | 2011-08 | 2013-07 | 2011-09 | 2011-08 | development |
| lockbox | 2013-11 | 2015-08 | 2013-12 | 2013-11 | final untouched recent check |

These blocks are defined over **observed source months**, not naive continuous calendar slices, because the GRACE record has genuine missing calendar months.

### Structural audit results

All four fold ledgers passed:

- h range exactly 1…7;
- zero `target` columns in the ledger;
- zero h>1 examples using source-month TWS;
- unique synthetic `example_id` values;
- strict chronological separation between folds;
- all 15,715 locations represented in each fold.

Validation-example counts and minimum legal-anchor coverage across h:

| Fold | Source rows | Synthetic examples | Minimum h-anchor coverage |
|---|---:|---:|---:|
| dev1 | 280,719 | 1,958,165 | 99.54% |
| dev2 | 280,836 | 1,863,166 | 88.32% |
| dev3 | 280,936 | 1,461,621 | 55.29% |
| lockbox | 281,062 | 1,524,917 | 60.86% |

Coverage falls in later periods because some required historical anchor calendar months do not exist in the GRACE source-row record. These rows are **dropped**, not imputed with future information or silently assigned a longer stale horizon.

This means the recent direct-horizon validator should score each h separately and combine horizon MSEs using the exact test weights. It should not compare raw pooled row counts across h as though every anchor were equally observable historically.

### Current validation architecture

The project now has two complementary causal validation modes:

1. **Exact historical test-template replay** — closest match to real test row/mask geometry, but limited to older history through 2012-04.
2. **Recent direct-horizon folds** — covers the newest train years and explicitly constructs legal h=1…7 anchors, but cannot reproduce every row-level historical availability detail of the real test mask.

Serious model promotion should use both modes rather than trusting either one alone.

### Next implementation task

Build the score layer and baseline runner:

1. exact test-h weighted RMSE;
2. persistence from `last_observed_TWS`;
3. lag-12/seasonal naive where legal;
4. harmonic + linear trend baseline;
5. report RMSE by fold and h before any GBDT training.

---

# 31. 2026-09-07 — Historical mask simulator implementation checkpoint

The first exact historical replay of the real test mask geometry has now been implemented and audited.

### Implemented invariants

- the real test row-level effective horizon `h` is derived first from the legal test visibility ledger;
- historical replay shifts the **actual row-level legal horizon**, not merely the Boolean mask flag;
- for a historical source month `t` and transplanted horizon `h`, the only legal TWS anchor is joined from exact calendar month `t + 1 - h`;
- hidden validation `TWS_t` is never used as the anchor when `h>1`;
- `target` is returned in a separate label table and is absent from the availability/feature ledger;
- mutating every simulated hidden validation `TWS_t` to extreme random values leaves `last_observed_date`, `last_observed_TWS`, and `h` unchanged;
- the historical replay preserves exact horizon range `1..7` and a horizon-share mix within 0.002 absolute share of the real test distribution.

### Latest exact calendar replay available in Train

The test contains 18 source months at relative offsets:

```text
[0, 4, 5, 6, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 34, 38, 39]
```

There are 62 historical start months for which all 18 shifted source-month slots exist in Train.

The latest exact shifted start is:

```text
2009-01
```

and its final source month is:

```text
2012-04
```

This limitation is caused by the real missing-month structure of later GRACE training data.

### Consequence for model selection

> **The exact test-calendar replay is a high-fidelity causality/geometry fold, but it must not become the only CV selector because it ends in 2012 and would underweight late-period nonstationarity.**

The later validation layer must therefore contain two complementary modes:

1. **Exact historical template folds** where the complete test calendar pattern can be shifted without invention. These are the highest-fidelity tests of mask/availability logic.
2. **Recent direct-horizon folds** that keep source-month exogenous variables at the recent validation date while selecting exact legal anchor month `t+1-h` for sampled/test-weighted `h`. These folds will evaluate 2012–2015 nonstationarity without fabricating missing TWS observations.

The two modes must share the same target-blind feature builder and exact calendar anchor joins.

---

# 31. Development log

## 2026-09-07 — Calendar-safe dataset audit implemented

Implemented `scripts/audit_dataset.py` and reproduced the audit locally and on Kaggle.

Verified:

- train rows = 2,154,021;
- test rows = 280,961;
- unique locations = 15,715 in both train and test;
- train source months = 138;
- test source months = 18;
- `target(t) == TWS(t+1 calendar month)` exactly for 1,977,398 matched pairs;
- max target-identity absolute error = 0;
- test masked rows = 186,913;
- test masked share = 0.6652631504;
- `TWS_t_masked` matches `TWS_t` missingness exactly.

Status: **PASS**.

## 2026-09-07 — Causal TWS availability ledger implemented

Implemented `src/availability.py` and `scripts/audit_horizons.py`.

The ledger records, for every test row:

- whether same-month TWS is legally visible;
- last legally observed TWS timestamp;
- last legally observed TWS value;
- next-calendar-month target timestamp;
- effective horizon `h`.

Exact verified test horizon distribution:

| h | rows | share |
|---:|---:|---:|
| 1 | 94,048 | 0.334737 |
| 2 | 62,576 | 0.222721 |
| 3 | 46,777 | 0.166489 |
| 4 | 31,076 | 0.110606 |
| 5 | 15,560 | 0.055381 |
| 6 | 15,479 | 0.055093 |
| 7 | 15,445 | 0.054972 |

Weighted mean horizon = **2.7143411363**.

Additional causality assertion:

```text
masked rows using same-month hidden TWS = 0
```

Status: **PASS locally and reproduced on Kaggle**.

---

# 32. 2026-09-07 — Baseline reproduction and EXP001 launch checkpoint

Kaggle reproduced the causal baseline suite exactly on dev1–dev3. The protected 2013-11→2015-08 lockbox remained untouched.

Verified persistence weighted RMSE:

| Fold | Persistence weighted RMSE |
|---|---:|
| dev1 | 0.628228 |
| dev2 | 0.696437 |
| dev3 | 0.684559 |
| mean | 0.669741 |

Cheap structural alternatives were uniformly worse across all three development folds:

- seasonal-anomaly persistence mean ≈ 0.701874;
- harmonic+trend mean ≈ 0.946143;
- lag-12-or-persistence mean ≈ 0.952773.

Decision:

> **Persistence is the incumbent baseline. Do not spend additional compute polishing standalone lag-12 or unshrunk harmonic/seasonal baselines unless later residual diagnostics show a specific slice where they add ensemble value.**

The first ML experiment, **EXP001**, is now defined as a pooled direct-residual LightGBM predicting:

```text
delta = target_TWS - last_legal_TWS
```

with the deliberately minimal feature set:

```text
last_observed_TWS
h
lat
lon
month_sin
month_cos
```

Training formulation:

- one deterministic sampled horizon per eligible historical source row;
- horizon sampling follows the exact test row-level h distribution;
- rows with unavailable historical anchors are dropped;
- per-row training weights rebalance surviving rows back to the exact test h mixture;
- source feature construction is target-blind;
- labels are joined only at the supervised boundary;
- validation uses the audited direct h=1..7 dev folds;
- validation weights make LightGBM L2 equal the exact test-horizon-weighted MSE;
- the lockbox is explicitly rejected by the EXP001 CLI.

Local target-blind dev1 dry run passed with:

```text
train rows: 975,657
validation rows: 1,958,165
dropped sampled rows missing historical anchor: 54,811
persistence reference: 0.628228
```

EXP001 intentionally does **not** use current SPEI or soil moisture yet. Its purpose is to isolate the value of the pooled ΔTWS formulation, h-conditioning, geography and seasonality before adding fresh hydrometeorological information in the next controlled ablation.

## 2026-09-07 — EXP001 promoted across all three development folds

Kaggle reproduced a strong and highly consistent improvement from the frozen EXP001 recipe on dev1–dev3. No parameter or feature changes were made between folds, and the protected lockbox remained untouched.

| Fold | Persistence | EXP001 | Absolute gain | Relative gain |
|---|---:|---:|---:|---:|
| dev1 | 0.628228 | **0.569746** | +0.058482 | +9.31% |
| dev2 | 0.696437 | **0.637836** | +0.058601 | +8.41% |
| dev3 | 0.684559 | **0.621771** | +0.062787 | +9.17% |

Mean EXP001 weighted RMSE across the three dev folds is approximately **0.609784** versus persistence mean **0.669741**, an average absolute gain of roughly **0.0600 RMSE**.

The horizon pattern is especially important. EXP001 beats persistence at **every h on every fold**, and the absolute improvement generally increases as the TWS state becomes staler. Examples:

- dev1: h1 +0.0311, h4 +0.0735, h7 +0.1049;
- dev2: h1 +0.0222, h4 +0.0787, h7 +0.1222;
- dev3: h1 +0.0292, h4 +0.0729, h7 +0.1142.

This strongly validates the central formulation:

> **pooled global direct residual / ΔTWS forecasting, explicitly conditioned on legal TWS age h, is materially better than persistence and becomes more valuable as the stale-state horizon grows.**

Feature importance is also stable across folds: `last_observed_TWS` dominates, followed by latitude/longitude, then `h`, then seasonal sin/cos. This supports keeping geography and explicit horizon age in the core model.

Decision: **PROMOTE EXP001 as the core incumbent. Do not tune tree hyperparameters yet. First test whether fresh supplied hydrometeorological variables add orthogonal signal.**

## EXP002 — Fresh supplied hydrometeorology ablation

EXP002 changes exactly one thing relative to EXP001: it adds the legally available **current source-month** supplied covariates:

```text
SPEI_01_t
SPEI_03_t
SPEI_06_t
SPEI_12_t
SOIL_MOISTURE_t
```

The complete EXP002 feature set is therefore:

```text
last_observed_TWS
h
lat
lon
month_sin
month_cos
SPEI_01_t
SPEI_03_t
SPEI_06_t
SPEI_12_t
SOIL_MOISTURE_t
```

All other choices remain frozen: same ΔTWS target, same deterministic sampled-h training construction, same horizon reweighting, same LightGBM parameters, same dev folds, and same lockbox protection.

Critical causal rule:

> TWS-derived state is restricted to the exact legal historical anchor `target_month - h`, while SPEI/soil/calendar features remain fresh at the current source month. The feature builder never receives `target`.

Local EXP002 dev1 dry run passed with the same 975,657 training rows and 1,958,165 validation rows as EXP001, confirming this is a clean feature-only ablation.

---

# 33. 2026-09-07 — EXP001 dev1 result: pooled ΔTWS LightGBM promoted

Kaggle completed the first real ML run of EXP001 on **dev1 only**, with the lockbox still untouched.

Configuration remained exactly as specified above:

- target = `target_TWS - last_legal_TWS`;
- pooled h=1..7 model;
- features = `last_observed_TWS`, `h`, `lat`, `lon`, `month_sin`, `month_cos`;
- 975,657 sampled training rows;
- 1,958,165 direct-horizon validation examples;
- no SPEI, soil moisture, TWS lag stack, rolling features, EOF features, external data, or hyperparameter search.

Headline result:

| Model | dev1 weighted RMSE | Absolute gain | Relative gain |
|---|---:|---:|---:|
| Persistence | 0.628228 | — | — |
| EXP001 core ΔTWS LightGBM | **0.569746** | **0.058482** | **9.31%** |

Best iteration = **51**. Runtime ≈ **100 s** on Kaggle CPU.

Per-horizon results:

| h | Persistence RMSE | EXP001 RMSE | Absolute gain |
|---:|---:|---:|---:|
| 1 | 0.516190 | **0.485096** | 0.031095 |
| 2 | 0.616391 | **0.564925** | 0.051466 |
| 3 | 0.667828 | **0.604004** | 0.063824 |
| 4 | 0.691629 | **0.618119** | 0.073509 |
| 5 | 0.713380 | **0.631997** | 0.081383 |
| 6 | 0.757236 | **0.662408** | 0.094828 |
| 7 | 0.785540 | **0.680662** | 0.104878 |

The improvement is present at **every horizon** and grows monotonically with horizon. This is especially important because it argues against an h=1-only artifact and strongly supports the intended direct-horizon residual formulation for stale-TWS rows.

Gain-based feature importance:

1. `last_observed_TWS` — 555,666;
2. `lat` — 127,722;
3. `lon` — 117,784;
4. `h` — 53,987;
5. `month_sin` — 27,043;
6. `month_cos` — 25,288.

Interpretation:

- stale state remains dominant, as expected;
- geography already carries substantial correction signal even without hydrologic covariates;
- explicit horizon conditioning is useful;
- simple seasonal phase contributes but is secondary;
- the strong longer-horizon gains make the model structurally complementary to persistence rather than merely a minor one-step correction.

Decision:

> **PROMOTE EXP001 unchanged to dev2 and dev3 before adding any feature or tuning any hyperparameter.**

The next experiment is not a new model. It is the frozen-recipe robustness check on dev2/dev3. Only if EXP001 remains superior across those folds do we proceed to EXP002, which will add fresh current-month SPEI and soil moisture as the first controlled hydrometeorological ablation.

---

# 34. 2026-09-07 — EXP002 dev1 result: fresh hydrometeorology strongly promoted

Kaggle completed the first controlled EXP002 run on **dev1 only**, with the lockbox still untouched.

EXP002 changed exactly one thing relative to EXP001: it added fresh source-month `SPEI_01_t`, `SPEI_03_t`, `SPEI_06_t`, `SPEI_12_t`, and `SOIL_MOISTURE_t`. The ΔTWS target, sampled-h training construction, exact horizon reweighting, LightGBM parameters, and validation fold were unchanged.

Headline result:

| Model | dev1 weighted RMSE | Gain vs persistence | Incremental gain vs EXP001 |
|---|---:|---:|---:|
| Persistence | 0.628228 | — | — |
| EXP001 core ΔTWS LightGBM | 0.569746 | +0.058482 | — |
| EXP002 + fresh hydrology | **0.534132** | **+0.094096** | **+0.035614** |

Relative to EXP001, EXP002 improves dev1 by approximately **6.25%**. Relative to persistence, total gain is **14.98%**.

Per-horizon EXP002 RMSE:

| h | Persistence | EXP001 | EXP002 | EXP002 gain vs persistence |
|---:|---:|---:|---:|---:|
| 1 | 0.516190 | 0.485096 | **0.481311** | 0.034879 |
| 2 | 0.616391 | 0.564925 | **0.540535** | 0.075856 |
| 3 | 0.667828 | 0.604004 | **0.560606** | 0.107222 |
| 4 | 0.691629 | 0.618119 | **0.563263** | 0.128366 |
| 5 | 0.713380 | 0.631997 | **0.565692** | 0.147688 |
| 6 | 0.757236 | 0.662408 | **0.580271** | 0.176965 |
| 7 | 0.785540 | 0.680662 | **0.587726** | 0.197815 |

The incremental hydrologic benefit is very small at h=1 but becomes large as TWS grows stale. This is exactly the expected behavior if fresh SPEI/soil information is helping bridge the unobserved storage evolution rather than merely duplicating current TWS.

Gain-based feature importance in EXP002 dev1:

1. `last_observed_TWS` — 584,169;
2. `lat` — 187,259;
3. `lon` — 163,329;
4. `SPEI_06_t` — 149,225;
5. `h` — 101,578;
6. `SOIL_MOISTURE_t` — 42,706;
7. `SPEI_01_t` — 39,361;
8. `month_sin` — 31,881;
9. `month_cos` — 29,204;
10. `SPEI_03_t` — 19,731;
11. `SPEI_12_t` — 9,900.

Interpretation:

- the supplied hydrometeorological variables contain substantial orthogonal signal beyond stale TWS/geography/season;
- `SPEI_06_t` is the strongest new variable by a wide margin in dev1;
- the strongest total gains occur at h=5–7, which supports the stale-state bridge interpretation;
- current source-month exogenous covariates remain causally legal because only TWS-derived state is restricted to the stale anchor;
- no hyperparameter tuning has been performed yet.

Decision:

> **PROMOTE EXP002 unchanged to dev2 and dev3. Do not add engineered hydrologic deltas, lag stacks, EOFs, or tune LightGBM until this raw fresh-hydrology gain is confirmed across both remaining development folds.**

## 2026-09-07 — EXP002 promoted across all three development folds

The frozen EXP002 recipe was then run unchanged on dev2 and dev3. The protected lockbox remained untouched.

| Fold | Persistence | EXP001 | EXP002 | EXP002 gain vs EXP001 |
|---|---:|---:|---:|---:|
| dev1 | 0.628228 | 0.569746 | **0.534132** | +0.035614 |
| dev2 | 0.696437 | 0.637836 | **0.580907** | +0.056929 |
| dev3 | 0.684559 | 0.621771 | **0.582126** | +0.039646 |

Three-fold means:

- persistence: **0.669741**;
- EXP001: **0.609784**;
- EXP002: **0.565722**.

EXP002 improves the already-strong EXP001 mean by **0.044063 RMSE (~7.23%)** and improves persistence by **0.104020 RMSE (~15.53%)**.

The improvement is again strongest at longer horizons. On dev2, for example, EXP002 reaches h7 RMSE **0.665266** versus persistence **0.937957**. On dev3, h7 is **0.644564** versus **0.869687** persistence.

`SPEI_06_t` is the dominant new hydrometeorological feature on every fold and even becomes the second-highest gain feature on dev3, behind only `last_observed_TWS`. Soil moisture and SPEI-01 also contribute materially; SPEI-12 is consistently weaker but nonzero.

Decision:

> **PROMOTE EXP002 as the new development incumbent. Do not tune LightGBM yet. The next controlled ablation is causal TWS history behind the legal anchor.**

## EXP003 — Causal TWS-history ablation

EXP003 keeps the complete EXP002 feature set and adds only exact-calendar TWS history strictly behind the legal last-observed TWS anchor:

```text
TWS_anchor_lag1
TWS_anchor_lag2
TWS_anchor_lag3
TWS_anchor_lag6
TWS_anchor_lag12
TWS_anchor_delta1
TWS_anchor_delta3
TWS_anchor_delta6
TWS_anchor_delta12
```

For every example, these lags are joined relative to `last_observed_date`, not the current source month. Therefore an h>1 example can never use TWS from its simulated hidden interval. Missing GRACE calendar months remain missing and are passed to LightGBM as NaN; no interpolation, backfill, row-shift, or future state is used.

All other choices remain frozen: same sampled-h training rows, same fresh source-month hydrology, same ΔTWS target, same LightGBM parameters, and same dev folds.

Local dev1 dry-run checks passed with the same 975,657 training rows and 1,958,165 validation examples as EXP002. The EXP003 feature matrix is target-blind and ready for Kaggle training.

## 2026-09-07 — EXP003 promoted across all three development folds

The frozen EXP003 recipe was run unchanged on dev2 and dev3 after its dev1 promotion. The protected lockbox remained untouched.

| Fold | EXP002 | EXP003 | Incremental gain |
|---|---:|---:|---:|
| dev1 | 0.534132 | **0.509946** | +0.024186 |
| dev2 | 0.580907 | **0.571174** | +0.009734 |
| dev3 | 0.582126 | **0.557687** | +0.024439 |

Full three-fold mean EXP003 weighted RMSE is approximately **0.546269**, versus **0.565722** for EXP002 and **0.669741** for persistence. EXP003 therefore improves EXP002 by approximately **0.019453 RMSE (~3.4%)** and persistence by approximately **0.123472 RMSE (~18.4%)**.

The improvement remains broad across horizons and folds rather than being an h=1 artifact. On dev2, EXP003 reaches h7 RMSE **0.669805** versus **0.937957** persistence. On dev3, h7 is **0.627126** versus **0.869687** persistence.

Across all three folds, the strongest added TWS-history signals are the pre-anchor change features (`TWS_anchor_delta1`, `delta3`, `delta12`, then `delta6`). The raw anchor lags are consistently weaker. This supports the interpretation that recent pre-anchor trajectory is more useful than simply adding more historical levels.

Decision:

> **PROMOTE EXP003 as the development incumbent. Before opening the protected recent lockbox, run the frozen EXP003 recipe once on the latest exact historical replay of the real test mask geometry. If it remains strong there, open the lockbox once.**

## 2026-09-07 — EXP003 dev1 result: causal TWS history strongly promoted

Kaggle completed EXP003 on dev1 with the exact same training rows, validation rows, sampled-h construction, ΔTWS target, fresh hydrology, LightGBM parameters, and protected lockbox policy as EXP002. The only change was the addition of exact-calendar TWS history behind the legal anchor.

Headline result:

| Model | dev1 weighted RMSE | Incremental gain |
|---|---:|---:|
| Persistence | 0.628228 | — |
| EXP001 core ΔTWS | 0.569746 | +0.058482 vs persistence |
| EXP002 + fresh hydrology | 0.534132 | +0.035614 vs EXP001 |
| EXP003 + causal TWS history | **0.509946** | **+0.024186 vs EXP002** |

EXP003 improves EXP002 by approximately **4.53%** on dev1 and improves persistence by **0.118282 RMSE (18.83%)**.

Per-horizon EXP003 RMSE:

| h | EXP002 | EXP003 | Incremental gain vs EXP002 |
|---:|---:|---:|---:|
| 1 | 0.481311 | **0.461370** | 0.019941 |
| 2 | 0.540535 | **0.510874** | 0.029661 |
| 3 | 0.560606 | **0.529941** | 0.030665 |
| 4 | 0.563263 | **0.536326** | 0.026937 |
| 5 | 0.565692 | **0.546503** | 0.019189 |
| 6 | 0.580271 | **0.562770** | 0.017502 |
| 7 | 0.587726 | **0.573857** | 0.013868 |

The improvement is present at every horizon. Unlike EXP001/EXP002, the *incremental* TWS-history gain is largest around h=2–4 rather than growing monotonically with h, suggesting these features primarily help characterize the pre-anchor local trajectory rather than bridge the entire hidden interval by themselves.

Gain-based feature importance is highly diagnostic. The strongest new features are not the raw lags but the differences relative to the legal anchor:

1. `TWS_anchor_delta1` — 133,003;
2. `TWS_anchor_delta3` — 127,654;
3. `TWS_anchor_delta12` — 119,758;
4. `TWS_anchor_delta6` — 45,235.

The raw historical levels are much weaker. This supports the interpretation that **recent pre-anchor TWS direction / displacement is useful signal beyond the anchor level itself**.

Decision:

> **PROMOTE EXP003 to dev2 and dev3 unchanged. Do not tune LightGBM or add more TWS lags until this exact feature set is confirmed on both remaining development folds.**

---

# 35. 2026-09-07 — EXP003 full validation, lockbox shift, EXP004 kill, EXP005 launch

## EXP003 confirmed on dev2/dev3

The frozen EXP003 recipe remained superior to EXP002 on both remaining development folds:

| Fold | EXP002 | EXP003 | Incremental gain |
|---|---:|---:|---:|
| dev1 | 0.534132 | **0.509946** | +0.024186 |
| dev2 | 0.580907 | **0.571174** | +0.009734 |
| dev3 | 0.582126 | **0.557687** | +0.024439 |

The full three-fold EXP003 mean is approximately **0.546269**, improving EXP002 by about **0.019453 RMSE (~3.4%)**. Pre-anchor TWS deltas remain much more important than raw lag levels across all three folds.

## Exact historical test-mask replay

The frozen EXP003 model was then evaluated on the latest exact historical transplant of the real test mask geometry (2009-01 -> 2012-04):

```text
persistence = 0.714536
EXP003      = 0.564441
gain        = +0.150095
best_iter   = 224
```

EXP003 improved persistence at every h. The absolute gains grew from **+0.0280 at h1** to roughly **+0.345 at h7**, providing strong evidence that the direct stale-state formulation survives the actual row-level mask geometry rather than only the synthetic direct-h folds.

## Protected recent lockbox opened once

The protected 2013-11 -> 2015-08 lockbox was opened only after EXP003 had been frozen and had passed dev1-dev3 plus the exact replay.

```text
persistence = 0.823804
EXP003      = 0.685193
gain        = +0.138610
relative    = +16.83%
best_iter   = 139
```

EXP003 still beat persistence at every h, but the absolute RMSE was materially worse than dev1-dev3 and the historical replay. The deterioration is already large at h1 (`0.627776`), so the issue is not merely stale-state horizon. Biases are mostly small, arguing against a trivial global bias correction. This is treated as evidence of a genuine late-period/nonstationary difficulty shift.

Decision: **EXP003 remains the structural incumbent, but 0.685 lockbox RMSE is not yet strong enough to justify the first leaderboard submission without one focused attempt to address the shift.**

## EXP004 — 48-month recency weighting: killed

EXP004 changed only the training weights, multiplying the usual horizon-rebalance weight by exponential recency decay with a 48-month half-life. It was tested on dev3 only to avoid repeatedly optimizing against the now-open lockbox.

```text
EXP003 dev3 = 0.557687
EXP004 dev3 = 0.556573
gain        = +0.001114
relative    = +0.20%
best_iter   = 334
```

This is below the predeclared meaningful-improvement threshold and adds complexity while roughly doubling the useful boosting depth. The gain is not large enough to justify promotion.

Decision: **KILL 48-month recency weighting as a primary branch. Do not sweep decay half-lives unless later evidence specifically demands it.**

## EXP005 — causal hydrologic gap-delta ablation

The next controlled experiment keeps EXP003 frozen and adds exactly five features describing how the supplied hydrometeorology changed between the legal TWS anchor month and the current source month:

```text
SPEI_01_gap_delta
SPEI_03_gap_delta
SPEI_06_gap_delta
SPEI_12_gap_delta
SOIL_MOISTURE_gap_delta
```

For each variable:

```text
gap_delta = current_source_value - value_at_legal_TWS_anchor_month
```

The joins are exact-calendar and location-specific. No interpolation or row shift is used. For h=1, anchor month equals source month and all five gap deltas must therefore be exactly zero; the feature builder asserts this. For h>1, both endpoints are at or before the source month and are causally legal.

This deliberately tests the smallest useful hidden-interval hydrology signal before attempting richer mean/min/max/trend summaries, which may be unreproducible when intermediate calendar months are absent from the provided panel.

Evaluation policy: **run EXP005 on dev3 first against EXP003 = 0.557687. Do not touch the lockbox. Promote only if the gain is materially larger than EXP004 and is not confined to a single horizon.**

## 2026-09-07 — EXP005 dev3 result: promote to dev1/dev2 validation

Kaggle completed EXP005 on dev3 with the frozen EXP003 recipe plus only the five current-minus-anchor hydrology gap deltas.

```text
EXP003 dev3 = 0.557687
EXP005 dev3 = 0.553415
gain        = +0.004272
relative    = +0.77%
best_iter   = 212
```

This gain is materially larger than the killed EXP004 recency-weighting gain (+0.001114), so the branch is worth validating across the remaining development folds.

Per-horizon behavior versus EXP003 is mixed but diagnostic:

| h | EXP003 | EXP005 | Incremental change |
|---:|---:|---:|---:|
| 1 | 0.496829 | 0.497522 | -0.000693 |
| 2 | 0.558275 | **0.550992** | +0.007283 |
| 3 | 0.596215 | **0.586466** | +0.009749 |
| 4 | 0.591669 | **0.581395** | +0.010274 |
| 5 | 0.594565 | **0.589083** | +0.005482 |
| 6 | 0.600602 | 0.601945 | -0.001343 |
| 7 | 0.627126 | 0.632098 | -0.004972 |

The gain is concentrated at h=2-5, which is consistent with the intended interpretation: hydrology change since the legal TWS anchor helps describe the hidden interval. The h=1 near-neutral result is expected because all five gap deltas are exactly zero by construction. The h=6-7 regression means this feature set is not yet safe to promote globally without cross-fold confirmation.

Feature importance is strongly supportive of the branch. `SPEI_12_gap_delta` and `SPEI_06_gap_delta` become the #2 and #5 gain features respectively, and `SOIL_MOISTURE_gap_delta` is also substantial. `SPEI_01_gap_delta` contributes almost nothing. This suggests medium/long hydro-climate accumulation matters more than one-month change.

Decision:

> **PROMOTE EXP005 only to dev1/dev2 validation. Keep EXP003 as the incumbent until EXP005 proves a stable mean gain across all three dev folds. Do not revisit the lockbox yet.**

## 2026-09-07 — EXP005 promoted across all three development folds

EXP005 then improved the frozen EXP003 incumbent on both remaining development folds:

| Fold | EXP003 | EXP005 | Incremental gain |
|---|---:|---:|---:|
| dev1 | 0.509946 | **0.503824** | +0.006122 |
| dev2 | 0.571174 | **0.562748** | +0.008425 |
| dev3 | 0.557687 | **0.553415** | +0.004272 |

The full three-fold EXP005 mean is approximately **0.539996**, versus **0.546269** for EXP003, an improvement of approximately **0.006273 RMSE (~1.15%)**. This is a smaller step than EXP002/EXP003 but is stable across all three folds and materially larger than the killed EXP004 recency-weighting gain.

The per-horizon pattern is consistent with the feature design. The strongest incremental improvements are generally at **h=2-5**, where there is a nonzero interval between the legal TWS anchor and current hydrometeorology. h=1 is nearly unchanged, while dev3 h6-h7 regress slightly. Across folds, `SPEI_12_gap_delta` and `SPEI_06_gap_delta` are the dominant added features, with soil-moisture gap change also useful; `SPEI_01_gap_delta` is consistently negligible.

Decision: **PROMOTE EXP005 as the new development incumbent. Keep the lockbox closed to further tuning.**

## EXP006 — categorical location identity ablation

EXP006 keeps every EXP005 feature and adds exactly one new spatial feature: a categorical `location_id` deterministically derived from the supplied `(lat, lon)` pair. No target information participates in constructing the identifier. Latitude and longitude remain present, making this a strict add-one-feature ablation.

Rationale: the same ~15.7k spatial locations recur throughout the panel and are also present in test. Continuous latitude/longitude force trees to approximate location-specific structure with axis-aligned geographic partitions. A native categorical location feature gives LightGBM a cheap way to learn groups of locations with similar residual response without the compute and implementation cost of the planned EOF/PCA branch.

Evaluation policy: **run EXP006 on dev3 first against EXP005 = 0.553415. Promote only for a material gain; do not touch the lockbox.**

## 2026-09-07 — EXP006 categorical location identity: killed

EXP006 was evaluated on dev3 with every EXP005 feature unchanged plus native categorical `location_id`:

```text
EXP005 dev3 = 0.553415
EXP006 dev3 = 0.555320
change      = -0.001905 (worse)
best_iter   = 113
```

`location_id` accumulated nontrivial split/gain importance, but the extra spatial memorization did not improve held-out temporal generalization. This is evidence against exact-location identity as a useful representation for the current pooled booster.

Decision: **KILL EXP006. Do not cross-fold it.**

## 2026-09-07 — EXP007 static causal EOF loadings: killed

EXP007 kept EXP005 unchanged and added eight location-level EOF/PCA loading features. The basis was fitted only on TWS fields whose source months were inside the fold-past training prefix. On dev3, the rank-8 basis used 102 historical monthly fields through 2011-07 and explained approximately **66.0%** of historical spatial TWS variance.

```text
EXP005 dev3 = 0.553415
EXP007 dev3 = 0.554025
change      = -0.000611 (worse)
best_iter   = 118
```

Several EOF loadings were used by the tree, but their static spatial information did not translate into lower RMSE. Together with EXP006, this suggests that adding more *static* location encoding to the current booster is low ROI.

Decision: **KILL EXP007 as an add-on feature branch. Preserve EOF/state-space ideas only for a genuinely dynamic low-rank forecasting branch later, not as static tree embeddings.**

## EXP008 — horizon-specialist LightGBMs

EXP008 changes the training formulation while freezing the full EXP005 feature set and the deterministic sampled-h examples. Instead of one pooled model conditioned on `h`, it trains seven independent LightGBMs, one for each effective TWS horizon `h=1..7`, and concatenates their validation predictions before applying the exact test-h weighted RMSE.

This isolates whether the response function differs enough by stale-state age that a single pooled tree is forcing harmful compromises. Even the rare h5-h7 training groups retain tens of thousands of rows, so the specialist models are computationally feasible without changing the existing tree hyperparameters.

Critical controls:

- same EXP005 features;
- same labels and legal TWS anchors;
- same deterministic sampled-h assignments;
- same LightGBM parameters and seed;
- no lockbox tuning;
- one model per h, with h constant inside each specialist.

Evaluation policy: **run EXP008 on dev3 first against EXP005 = 0.553415. Promote to dev1/dev2 only if the combined weighted RMSE improves materially and the benefit is not driven by a single tiny horizon.**

## 2026-09-07 — EXP006 categorical location identity: killed

Kaggle tested EXP006 on dev3 with EXP005 otherwise frozen:

```text
EXP005 dev3 = 0.553415
EXP006 dev3 = 0.555320
change      = -0.001905 (worse)
best_iter   = 113
```

The exact categorical location identifier received nonzero gain importance, but it did not improve out-of-period RMSE. Performance deteriorated most clearly at longer horizons (for example h7 rose from `0.632098` under EXP005 to `0.637959`). This suggests that memorizing exact grid identity is not a useful substitute for a smoother/low-rank spatial representation.

Decision: **KILL EXP006. Do not test dev1/dev2 and do not carry `location_id` into later incumbents.**

## EXP007 — fold-causal rank-8 EOF spatial embedding ablation

EXP007 returns to the planned low-rank spatial branch. It keeps every EXP005 feature and adds eight static location loadings from an EOF/SVD decomposition of historical TWS fields.

Critical causality rule: for each validation fold, the EOF basis is fit **only on source-month TWS fields available inside that fold's training prefix**. If the fold's latest legal training target month is `c`, the EOF fit stops at source month `c-1`; validation-period TWS never participates. The decomposition receives no target column.

Implementation details:

- exact `(lat, lon)` grid columns;
- TWS fields centered by each location's training-prefix mean;
- missing location/month cells are replaced by that same training-prefix location mean before centering, giving zero anomaly rather than temporal interpolation;
- rank fixed at **8** for the first ablation;
- SVD loading signs canonicalized for deterministic reruns;
- only the eight location loadings are added to EXP005; no EOF coefficient forecast or reconstructed TWS is used yet.

Rationale: EXP006 showed that exact location memorization is not helpful, but that does not falsify spatial structure. EOF loadings provide a smooth low-dimensional description of locations that historically co-move in TWS and should be much less prone to identity overfit than a 15.7k-level categorical feature.

Evaluation policy: **run EXP007 on dev3 first against EXP005 = 0.553415. Do not revisit the lockbox. Promote only for a clear gain; otherwise stop adding static spatial encodings and move to the next temporal/training-formulation experiment.**
