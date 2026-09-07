# A Step Ahead of Drought — Master Research, Validation, and ML Development Plan

**Competition:** Zindi — *A Step Ahead of Drought: Forecasting Global Water Storage Challenge*  
**Organizer:** ITU / AI for Good, with Copernicus/JRC context  
**Competition close:** 2026-09-13  
**Document role:** Canonical research ledger, modeling blueprint, experiment discipline, leakage specification, compute plan, and living decision record for the entire challenge.  
**First compiled:** 2026-09-07  
**Current phase:** Research saturation completed; notebook/model training intentionally not started yet.  
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
