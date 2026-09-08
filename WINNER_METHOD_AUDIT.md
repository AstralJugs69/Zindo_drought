# Winner-method audit

**Audit date:** 2026-09-08  
**Purpose:** turn the supplied winner references into bounded, causal experiment
inputs. A method is not treated as verified implementation merely because an
interview describes it. Third-party repositories were inspected as research
material only and were not executed.

## Decision summary

The strongest transferable idea is not a winner's full pipeline. It is the
small, testable combination of calendar-aware covariate trajectory summaries and
same-month regional context now defined as Stage B in
`HYDROLOGICAL_HISTORY_EXPERIMENT_PLAN.md`. The Stage A matched domain diagnostic
is motivated by the air-quality references, and a compact masked GRU remains a
conditional Stage C candidate. No winner evidence authorizes noncausal
imputation, row-order lags, full-season aggregates, random time folds, a log
target, SMAPE, or a large TFT in this project.

| Reference | Evidence status | Concrete project action |
|---|---|---|
| Wazihub Soil Moisture | Official first-place interview; no linked source code located | Test causal counters, elapsed-time slopes, and duration features in B1/B3. |
| Layer.ai Air Quality | Supplied primary pages were not fetchable through the available public reader; code/notebook not independently located | Run the predeclared broad and matched domain-shift diagnostic; gate a compact sequence model on measured history signal. |
| AirQo Uganda | Repository and actual notebook/code inspected at pinned commit | Retain fold-aware OOF discipline only; reject its backwards lags and non-calendar indexing. |
| GEOAI Cropland | Official first-place article and repository code inspected at pinned commit | Test only causal 3/6-month trend, variability, and regional features; reject annual/full-history summaries and random CV. |
| IBM SkillsBuild Hydropower | Supplied official article was not fetchable through the public reader; no code link verified | Treat the compact sequence-plus-static-context pattern as methodology-only, conditional Stage C evidence. |

## A. Wazihub Soil Moisture Prediction

- **Verified challenge/rank:** The [official interview](https://zindi.world/learn/meet-the-winners-of-the-wazihub-soil-moisture-prediction-challenge) identifies Olayinka Fadahunsi / DrFad as first place. It describes next-few-days plot soil-humidity prediction from past sensor data with incomplete/noisy values.
- **Code:** No first-place repository or notebook is linked in the inspected interview; this is methodology-only evidence.
- **Verified method:** the interview reports counter features, lag features, field-specific flow/drain-rate features, irrigation duration, a linear rise/decline assumption, and an ensemble of five tree models.
- **Inference/validation/loss:** the article does not document an implementation file, fold scheme, or training loss. Those properties are **not verified**.
- **Transfers:** causal observed-duration, actual-elapsed-time slope, current-minus-trailing-level, and per-location support fields. These map directly to `src/hydro_trajectory.py`.
- **Does not transfer:** filling values from observations “around” a missing point can consume a future value; field-specific flow rules have no established analogue for global TWS; the stated linear behavior is a hypothesis, not a physical law here.
- **Concrete experiment:** B1 local and B3 both, with source covariates only, real-calendar 3/6-month windows, masks/support, and matched B0/C1 rows.

## B. Layer.ai Air Quality Prediction

- **Primary sources:** [official winner article](https://zindi.world/learn/meet-the-winners-of-the-layerai-air-quality-prediction-challenge) and [first-place technical discussion](https://zindi.africa/competitions/layerai-air-quality-prediction-challenge/discussions/13525).
- **Verified challenge/rank/code:** not independently verified in this audit. The public reader rejected or timed out on these URLs, and no linked notebook could be inspected. The supplied descriptions of adversarial diagnostics, TFT/PyTorch Forecasting, shared locations, and unseen-station handling are therefore **methodology claims awaiting source access**, not implementation facts.
- **Transfer with guardrails:** a train-versus-Test classifier is useful as a diagnostic, not an objective. `scripts/run_history_availability_audit.py` uses a shallow, fixed, spatial-grouped classifier in broad and horizon/season/5-degree-matched settings, then reports remaining calendar confounding.
- **Does not transfer:** no log target (TWS is signed), no SMAPE, no unexamined TFT recipe, and no assumption that spatial treatment of new stations applies to this stable location grid.
- **Concrete experiment:** Stage A; Stage C only after B shows legal-history signal and the flattened controls are beaten.

## C. AirQo Ugandan Air Quality Forecast

- **Primary announcement:** [Zindi discussion](https://zindi.africa/competitions/airqo-ugandan-air-quality-forecast-challenge/discussions/1708) was not publicly retrievable in this audit, so the winner rank is not independently verified.
- **Code inspected:** [mnm-rnd/competitions AirQo directory](https://github.com/mnm-rnd/competitions/tree/master/zindi/airqo-ugandan-air-quality-forecast-challenge), commit `3e00e9c06de6aa0c3cc16cf44b36a1f68bde315b` (shallow clone on 2026-09-08).
- **Supporting implementation:** `README.md` describes a 121-observation pseudo-series and stacked LightGBM/CatBoost workflow. `FinalNotebook.ipynb` builds GroupKFold base OOF (`GroupKFold`, `n_splits=3`, cells around lines 210–303) and adds the grouped mean OOF feature to the CatBoost input (around lines 351–354); it then uses a 50-fold KFold CatBoost fit. `mlod/models/booster_models.py` implements LightGBM/CatBoost RMSE models.
- **Critical non-transfer:** `mlod/preprocessors.py:lag_shift` creates both `group.shift(+k)` and `group.shift(-k)` values. The latter is a future observation for a forecast event. Its index is observation count rather than calendar time. The notebook's random/meta KFold is not a chronological replay. These are not valid feature or validation templates for TWS.
- **Transfers:** OOF provenance as a downstream feature only when the split is legal; explicit feature construction and an ensemble as a separately validated option.
- **Concrete experiment:** retain exact-ID OOF and residual-correlation analysis for any future blend. Do **not** stack during Stage B; only a weight selected on inner OOF can advance.

## D. GEOAI Cropland Mapping in Dry Environments

- **Verified challenge/rank:** the [official article](https://zindi.world/learn/winning-solution-to-the-geoai-cropland-mapping-challenge) identifies Chigozie Nkwocha as first place. Its target is binary cropland classification, with accuracy as the reported competition metric.
- **Code inspected:** [Chygos/competitions GeoAI directory](https://github.com/Chygos/competitions/tree/main/zindi/GeoAI_cropland_mapping), commit `d3166f1e95bb33fd61532ec7337ff2d6567fe687` (shallow clone on 2026-09-08).
- **Supporting implementation:** `notebooks/feature_engineering.py` defines harmonic regression, monthly trend/acceleration, 6-month rolling summaries, coordinate bins, and regional aggregation; `notebooks/cropland_modelling.py` trains LightGBM/XGBoost/CatBoost classifiers and uses `StratifiedKFold` accuracy CV.
- **Transfers:** use a *small*, causal version of trend, acceleration proxy, variability, and regional disagreement features. This specifically motivated the 3/6-calendar-month local and regional feature block.
- **Does not transfer:** its code aggregates annual/full-season series and fits per-ID harmonic regression over complete history. That is valid for static classification but illegal for a forecast-time event. Random stratified CV, probability averaging, class thresholding, and accuracy do not match signed TWS/RMSE.
- **Concrete experiment:** B1/B2/B3, with no annual summaries, no future harmonics, and the same replay rows and RMSE metric as B0.

## E. IBM SkillsBuild Hydropower Climate Optimisation

- **Primary source:** [official second-place article](https://zindi.world/learn/2nd-place-solution-tackling-the-future-of-hydropower).
- **Verified challenge/rank/code:** the supplied URL was not fetchable through the available public reader in this audit and no source repository/notebook was located. The rank and reported compact-sequence/static-branch architecture remain methodology-only evidence.
- **Transfer:** a compact sequence branch with static geography/season/horizon context is a sensible *architectural analogue*, not proof that neural sequence modelling will help TWS.
- **Does not transfer:** its target, time resolution, inputs, loss, and validation protocol are unverified here; none may be assumed compatible.
- **Concrete experiment:** Stage C GRU only after Stage B legal-history controls pass. It uses explicit masks/elapsed months, prefix-fitted normalization, signed anchor-delta output, two context spans, and two fixed seeds.

## Audit consequences

The supplied winner material supports a causally bounded feature test and better
evaluation discipline, but it does not support a claim that sequence modelling,
physical decay, imputation, or a large architecture will improve the incumbent.
Every promoted project method remains subject to the availability contract, the
predeclared plan, and the paired replay evidence.
