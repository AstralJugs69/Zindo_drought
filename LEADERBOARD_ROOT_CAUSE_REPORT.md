# B3 leaderboard root-cause report

**Date:** 2026-09-09  
**Scope:** read-only investigation of the frozen dense-history B3/98 submission  
**Public observation:** RMSE `0.750207364` (authenticated leaderboard rank 230)  
**Submission:** `submission_b3_dense_history_98_5bc9e52.csv`, SHA-256
`62f1876ee7a26dfc5289c68d724353e9b89185d146eb7e8335a827e923424e7c`

> **Current-runtime verification (2026-09-10):** The exact diagnostic state was
> rebuilt on the persistent `zindi-gcp` VM from hash-matched Train/Test inputs.
> A fresh Dec-2014 D0 package, a Train-only full B3 model, and the matched A/B/C
> comparison now exist under `/home/milli/zindi_drought_gcp/drought_runs/`.
> The comparison was read-only and wrote no Test predictions or submission;
> its model fit ran under the isolated GCP environment, so the booster bytes
> are not expected to equal the historical Kaggle booster.  Training row,
> label, and weight hashes do match the frozen full-B3 contract.  Historical
> sections below remain valid as historical evidence; the current matched
> results are recorded in section 0.

## Executive conclusion

The public result is not explained by a corrupt CSV, an ID/order mistake, a
next-row target shift, or masked-Test TWS accidentally becoming an anchor.  The
strongest measured explanation is a combination of (1) a late-period change in
the target process and (2) historical validation that gives far more weight to
older, easier periods than the 2015--2018 Test support.  Sparse legal history
and anchor age are an additional transfer stressor, but the safe B3 feature
contract is internally reproducible and Test covariates remain inside the Train
ranges.  The exact hidden-Test error cannot be decomposed because Test labels
are unavailable.

This is a measured explanation of the gap, not a claim that one physical regime
or one geographic cell has been identified as the hidden leaderboard failure.

The current matched replay does **not** rule out sparse covariate effects.  Its
B/C intervention is empty: `sparse_withheld_window_rows=0` and
`B_C_changed_feature_count=0`, so the fixed full B3 model receives identical
features in both views and naturally produces identical predictions.  It is a
useful no-treatment control, not evidence that genuinely unavailable sparse
covariates do not matter.  The A-to-B improvement remains a joint
later-training-exposure/fitted-model effect; temporal transfer, support
mismatch, and the possible 98-round capacity limit remain open hypotheses.

## Investigation contract and provenance

- The original 2026-09-09 leaderboard investigation was read-only.  The
  authorized 2026-09-10 GCP recovery additionally fit one frozen Dec-2014 D0
  model and one frozen Train-only full B3 model solely to recreate the missing
  diagnostic state; the matched A/B/C stage itself fit no model.
- No Test prediction file, calibration, or leaderboard upload was made.  Large
  model and OOF artifacts remain remote; only compact diagnostic summaries are
  referenced below.
- The staged runner is
  `scripts/run_leaderboard_failure_investigation.py`.  The unit guards are in
  `tests/test_leaderboard_failure_investigation.py`.
- Completed stages use unique remote directories.  The target/integrity/OOF
  stages ran at commit `15a9624b5bce91f4de861e3d6a58be27d33e621b`; full-fit,
  and Test-support stages ran at `1b23897830dd88e1d7400b5a15583a7b44ac2183`;
  the final tree metadata stage ran at
  `f559243eef2a34fd68ea275d919f020cc440f2a2`;
  the final sample-inference stage ran at
  `0b8e6aabc9d49d529eeb6806e79e4e5b1721d5f5`.  These commits differ only in
  diagnostic code fixes; the saved model/data/configuration were unchanged.

## 0. Current GCP matched A/B/C rebuild (2026-09-10)

The rebuilt diagnostic ran on `zindi-gcp` in tmux session `zindi` at commit
`7e44744739065dfd6bf7a5e53ae3af28b3b79450` (the follow-up runner patch that
persists `feature_schema.json` is `09d52e3`).  The input hashes are the same
as the integrity table below.  The remote artifacts were inspected in place;
large models/OOFs were not copied to Windows.

The D0 late-fold package is
`/home/milli/zindi_drought_gcp/drought_runs/gcp_d0_dec2014_20260910.zip`
(SHA-256 `f5065e604b84289f6c33ce8a99e620702a878810289c244547e0e04d7bb73db7`),
with 109,439 validation rows, raw RMSE `0.8096875764630631`, and official
h1--7 weighted RMSE `0.7895493322211262`.  The full Train-only B3 package is
`/home/milli/zindi_drought_gcp/drought_runs/gcp_full_b3_train_only_20260910.zip`
(SHA-256 `4de3ec9d52865d1e45247d511cf7a374ac64c314ecb57d16eb4c2795122c4858`);
it fit once on 1,976,942 rows and explicitly read zero Test rows.  Its model
SHA-256 is `3895b7e384b067f606994591f653a901480854aeda57aad045e3ebe9395811de`.

The matched output directory is
`/home/milli/zindi_drought_gcp/drought_runs/gcp_matched_comparison_20260910_v2/`.
The saved D0 OOF reproduces as A within `4.440892098500626e-16`.  B and C use
the same replay ledger and structural panel; their base features are equal,
zero features changed, and their predictions/SSE are identical.  B uses
2,122,894 sparse source rows while C sees 2,154,021 retrospective dense rows;
the sparse withheld-window count is zero.  Therefore B/C is an intentionally
empty sparse-versus-dense intervention: it validates feature/ledger identity
for the no-withholding case but cannot estimate the effect of covariates that
would actually be missing in a sparse deployment.  C is not a deployable view
when those historical covariates are unavailable.

| replay scope | rows (h1--7 / >7) | A D0 weighted / raw | B full sparse weighted / raw | C full dense weighted / raw | persistence weighted / raw |
|---|---:|---:|---:|---:|---:|
| complete replay | 109,439 (109,349 / 90) | 0.789549 / 0.809688 | 0.737579 / 0.738504 | 0.737579 / 0.738504 | 0.828644 / 0.861803 |
| exact full-training exposure | 15,544 (15,544 / 0) | 0.793304 / 0.791330 | 0.738129 / 0.736472 | 0.738129 / 0.736472 | 0.834851 / 0.832697 |
| different anchor | 89,469 (89,394 / 75) | 0.787844 / 0.811208 | 0.736430 / 0.735651 | 0.736430 / 0.735651 | 0.826381 / 0.866191 |

The exposure ledger contains 105,013 rows whose source ID is in the full
training sample (15,544 exact exposure and 89,469 different-anchor rows) and
4,426 rows whose source ID is not sampled.  The corrected additive
aggregation passes every cell check: `all_ok=true`, maximum cell-versus-row
SSE difference `1.8189894035458565e-12`, and maximum RMSE decomposition error
`2.220446049250313e-16`.  The details manifest records
`no_row_level_prediction_file_written=true`, `submission_written=false`, and
`test_labels_used=false`.  Compact hashes are
`matched_comparison.json` `17505659518cdcee156946258d7f414bf87ff7d6c8b9af65e5193c257547536a`,
`matched_comparison_details.json`
`764f7fe16dda19976126dd4eb0bdb3ff3a1bf49a60d933c708ecee94435ec9dc`, and
`matched_overall.csv`
`d7a84c3c1e7a957e80a36c5e372f71efd2d32bcdf8ea470cefd538194c83c192`.

This paired result is diagnostic rather than a new leaderboard candidate.  It
does not authorize the old undefined recency-weighted experiment; that idea is
closed.  A fixed 98-vs-392-round capacity comparison is now the bounded next
diagnostic, and must not produce Test outputs or a submission.

## 1. Integrity and exact alignment

The remote preflight read the source files by chunks and compared IDs and
headers without copying them locally:

| Artifact | Rows | SHA-256 / result |
|---|---:|---|
| Train.csv | 2,154,021 | `97ff1912b35871574a01900c24a792a9a418653e87f01dccc1788c6838d94b1f` |
| Test.csv | 280,961 | `314da7996fa947b30797d82ea8d0ea34240fe52683ecf102a21c37f00f61c196` |
| SampleSubmission.csv | 280,961 | `77881bc7257791d583c5a1f496a5639864d490a0237515338fc5305ec943db22` |
| submitted B3 CSV | 280,961 | `62f1876ee7a26dfc5289c68d724353e9b89185d146eb7e8335a827e923424e7c` |

Train, Test, SampleSubmission, and the submitted CSV all have unique IDs.  The
Test ID sequence is exactly the SampleSubmission sequence; the submitted CSV
has the same set and order, two `ID,Target` columns, 280,961 finite targets,
and zero duplicate IDs.  Test has 186,913 masked TWS values; all five supplied
hydro-covariate columns are finite.

The label contract was checked independently with an exact calendar join:
`(lat, lon, source_month + 1 calendar month) -> TWS_t`.  Of 2,154,021 Train
rows, 1,977,398 have a next-calendar row and every matched target differs by
exactly `0.0` (maximum and mean absolute difference `0.0`).  The remaining
176,623 rows are explicitly unmatched terminal rows; they are not silently
shifted or dropped.  There are zero duplicate `(lat, lon, calendar_month)`
keys.  This rules out the common row-order/next-available-row explanation for
the leaderboard gap.

The compact integrity outputs are in
`/kaggle/working/drought_runs/leaderboard_failure_20260909T123000Z/`:
`integrity_details.json` (SHA-256
`6fe37037330495b9aa6940679e52477f2d0c7b28b0cb3a7828d07dfa82497ac3`),
`target_alignment_details.json` (SHA-256
`b1f306d4cc07ec7035ee5e8d931c619d8d02adb0aa3f34a4227660cf5ce6bedf`), and
their stage manifests.

### Fresh saved-model inference check

A deterministic stratified sample of 1,686 Test IDs (the requested cap was
2,048; the exact month-by-horizon strata contained 1,686 rows) was rebuilt from
the saved model, full legal feature maps, and the saved visibility contract.
The rebuilt predictions join the submitted CSV on ID with 1,686/1,686 rows and
maximum absolute difference `4.440892098500626e-16`.  Prediction order is
unchanged for batch sizes 1, 17, 257, and 1,686, and for a deterministic
shuffled request order (all maximum differences `0.0`).

The same test also changed all 186,913 raw masked TWS values by `+123456` before
rerunning the legal visibility builder.  The legal ledger was unchanged and
the maximum prediction difference was `0.0`.  Changing all hydro covariates
strictly after source month 2016-03 left 259 earlier sampled predictions
unchanged (`0.0` maximum difference).  These are strong implementation
invariance results; they do not establish hidden-label generalization.

Evidence: `/kaggle/working/drought_runs/leaderboard_failure_20260909T160000Z/`
`sample_inference_details.json` (SHA-256
`74d0a19d945333a98c56d8679c13211c62a66acaee94ae249ec93e9c6f387e45`).

## 2. What became difficult in 2015?

The direct one-month target-minus-current-TWS behaviour changes sharply in
2015.  The table uses every supplied Train row in the stated source month;
RMSE is `target - TWS_t`, not a model score.

| Source month | Rows | Target-current RMSE | Bias | |change| > 1 | > 2 | > 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2012-01 | 15,648 | 0.613278 | -0.259167 | 9.87% | 0.10% | 0.00% |
| 2012-02 | 15,666 | 0.615270 | -0.066005 | 9.60% | 0.75% | 0.03% |
| 2013-01 | 15,644 | 0.433167 | 0.004985 | 3.75% | 0.12% | 0.00% |
| 2013-02 | 15,665 | 0.515264 | -0.099645 | 6.53% | 0.61% | 0.00% |
| 2013-06 | 15,576 | 0.520721 | 0.164651 | 6.02% | 0.13% | 0.00% |
| 2014-01 | 15,652 | 0.554146 | 0.090567 | 8.14% | 0.50% | 0.01% |
| 2014-06 | 15,579 | 0.536063 | 0.101295 | 7.60% | 0.13% | 0.00% |
| **2015-01** | **15,653** | **1.012330** | **0.029057** | **26.67%** | **6.74%** | **1.01%** |
| **2015-02** | **15,670** | **1.189415** | **0.007460** | **28.69%** | **8.94%** | **3.57%** |
| **2015-06** | **15,581** | **1.181583** | **-0.272616** | **38.05%** | **9.60%** | **1.79%** |

The RMSE identity was checked for every monthly group:
`RMSE² = bias² + centered variance`, with absolute residuals below
`2.3e-16`.  Thus the 2015 deterioration is overwhelmingly centered
variability/extreme changes, not a small global bias that could be safely
corrected after the fact.

The matched location-by-calendar-month panel gives the same result.  Comparing
2015 with the mean of the same location/month keys in 2012--2014 yields
difference RMSE/bias/centered-std of:

| Calendar month | Matched rows | Difference RMSE | Bias | Centered std |
|---|---:|---:|---:|---:|
| January | 15,653 | 1.194051 | 0.083461 | 1.191130 |
| February | 15,669 | 1.293305 | 0.089959 | 1.290172 |
| June | 15,581 | 1.358138 | -0.405583 | 1.296164 |

This is a calendar-matched temporal shift, not merely a different set of
locations.  The source panel covers 138 calendar months from 2002-05 through
2015-08, 15,715 locations, and has zero duplicate location-month rows.  A
global location-month rectangle would contain 2,514,400 slots; 360,379 are
unobserved, which is reported as a support fact rather than imputed.

### Spatial concentration of the 2015 change

For January, February, and June 2015, rows were binned into 5-degree cells and
ranked by their target-current squared error.  The error is broad rather than
confined to one suspect cell:

| Source month | Cells / rows | Cells needed for 25% SSE (rows) | 50% SSE (rows) | 75% SSE (rows) | Top 10-cell SSE share |
|---|---:|---:|---:|---:|---:|
| 2015-01 | 893 / 15,653 | 29 (668) | 85 (1,885) | 197 (4,241) | 10.70% |
| 2015-02 | 891 / 15,670 | 20 (473) | 66 (1,503) | 175 (3,795) | 15.52% |
| 2015-06 | 892 / 15,581 | 29 (698) | 87 (2,056) | 200 (4,490) | 12.07% |

The largest cells have both positive and negative signed changes.  A small
geographic specialist or a single-cell data defect is therefore not supported.
The complete ranked table is
`target_cells_2015.csv` (SHA-256
`e73bb4d1c8eaf0e1691bb0174b15862ae843c1180bf3ce6c7530449431e00329`).

## 3. Existing OOF decomposition

The runner read all six saved D0 files: four outer origins (2007-09, 2009-01,
2014-04, 2014-12) and the two inner origins (2003-04, 2004-04).  Inner and
outer evidence are kept separate conceptually.  Rows are joined and grouped by
`(origin, sample_id)`, never by `sample_id` alone; persistence is the legal
`last_observed_TWS` anchor.

| Origin | Rows | B3 raw RMSE | Persistence raw RMSE | B3 h1-7 weighted |
|---|---:|---:|---:|---:|
| 2003-04 (inner) | 275,077 | 0.580719 | 0.679050 | 0.580620 |
| 2004-04 (inner) | 274,671 | 0.560806 | 0.666394 | 0.560684 |
| 2007-09 (outer) | 278,447 | 0.528354 | 0.625852 | 0.528340 |
| 2009-01 (outer) | 273,200 | 0.560377 | 0.711862 | 0.560226 |
| 2014-04 (outer) | 77,960 | 0.582737 | 0.768072 | not available: 97 h>7 rows |
| 2014-12 (outer) | 109,439 | 0.809688 | 0.861803 | 0.789549 |

The four outer files pooled to raw RMSE `0.595116` versus persistence
`0.712119`; the h1-7 weighted values are `0.583339` versus `0.689653`.
Those figures are not an independent confirmation set: the 2007-09 and
2009-01 physical target keys overlap by 60,376, while the two inner origins
overlap by 75,953; all older origins contribute many more rows than the latest
stress fold.

The Dec-2014 outer fold is the relevant late-transfer warning.  Its h1-7
per-horizon B3/persistence RMSEs are:

| h | Rows | B3 | Persistence | B3 - persistence |
|---:|---:|---:|---:|---:|
| 1 | 15,637 | 0.636306 | 0.637859 | -0.001553 |
| 2 | 15,627 | 0.842080 | 0.875550 | -0.033470 |
| 3 | 15,607 | 1.045686 | 1.093427 | -0.047741 |
| 4 | 15,659 | 0.549182 | 0.669449 | -0.120266 |
| 5 | 15,638 | 0.606719 | 0.675876 | -0.069156 |
| 6 | 15,596 | 0.675010 | 0.755945 | -0.080935 |
| 7 | 15,585 | 1.120588 | 1.162736 | -0.042148 |

The corresponding source-month decomposition (already recorded in the prior
investigation) is Dec-2014 `0.636`, Jan-2015 `0.843`, Feb-2015 `1.053`,
Mar `0.541`, Apr `0.604`, May `0.675`, and Jun `1.120`; aggregate
prediction-minus-target bias is `+0.210063`.  The stress tail is only 90 rows,
so including h>7 does not explain the late deterioration.

Compact OOF outputs are under
`/kaggle/working/drought_runs/leaderboard_failure_20260909T123000Z/`, including
`oof_overall.csv` (SHA-256
`a97c04e61ddd77b87453f9bc8317a73299cd88fa277b582d83ab50c7d29fb234`),
`oof_by_horizon.csv`, `oof_by_anchor_age.csv`, `oof_by_geo5.csv`,
`oof_late_months.csv`, `oof_late_cells.csv`, and `oof_overlap.csv`.

## 4. Saved full-model fit versus transfer

The full-fit stage reconstructed the exact frozen training sample and checked it
against the saved configuration:

- 1,976,942 sampled legal rows; horizon counts `721,266 / 440,770 / 302,504 /
  197,211 / 103,809 / 107,884 / 103,498` for h1--h7.
- 177,079 missing-anchor rows dropped, exactly as in the original manifest.
- Row-ID hash `e96d01f8d1c233b49fe64fd82e1c54616b862cab33863e29bf413a2a6999faf2`,
  label hash `39c3cd2d1cc4bee9ced9bc71df75e80e32f0ab073ac83100efa2e72c98222be2`,
  and weight hash `9f1a66082cffd7c84da17344bbf50f9426a713d63babe968c205fc1c9c495330`
  all match the saved config.
- Train/Test/SampleSubmission hashes match the original fit preflight.
- The saved model is 98 trees, 742,881 bytes, SHA-256
  `a1f1c08882676ca00d8fc613fe118d241ceadb2cf61db3ee5d0af6ffa6632c11`.
- Rebuilt 446-feature schema hash is
  `ef2a7dbfb77c56b96d741e07a1b68a45ea6f9eabae00a41c929ea5b4bfb12a41`.
- The independent tree-metadata pass finds exactly 98 trees, each with 63
  leaves (6,174 leaves total; minimum/mean/maximum `63 / 63.0 / 63`).  The
  parsed model reports total split gain `3,269,653.4463882446`; these values
  are structural metadata, not a claim of causal importance.

Scoring those rows without refitting gives in-sample B3 RMSE `0.535382`, MAE
`0.393937`, bias `-0.000059`, versus persistence RMSE `0.686105`.  The gap from
this fit score to the Dec-2014 OOF stress score (`0.809688`) is direct evidence
of transfer/fit optimism; it is not evidence of an export error.  In the late
2015 source months, in-sample B3 still improves persistence (for example,
2015-02 `1.031749` vs `1.168093`, 2015-06 `0.893896` vs `1.168804`) but leaves
large residual variability, consistent with a changed target process.

Saved-tree split gain is descriptive, not causal.  Across 98 trees it is:

| Feature family | Split-gain share |
|---|---:|
| Legal TWS anchor/history | 38.957% |
| Regional context/support | 31.758% |
| Local hydro/trajectory | 19.913% |
| Geography/horizon | 9.372% |

The fit outputs are in
`/kaggle/working/drought_runs/leaderboard_failure_20260909T150000Z/`:
`full_fit_details.json` (SHA-256
`02dd693943767b3bc3b12cd85a26536168b9edbc0786cfe37c15a20b3ea24491`),
`fit_by_source_month.csv`, `fit_by_geo5.csv`, `fit_late_2015_by_geo5.csv`, and
`fit_tree_gain_groups.csv` (SHA-256
`25daaba45b9166038507109a918e50046e63bd37c44557c9c26ceaa2e5c09169`).
The leaf-count metadata is in
`/kaggle/working/drought_runs/leaderboard_failure_20260909T170000Z/`:
`tree_inspection_details.json` (SHA-256
`2a2fe5b7cf98859724f8cb33fbb7406e2c0e9fdc3096ff3c6d36ed8b81525cbb`) and
`fit_tree_gain_groups.csv` (SHA-256
`25daaba45b9166038507109a918e50046e63bd37c44557c9c26ceaa2e5c09169`).

## 5. Test support and representativeness

The actual Test has 280,961 rows across 18 supplied source months and the exact
competition h counts `94,048, 62,576, 46,777, 31,076, 15,560, 15,479,
15,445` for h1--h7.  The legal Test ledger has 280,961 unique rows and no
missing anchor.  Source-month support is sparse: for example, local
SPEI-01 finite observations average about 2.99 in a 3-month window and 5.98 in
a 6-month window at 2015-09, but only 1.00 and 2.98 at 2016-01.  By 2017-06,
the mean legal anchor age is about 5.97 months and the 3-/6-month local history
counts are about 3.00/5.99.  Thus elapsed anchor age and observed history do
change materially across Test events.

This is not a simple numeric covariate extrapolation: for each of the five
hydro variables, Test current and legal-anchor values have zero strict
out-of-Train-range share, and the same is true for the recent Dec-2014 OOF
support rows.  All supplied current/anchor values are finite.  The evidence
therefore points to missing/sparse temporal state and target-process change,
not a gross SPEI/soil scale mismatch.

The compact support tables are
`test_support_by_month_h.csv` (SHA-256
`a65529709175f230a417d72094d8033ba481a0fb035c15d0fc60138c8d897ac2`) and
`validation_support_recent.csv` (SHA-256
`7bc9771de723cd2b81d6f88c4e4c8a5f0457a8be7adc9d8cc78f69bdfcef1bdc`).

## 6. Ranked hypotheses

| Rank | Hypothesis | Evidence for | Evidence against / limits | Verdict |
|---:|---|---|---|---|
| 1 | Late temporal/regime transfer: 2015 target changes are much larger and more variable than the historical mix | 2015 target-current RMSE 1.01--1.19 vs 0.43--0.61 in matched earlier months; 2015 matched-panel RMSE 1.19--1.36; Dec-2014 OOF is the only late stress fold and is much worse; full-fit optimism is large | Test labels are hidden; no physical source/provenance attribution was proved | **Strongest explanation** |
| 2 | Validation support is unrepresentative: older dense/overlapping folds dominate pooled metrics, while Test has sparse months and growing anchor age | Exact Test h mix and legal ledger; history counts/anchor ages vary; outer pooled h1-7 `0.583339` and older origins dominate row count; overlap is measured | B3 safe feature contract and invariance tests pass; support shift alone cannot quantify hidden error | **Supported contributor** |
3 | Export/ID/order or hidden-TWS leakage defect | None after exact hashes, ID joins, replay, masked perturbation, shuffled/batch checks | These tests do not prove every possible numerical bug | **Contradicted for the tested paths** |
4 | Supplied Train target is a row-shift or duplicate-location corruption | Exact next-calendar join, zero duplicate location-month keys, zero matched differences | Terminal rows naturally lack a next row; external upstream revisions were not independently audited | **Not supported** |
5 | Pure capacity or post-export calibration is the main cause | Full fit is optimistic; tree relies heavily on anchor/history | Same frozen model beats persistence in every measured late slice; no safe global correction can explain centered 2015 variability; no calibration was fit | **Possible mechanism, not isolated** |

The public score (`0.750207364`) is a hidden-label Test observation.  Earlier
ledgered public scores (`0.75789118` and `0.753313479`) and the OOF values are
not interchangeable samples; none supports a claim of independent confirmation
or a location-specific hidden failure.  The duplicate platform rows already
recorded in `NEXT_ACTION.md` were not retried.

## 7. Exactly one next experiment: capacity budget (bounded, not a grid)

The old undefined recency-weighted proposal is closed: July/August 2015 was
already used as development evidence, so it is not an untouched confirmation.
The bounded capacity test instead compares the unchanged D0-dense B3 recipe at
**98 versus 392 boosting rounds**.  It uses the current corrected observation
view, the same seed (`20260908`), learning rate, 63 leaves,
`min_data_in_leaf=1000`, 446 features, sampled IDs, labels, anchors, and
horizon weights.  Features are built once per origin and reused; there is no
early stopping, feature/recency/depth/loss/neural/ensemble grid, Test
prediction, calibration, or submission.

The primary out-of-time origins are **2014-04** and **2014-12**, with training
targets strictly before each origin.  Save iteration-98 and iteration-392
models/OOFs, train and validation RMSE, monthly/horizon/5-degree-cell additive
SSE summaries, exact identity hashes, and checkpoint drift against any
independently recovered D0 control.  April's missing h=4 is reported as
incomplete support; it is not imputed and does not block its raw comparison.

Advance only if iteration 392 improves available raw h1--7 RMSE by at least
`0.005` on both recent origins and has no more than `0.02` raw-RMSE regression
on any recent source month with at least 1,000 rows.  If that passes, run one
2009-01 transfer check and require no more than `0.005` h1--7 raw-RMSE
regression.  Otherwise close the capacity hypothesis.  Any later full-data
diagnostic remains Train-only and separately authorized.

## Artifact index and reproducibility

Current GCP compact outputs remain remote and are inspectable with
`ssh zindi-gcp`:

- `/home/milli/zindi_drought_gcp/drought_runs/gcp_d0_b3_parity_20260910/` —
  frozen Sep-2007 parity manifest.
- `/home/milli/zindi_drought_gcp/drought_runs/gcp_d0_dec2014_20260910/` —
  rebuilt late D0 model/OOF package.
- `/home/milli/zindi_drought_gcp/drought_runs/gcp_full_b3_train_only_20260910/` —
  Train-only full B3 model, schema, and manifest.
- `/home/milli/zindi_drought_gcp/drought_runs/gcp_matched_comparison_20260910_v2/` —
  matched A/B/C details and compact tables.

Historical Kaggle paths are retained as provenance labels only.  Their
post-restart artifacts were not independently recovered and are **not** claimed
to remain accessible through `ssh kaggle`:

- `/kaggle/working/drought_runs/leaderboard_failure_20260909T123000Z/` —
  integrity, exact alignment, target behaviour, cell concentration, OOF
  decomposition, and overlap tables.
- `/kaggle/working/drought_runs/leaderboard_failure_20260909T150000Z/` —
  full saved-model fit, Test support, and recent-validation support.
- `/kaggle/working/drought_runs/leaderboard_failure_20260909T170000Z/` —
  independent 98-tree leaf metadata and the mapped split-gain table.
- `/kaggle/working/drought_runs/leaderboard_failure_20260909T160000Z/` —
  deterministic sample inference/invariance report.

The stage manifests and SHA-256 values above are the durable provenance.  No
large model, OOF, or raw challenge artifact was copied to Windows for this
report.
