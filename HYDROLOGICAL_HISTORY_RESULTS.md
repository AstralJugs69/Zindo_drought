# Hydrological history results

## 2026-09-09 — B3 leaderboard root-cause investigation

`LEADERBOARD_ROOT_CAUSE_REPORT.md` contains the complete read-only diagnostic
record. The investigation used separate, memory-bounded Kaggle stages and did
not fit, calibrate, score a new candidate, write a Test prediction file, or
upload to Zindi. The compact stage outputs remain remote under
`/kaggle/working/drought_runs/leaderboard_failure_20260909T123000Z/` (integrity,
target alignment/behaviour, and OOF decomposition),
`leaderboard_failure_20260909T150000Z/` (saved-model fit, tree gains, and Test
support), `leaderboard_failure_20260909T160000Z/` (sample inference), and
`leaderboard_failure_20260909T170000Z/` (independent tree metadata).

Key reproducibility results:

- The submitted B3 CSV SHA is still
  `62f1876ee7a26dfc5289c68d724353e9b89185d146eb7e8335a827e923424e7c`; Test and
  SampleSubmission IDs match exactly and all 280,961 predictions are finite.
- Exact `(lat, lon, source_month + one calendar month)` target alignment has
  1,977,398 matched rows with maximum difference `0.0`; 176,623 terminal rows
  are explicitly unmatched; no duplicate location-month keys exist.
- Rebuilding the saved model and legal 446-feature maps for 1,686 deterministic
  Test IDs matches the submitted file to `4.44e-16`. Batch sizes 1/17/257/1686,
  shuffled request order, all masked-TWS perturbations, and later-covariate
  causality checks have maximum difference `0.0`.
- The saved booster has exactly 98 trees, each with 63 leaves (6,174 total;
  min/mean/max `63 / 63.0 / 63`). Mapped split-gain shares are legal
  TWS/history `38.957%`, regional context/support `31.758%`, local
  hydro/trajectory `19.913%`, and geography/horizon `9.372%`; this is
  descriptive structure, not causal attribution.

Measured transfer evidence explains why the public result is lower than the
pooled historical picture. Target-current RMSE in Train source months
2015-01/02/06 is `1.012330/1.189415/1.181583`, versus `0.433167--0.615270`
for the matched 2012--2014 months; 2015 matched-panel differences are
`1.194051/1.293305/1.358138` RMSE for Jan/Feb/Jun. The 75%-SSE thresholds are
197/175/200 5-degree cells (only 27.1%/24.2%/28.8% of rows), so the change is
dispersed rather than a single geographic defect. The four outer D0 folds pool
to B3/persistence raw RMSE `0.595116/0.712119` and h1--7 weighted
`0.583339/0.689653`, but Dec-2014 is `0.789549` weighted and `0.809688` raw.
The full saved-model fit is `0.535382` versus `0.686105` persistence, making
fit optimism visible. Test hydro values remain finite and inside Train ranges;
support differences are mainly sparse legal history and anchor age.

**Decision:** rank late-period target-process/transfer shift first and sparse
Test support second; treat export/ID/target-row defects as contradicted for the
tested paths. The only proposed follow-up is the frozen late-prefix,
Test-schedule-matched recency-weighted B3/98 experiment documented in
`NEXT_ACTION.md`; it is not run or authorized as a submission.

**Status:** Stage A and all fixed-capacity Stage B outer/inner ablations are
completed. B3/98 is the frozen Stage B selection by mean inner weighted RMSE;
the margin over B2 is too small to treat as a practical deployment promotion.
**Stage B runner provenance:** outer B1/B2/B3 runs used
`545fe9963fe6388eb89e065b9f39afcc233ba081`; the inner-fold extension and
selection runs used `c5591b428ea24bc960ddb172079de70fda957d16`; Stage C
controls used `3eacd99130cb278cdd69b7b5871b0730dbdaa6a0`.
**Execution pin:** pass the checked-out, pushed full `git rev-parse HEAD` value
to each runner's `--expected-commit`; the runner records it in its manifest.
**Environment:** the repaired Kaggle checkout and experiment root are writable
by `kaggle`; all fitting/scoring in this record is Kaggle-only.

## 2026-09-09 — full-training B3/98 Test submission and Zindi result

The authorized frozen deployment run used
`scripts/run_b3_full_submission.py` on Kaggle from commit
`8fa8956e3a27da544f5954292ea5ff6bc44c9c22`. It fit dense-history B3/98 once
(`fit_count=1`, seed `20260908`, 98 rounds, 63 leaves,
`min_data_in_leaf=1000`) on every legal supplied Train label. The run used
`1,976,942` sampled legal rows, dropped `177,079` missing-anchor rows, and
retained all horizons h=1..7. The final feature schema has 446 columns. The
exact replay integration check passed on 278,447 rows with maximum prediction
difference `4.440892098500626e-16` against the saved D0 model/OOF path.

Test-time contract checks were completed without using Test labels. The
competition visibility ledger contains 280,961 rows: 94,048 visible Test TWS
rows and 186,913 masked rows, with no illegal-anchor exclusions or masked
self-anchors. Batched inference used 20,000-row batches, produced finite
predictions for every SampleSubmission ID, and generated exactly one local
output file with no clipping or calibration:

```text
remote: /kaggle/working/drought_runs/b3_full_submission_20260909T010000Z/submission_b3_dense_history_98_5bc9e52.csv
local:  artifacts/b3_full_submission_20260909T010000Z/submission_b3_dense_history_98_5bc9e52.csv
rows:   280,961
bytes:  11,110,853
sha256: 62f1876ee7a26dfc5289c68d724353e9b89185d146eb7e8335a827e923424e7c
```

The remote manifest is `status=completed`, `fit_count=1`, and records the same
source commit and output SHA. Only the CSV and small manifests were copied to
Windows; the model, visibility ledger, and other large training artifacts were
not independently recovered after the Kaggle runtime restart.  The historical
Kaggle path is retained as provenance only; current artifacts must be inspected
on `zindi-gcp`.

### Zindi processing record

The authenticated Zindi submissions API was checked on 2026-09-09. The account
state contained **two** successful rows with this exact B3/98 filename rather
than one: `1JLvpsYJ` at `2026-09-09T04:13:32.148Z` and `38KHQMrj` at
`2026-09-09T04:13:54.579Z`. Both report public score `0.750207364` and
`status=successful`; the platform metadata exposes the filename but not a
payload SHA. The authenticated leaderboard shows `astraljugs` at rank **230**
with public score `0.750207364` and four total submissions. This ledger records
the duplicate platform state as observed and intentionally issued **no third
upload**. The local/remote output SHA above is the reproducibility hash for the
file associated with the run; the public leaderboard score is not independent
generalization evidence.

## 2026-09-09 — deterministic covariate-history gap augmentation (final decision)

This experiment asked whether B3/98 becomes more availability-faithful when its
training covariates are exposed to deterministic, Test-shaped history gaps. The
Test table was used only to derive the set of relative source-date masks (no
labels, target values, Test predictions, or submission files were produced).
For each sample, the implementation in
`src/covariate_gap_augmentation.py` recomputes the legal B3 trajectories under
the mask, preserving calendar time, the current slot, and the legal TWS anchor.
It produces the same 446-column schema as B3. `D0_dense` is the unmasked
reference, `D1_sparse` applies the sparse Test-like mask, and `D2_mixed` mixes
the two deterministically. The schedule-pattern fingerprint for every run is
`5f3c8dca628d30982f933fa0b6b861f93130376087b61ea5236893880c1c5f7d`.

The training sample IDs, labels, weights, validation IDs, LightGBM settings
(98 rounds), and row sampler were held fixed within each origin. Inner fits used
the predeclared 2003-04 and 2004-04 replays; augmented recipes used seeds
20260909 and 20260910. A recipe qualified only if its seed-mean official h=1..7
weighted RMSE improved D0 on both inner origins; the lower equal-origin mean
then selected the recipe for the outer replay.

### Inner selection

| Inner origin | D0 dense | D1 sparse (two-seed mean) | D2 mixed (two-seed mean) | D1 gain vs D0 | D2 gain vs D0 |
|---|---:|---:|---:|---:|---:|
| 2003-04 | 0.580620 | 0.578124 | **0.578100** | +0.002496 | +0.002520 |
| 2004-04 | 0.560684 | **0.555256** | 0.555782 | +0.005427 | +0.004901 |
| Equal-origin mean | 0.570652 | **0.566690** | 0.566941 | — | — |

Both augmented recipes passed the directional inner screen. `D1_sparse` was
selected by the predeclared lower equal-origin mean (the analyzer records the
selection rule and the 0.0001 D2 tie preference).

### Conditional outer replay

The selected D1 recipe was fit with both seeds on all four declared outer
origins and compared by paired validation IDs with the corresponding D0 fit.
The official metric is the competition h=1..7 weighted RMSE. April 2014 has no
h=4 rows in the replay calendar, so its official metric is intentionally `n/a`;
the renormalized present-horizon value is diagnostic only and cannot pass the
promotion gate.

| Origin | D0 official | D1 official / present | D1 − D0 (official / present) | D0 raw | D1 raw | h>7 tail D0 / D1 |
|---|---:|---:|---:|---:|---:|---:|
| 2007-09 | 0.528340 | 0.522337 | **−0.006003** | 0.528354 | 0.522351 | n/a / n/a (0 rows) |
| 2009-01 | 0.560226 | 0.556057 | **−0.004169** | 0.560377 | 0.556197 | n/a / n/a (0 rows) |
| 2014-04 | n/a (h=4 absent) | n/a / 0.546358 | n/a / **+0.006348** | 0.582737 | 0.587337 | 1.103301 / 1.001798 (97 rows) |
| 2014-12 | 0.789549 | 0.793342 | **+0.003792** | 0.809688 | 0.810825 | 1.124937 / 1.119641 (90 rows) |

The April present-horizon denominator is only 0.889394 of the official weight
because h=4 is absent. The h-by-h table is retained in the analyzer artifact;
for April, the D1-minus-D0 raw RMSE deltas are +0.010784 (h1), +0.005206
(h2), +0.004630 (h3), −0.017271 (h5, five rows), +0.007949 (h6), and
−0.003834 (h7). The recent-origin regressions are concentrated in fresh
anchors: D1-minus-D0 raw RMSE by anchor age (0, 1, 2, 3+) is −0.008053,
−0.003504, −0.002304, −0.006054 for 2007-09; +0.001267, −0.004775,
−0.004970, −0.007117 for 2009-01; +0.011007, +0.005548, +0.005028,
+0.002049 for 2014-04; and +0.014100, +0.003685, −0.002959, −0.000447
for 2014-12.

The analyzer therefore records `official_gate_complete=false`, blocker
`2014-04`, and `promote=false`. Even though D1 helps the two older outer
origins and slightly reduces the December stress tail, it regresses December's
complete official score and worsens April's supported-horizon diagnostic. The
equal-origin present-horizon delta (−0.000008) is not evidence of promotion
because the official gate is incomplete.

**Durable decision:** reject `D1_sparse` as a training/deployment replacement
and retain dense **B3/98** as the strongest verified development baseline. Keep
the augmentation implementation and Kaggle artifacts as audit evidence; do not
run a tuning grid, generate Test predictions, or create a submission from this
branch. These four outer replays remain development robustness evidence, not
independent confirmation.

### Reproducibility and artifact index

The inner/outer runner implementation was committed at
`40fde25ec766084937451b5ab8e0ec188de389ae`; the December full-tail correction
was `5dd67cbaa459dba63909038cb8488cda50910811`; the final analyzer/reporting
tests are in `8154633d450ed9ca76b038f7c72b23ccc0cef621`. The completed remote
checkout is clean at the latter commit. All run directories below are under
`/kaggle/working/drought_runs/` and each manifest records the fixed training,
validation, feature, and source fingerprints.

| Artifact | Remote path | Size (bytes) | SHA-256 |
|---|---|---:|---|
| Inner analysis | `covariate_gap_inner_20260909T000000Z_analysis_v3/gap_augmentation_analysis.json` | 3,578 | `0f76aefab39be32e0a98512ce76d6beeac6d7c0f42b4f1dceb1e2bc3f16225bb` |
| Outer analysis | `covariate_gap_outer_20260909T000000Z_analysis_v3/gap_augmentation_analysis.json` | 24,674 | `97af01d9f48204991f2779fefcb378be3f3a6230e76ca51247423f4bbe5c5c9a` |
| Outer D0 2007-09 package | `covariate_gap_outer_20260909T000000Z_D0_2007_09` | 10,387,175 | `89a3c8f6bdb7c5f90006ba41954878844d117a91d989fe80568185eee98f2da9` |
| Outer D1 2007-09 packages (s09/s10) | `...D1_2007_09_s20260909`, `...D1_2007_09_s20260910` | 10,376,966 / 10,380,810 | `5b5a0e349f14f574b8d4c02f26592be8baed4c8bd89a59b3d685b71965cc1613` / `40fdbe127dc6d2481b792b2130783a9bb2918b33fa7b32ebbc7f3354a08aa206` |
| Outer D0 2009-01 package | `covariate_gap_outer_20260909T000000Z_D0_2009_01` | 10,233,865 | `66108024abb79e7fd3393f058bee2fc01481beb5f906c8c141d8f6dff9707d99` |
| Outer D1 2009-01 packages (s09/s10) | `...D1_2009_01_s20260909`, `...D1_2009_01_s20260910` | 10,245,259 / 10,241,082 | `7e5bd6645c833364609fa5a84e122bc94ad522e2141a69c2a1d6ae3f6ee1ebb9` / `22fff1e6927c75db401902092cdf63b74fbebf793799f9ceb9da87c301959475` |
| Outer D0 2014-04 package | `covariate_gap_outer_20260909T000000Z_D0_2014_04` | 3,161,345 | `46113f752b106d462cbb8ae1b5053afcec702cc494ec79897579f0f6128a1084` |
| Outer D1 2014-04 packages (s09/s10) | `...D1_2014_04_s20260909`, `...D1_2014_04_s20260910` | 3,158,733 / 3,157,955 | `673e4862c0fa0c19c041ac10fcdb0b536f884468a2fc561f5e9870bb344735bb` / `075593f4ac0490367ecae458ba89c0e65377a164336a705daf0234ba53c665e8` |
| Outer D0 2014-12 package | `covariate_gap_outer_20260909T000000Z_D0_2014_12` | 4,311,661 | `e73a4fd4bd920248757587dad139b45c8194a9e12be4fe0f6158570b31bd50c3` |
| Outer D1 2014-12 packages (s09/s10) | `...D1_2014_12_s20260909`, `...D1_2014_12_s20260910` | 4,307,893 / 4,308,635 | `2e415391030e18ab4f5f5b3bbb7bb5dcf65893d79df6290c6a2a55ce87c1f734` / `77077a83805abf2092ea0c2a2d7063d11fc80489756f9e4a5d3da4f7edf33465` |

The ten inner package hashes are recorded in their manifests: D0
2003-04/2004-04 (`f458c046b32ab44d02130564efa9dc7a4fb235132a6b83a139378f71ebc74b56`,
`f7d590e6a6ca7a2eef5aa5750301052e7269f1d91fc6d952bf298895b3d68095`), D1
2003-04/2004-04 seeds 09/10 (`a25602951e7e37a3bf1ba0a835debe0963b9f8e1e6df1f2eaf6776791072d44d`,
`a6c33c835819fdea809f1321d50841e79623f3fd3117c2c6be1b85395fe8ed4e`,
`7623bad3da95e9c1bb9d8fc10188933ba975335b14906041d4c1e3e1eb1c06bf`,
`5f4e885b53abbf584aa528a3a84fb7bd2f7bd25998e175d3dbeb8588bebe23f0`), and
D2 2003-04/2004-04 seeds 09/10
(`1154f989008593d99f9738aad2bdb18374395c81e36d3c6d4f7e151dd61c6ad1`,
`64abbcaab7a340ff137c50d50c008362b02887645af7f8f9ab8ec374ae8438ce`,
`fcc8b2ee51e1d26545505583e7e9a94540a627aaf6cf658258edabf4ad0674a2`,
`b5e890b92ab6ce93638151043ec76bffce2b9f36ae17314def14ab5f2bf6e2b2`).

## Verified inherited baseline evidence

The completed Kaggle artifact
`/kaggle/working/drought_runs/regional_capacity_matched_20260908T0300Z` is
present. Its capacity was selected only on the frozen chronological inner
replays: C0 selected 59 rounds; combined regional C1 selected 98 rounds, both
with 63 leaves and `min_data_in_leaf=1000`. The corresponding C1/98 outer
evaluation is **not** present in that artifact: its outer comparison is C1 at
the C0-matched 59 rounds. Therefore C1/98 must not be claimed as scored or as a
baseline metric until a distinct run evaluates it.

The verified C1/59 information comparison, all development evidence, is:

| Origin | Frozen R01/173 | C0/59 | Regional C1/59 | C1 gain vs C0 |
|---|---:|---:|---:|---:|
| 2007-09 | 0.537422 | 0.535141 | 0.527274 | 0.007867 |
| 2009-01 | 0.574104 | 0.582306 | 0.567747 | 0.014558 |
| 2014-04 | 0.609887 | 0.615340 | 0.599374 | 0.015966 |
| 2014-12 h=1..7 | 0.822625 | 0.812785 | 0.808313 | 0.004472 |

This supports regional contemporaneous covariates at matched capacity across the
four already-used development/stress replays. It neither establishes a C1/98
outer result nor independently confirms any recipe.

## Stage A — B0 availability and distribution audit

Kaggle run `history_availability_audit_20260908T055537Z` completed from the
clean, pinned commit `1810664d9145c07e2ebc7497555a5da63522a3a2` in **209.57 s**.
It made no Test predictions. The target-blind B0 schema contains 56 features;
all are present in Test with zero missingness-share gap. Test covers the same
893 five-degree spatial cells and horizons 1--7. The historical replay ledger
contains 738,956 rows across 2007-09, 2009-01, 2014-04 and 2014-12; Test has
280,961 rows.

| Diagnostic | Result | Interpretation |
|---|---:|---|
| Broad spatial-grouped domain AUC | 0.936608 | Historical and Test rows are readily separable. |
| Matched `h` / season / 5-degree-cell domain AUC | 0.900016 | Shift remains after geometry matching; do not treat it as a feature gain. |
| Test mass in unseen horizon/season/geo strata | 0.000000 | No categorical-support or feature-contract defect found. |

The largest simple standardized mean gaps are legal anchor TWS (0.1775),
horizon (0.1572), seasonal cosine (0.0964), and current SPEI gaps (at most
0.0793). The matched classifier still has unavoidable disjoint-calendar time
confounding, so this evidence is a deployment-risk guardrail rather than a
causal explanation. Stage B remains eligible, but its paired frozen-origin
controls—not the audit AUC—decide promotion.

The remote artifact bundle is
`/kaggle/working/drought_runs/history_availability_audit_20260908T055537Z/audit_artifacts.zip`
(8.8 KiB, SHA-256
`7f20174f4894a45c8ee1afd856f43efd6b6b08a24616a1c66e3c25a8d65d2788`). The
identical, Git-ignored local durable copy is
`artifacts/history_availability_audit_20260908T055537Z.zip`.

## Stage B0 — fixed 98-round C1 reference pilot

Kaggle run `hydro_trajectory_b0_pilot_20260908T060339Z` completed from clean
commit `d21ccba667840d88c20b7c0dc037f835996bb437` in **549.23 s**. It is the
plan's `B0 = C1/98` current-regional information set: 56 base/current-regional
features, no local or regional trajectory block. The Test schema has the same
56 names and is finite-or-missing; the runner constructed that contract but made
no Test prediction.

| Origin | Candidate | Training / validation rows | Raw RMSE | Weighted RMSE |
|---|---|---:|---:|---:|
| 2007-09 | B0 = C1/98 | 851,814 / 278,447 | **0.523809** | 0.523795 |

All horizons 1--7 are represented. The fit took 32.80 s after preprocessing.
The run recorded 110 local and 280 regional trajectory-map features during the
structural pilot, even though B0 did not consume them. The phase snapshots show
5.715 GiB RSS when those maps were ready and 6.189 GiB after fitting; a live
monitor observed a temporary approximately 12.7 GiB RSS during construction.
This pilot commit did not persist a true peak, so a continuous peak-RSS/minimum-
available-memory tracker has been added before the next run. Do not infer a
capacity gain from comparison with the inherited C1/59 artifact: it is a
separately persisted run with different sampled-row provenance. This is the
first actual outer C1/98 score, not independent confirmation.

The remote archive is
`/kaggle/working/drought_runs/hydro_trajectory_b0_pilot_20260908T060339Z.zip`
(8,126,728 bytes, SHA-256
`c23a3d8229825bc0d50ead0c8c6b3558b3cfb6c60b16d04ab5907a91c6f62c4a`). The
checksum-matched, Git-ignored local copy is
`artifacts/hydro_trajectory_b0_pilot_20260908T060339Z.zip`.

## Stage B — completed B0 reference and single-block ablations

The three remaining B0 origins, B1 local history, and B2 regional history ran
from the clean, pinned commit `545fe9963fe6388eb89e065b9f39afcc233ba081` at a
fixed 98 rounds. Each run constructed a Test feature contract only and reports
`no_test_predictions=true`; none generated Test predictions or a submission.
The B0 2007-09 pilot above used the preceding `d21ccba` commit, but the runner's
scoring/feature logic and its exact training-row hash match the later runs; the
later commit adds continuous memory instrumentation and documentation only.

| Origin | B0 raw RMSE | B1 local raw RMSE | B2 regional raw RMSE | B3 both raw RMSE | Best raw RMSE |
|---|---:|---:|---:|---:|---:|
| 2007-09 | 0.523809 | 0.517816 | 0.518658 | **0.517361** | B3 |
| 2009-01 | 0.559511 | **0.555030** | 0.555645 | 0.556115 | B1 |
| 2014-04 | 0.590070 | 0.584558 | **0.581446** | 0.582737 | B2 |
| 2014-12 h=1..7 | 0.811981 | 0.809599 | 0.809731 | **0.809371** | B3 |

`B1_local` adds 110 local trajectory columns (166 total); `B2_regional` adds
280 regional/disagreement columns (336 total). Both improvements have the same
favorable direction across all four already-used development/stress replays,
with no raw-RMSE deterioration, so they meet the outer-guardrail portion of the
plan's practical screen. The frozen 2003-04 and 2004-04 inner replays remain
required before a candidate can advance to Stage C. This is development
evidence, not independent confirmation or authorization to produce a
competition prediction.

The B0 remaining run finished in 565.45 s with a 13.168 GiB peak RSS and 8.607
GiB minimum available memory. Its remote package is
`/kaggle/working/drought_runs/hydro_trajectory_b0_remaining_20260908T063458Z.zip`
(13,953,232 bytes, SHA-256
`afd84ca18cc1fd306eac20f09fd498316295876d3cc9cfdf8d170909446abedf`); the
checksum-matched local ignored archive is
`artifacts/hydro_trajectory_b0_remaining_20260908T063458Z.zip`.

The B1 package is
`/kaggle/working/drought_runs/hydro_trajectory_b1_local_20260908T064928Z.zip`
(22,109,320 bytes, SHA-256
`d30136fb4357285fc1173e1050d079e7dfe170f92011888f33b8fb25971828a7`),
completed in 784.37 s with a 13.422 GiB peak RSS. Its checksum-matched local
ignored archive is `artifacts/hydro_trajectory_b1_local_20260908T064928Z.zip`.
The B2 package is
`/kaggle/working/drought_runs/hydro_trajectory_b2_regional_20260908T070438Z.zip`
(22,101,247 bytes, SHA-256
`95b8c2e1230d7a46d0a80e1ab6a428fbfa2b0ce9eec1fd85e6acc09faf62715e`),
completed in 986.61 s with a 20.521 GiB peak RSS. At this checkpoint those two
full packages are remotely durable. The checksum-matched local ignored B1/B2
archives are `artifacts/hydro_trajectory_b1_local_20260908T064928Z.zip` and
`artifacts/hydro_trajectory_b2_regional_20260908T070438Z.zip`, respectively.

## Stage B3 — combined-block ablation

Because B2's all-origin peak was 20.521 GiB, B3 first ran its 446-column
combined feature set on the smallest declared replay as a resource-safety pilot.
The clean `545fe99` Kaggle run
`hydro_trajectory_b3_resource_pilot_20260908T074000Z` completed 2007-09 in
473.45 s: raw/weighted RMSE **0.517361 / 0.517346** on 851,814 / 278,447
training/validation rows. That is a +0.006449 raw-RMSE gain versus B0, while
the continuous peak RSS was only 14.626 GiB (13.774 GiB minimum available).
The 446-column Test contract was finite-or-missing, and no Test prediction was
made. Its remote package is
`/kaggle/working/drought_runs/hydro_trajectory_b3_resource_pilot_20260908T074000Z.zip`
(8,121,484 bytes, SHA-256
`3afe59557412effe91a2fd0417fb3f54e826f034f06a8a96ca97b72c396bba6f`).

The remaining three B3 replays then completed in the clean, pinned job
`hydro_trajectory_b3_remaining_20260908T074500Z`: raw/weighted RMSE was
0.556115 / 0.555967 (2009-01), 0.582737 / n.a. (2014-04), and
0.809371 / 0.789596 (2014-12 h=1..7). Thus B3 improved B0 in all four outer
replays by +0.006449, +0.003396, +0.007332, and +0.002610 raw RMSE,
respectively. It did not dominate the single-block variants in every replay:
B1 is lower by 0.001084 at 2009-01 and B2 is lower by 0.001291 at 2014-04.
The job completed in 1,012.04 s with a 26.643 GiB peak RSS and 3.514 GiB
minimum available memory. Its 446-column Test contract was finite-or-missing,
`no_test_predictions=true`, and it created no Test prediction. The remote
package is
`/kaggle/working/drought_runs/hydro_trajectory_b3_remaining_20260908T074500Z.zip`
(13,993,051 bytes, SHA-256
`3ba286a2e60221e906e26ea58772e716cf8a786e768397a83d8d71a277f36318`).

## Frozen Stage B inner selection

The same fixed 98-round candidates then ran, one candidate per job, on the
predeclared 2003-04 and 2004-04 template inner replays from clean commit
`c5591b428ea24bc960ddb172079de70fda957d16`. Each terminal manifest reports
`no_test_predictions=true`; each source fold has the exact same candidate-row
hash, labels, replay geometry, and horizon support within its origin.

| Candidate | 2003-04 raw / weighted RMSE | 2004-04 raw / weighted RMSE | Mean inner weighted RMSE | Result |
|---|---:|---:|---:|---|
| B0 | 0.582833 / 0.582705 | 0.560782 / 0.560624 | 0.571665 | reference |
| B1 local | 0.585386 / 0.585258 | 0.557430 / 0.557283 | 0.571270 | fails direction: worse than B0 at 2003-04 |
| B2 regional | 0.581978 / 0.581879 | **0.555266 / 0.555137** | 0.568508 | consistent B0 improvement |
| B3 both | **0.580254 / 0.580153** | 0.556942 / 0.556813 | **0.568483** | selected by mean-inner rule |

B3 improves B0 on both inner replays (+0.002579 and +0.003840 raw RMSE), so it
and B2 clear the direction gate. The predeclared mean-inner weighted rule picks
`B3_both` at 0.568482991 versus B2's 0.568508034—a **0.000025043** difference.
That is far too small to establish a meaningful B3-over-B2 deployment advantage,
but avoids a post-hoc outer-replay choice: the fixed Stage C tree control is
**B3 at 98 rounds**, and B2 remains a materially competitive simpler reference.

The completed inner packages are remotely durable: B0
`hydro_trajectory_inner_b0_20260908T111000Z.zip` (15,810,000 bytes, SHA-256
`09812e43b526e337bcf747570459e7127bf1138970bd529cd2fd03bda65d16ee`), B1
`hydro_trajectory_inner_b1_local_20260908T112000Z.zip` (15,954,570 bytes,
SHA-256 `b3366c4fa0af038828ca9b2f4d7f3b1d1b41b0610de875d159268044215c1f94`),
B2 `hydro_trajectory_inner_b2_regional_20260908T113000Z.zip` (15,835,325
bytes, SHA-256 `e48793b7950e18c8d3c6e541c8f0002c1e175eb6c70ed90ff30307a561730802`),
and B3 `hydro_trajectory_inner_b3_both_20260908T114000Z.zip` (15,899,996
bytes, SHA-256 `eddb642c4570535c397b53b55fb9588f2698422df463937605615a1b1447ddb3`).

## Stage C — direct-history controls only

Kaggle run `hydro_sequence_controls_20260908T120000Z` completed from clean
commit `3eacd99130cb278cdd69b7b5871b0730dbdaa6a0` in 584.15 s. It reconstructs
fixed six- and twelve-calendar-month local hydrology grids with explicit
per-variable observation masks and true calendar-offset channels. The flattened
tree controls append that direct history to the selected 446-column B3/98
static block. The ridge controls use the same features, missing indicators, and
a fixed alpha=1000 with medians/centers/scales fitted only from each training
prefix. Test was used only to reconstruct frozen replay geometry; no Test
feature prediction, competition prediction, or submission was made.

| Candidate | 2003-04 raw / weighted RMSE | Gain vs B3 raw | 2004-04 raw / weighted RMSE | Gain vs B3 raw |
|---|---:|---:|---:|---:|
| B3/98 tree reference | **0.580254 / 0.580153** | — | **0.556942 / 0.556813** | — |
| Flattened tree, 6 months | 0.581004 / 0.580908 | -0.000751 | 0.557140 / 0.557011 | -0.000198 |
| Flattened tree, 12 months | 0.583449 / 0.583347 | -0.003196 | 0.558067 / 0.557932 | -0.001124 |
| Prefix-normalized ridge, 6 months | 0.619102 / 0.619002 | -0.038849 | 1.566490 / 1.566394 | -1.009548 |
| Prefix-normalized ridge, 12 months | 0.788872 / 0.788772 | -0.208619 | 1.060587 / 1.060624 | -0.503644 |

The tree reference exactly reproduced B3's selected inner scores and row hashes.
Neither direct-history tree improves either frozen replay; both ridge controls
are substantially worse. Therefore the Stage C prerequisite—an equivalent
flattened-history control with usable signal beyond B3—is falsified. **Neural
sequence modelling was not tested** by this run: no GRU, TCN, or MLP was fit.
This is a negative result for the tested compact flattened local raw-history
formulation, not a claim about neural sequence architectures. The former no-GRU
gate is superseded by the separately predeclared, availability-faithful neural
comparison in `NEURAL_SEQUENCE_EXPERIMENT_PLAN.md`.

The remote archive is
`/kaggle/working/drought_runs/hydro_sequence_controls_20260908T120000Z.zip`
(78,524,335 bytes, SHA-256
`18ae60a07b4569dd341fbb5086555e3906146b065fe585322779be824f0ae25d`). Its
terminal manifest records 10 result rows, `no_test_predictions=true`, a 13.971
GiB peak RSS, and 5.640 GiB minimum available memory.
The small local ignored decision record is durable at
`artifacts/hydro_sequence_controls_20260908T120000Z.manifest.json` and
`artifacts/hydro_sequence_controls_20260908T120000Z.metrics.json`; the latter
matches the manifest's `metrics.json` SHA-256
`b14fbfbd1f8401faac2b665b809ead1104cd6eca831177b7ec2eed16ab813853`.

## Implemented and verified

- `scripts/run_history_availability_audit.py` produced the Stage A target-blind
  B0-vs-Test availability/distribution audit above, including broad and matched
  horizon/season/5-degree-cell domain diagnostics with spatial grouped folds.
- `src/hydro_trajectory.py` constructs causal local 3/6-calendar-month
  covariate summaries and regional trajectory/disagreement features. It has no
  TWS or target input.
- `scripts/run_hydrological_trajectory.py` implements the fixed-capacity paired
  B0, B1-local, B2-regional, B3-both ablation. It validates Test feature-schema
  parity but never predicts Test targets or creates a submission.
- `WINNER_METHOD_AUDIT.md` records source/code evidence and transfer limits.

Local verification at `3eacd99`: `python -m pytest -q` passed **33 tests**;
syntax compilation passed for the trajectory and Stage C control runners. On
the executing Kaggle environment, `tests/test_hydro_trajectory.py` passed
**7/7** before Stage C. This is implementation verification, not
model-performance evidence.

## Unanswered questions

| Question | Status |
|---|---|
| Did Stage A reveal a feature-contract defect? | No: all 56 B0 features are available in Test; material temporal distribution shift remains. |
| Do local trajectories help B0? | Yes in these four development/stress replays: raw gain +0.002382 to +0.005994. |
| Do regional trajectories add beyond contemporaneous C1? | Yes in these four replays: raw gain +0.002250 to +0.008623. |
| Does C1/98 transfer to the declared outer replays? | Measured for all four: B0 raw RMSE 0.523809 / 0.559511 / 0.590070 / 0.811981. |
| Does combined local plus regional history improve all four replays? | Yes versus B0: raw gain +0.002610 to +0.007332; it is not the best single candidate on every replay. |
| Does a sequence model add beyond the selected B3/98 tree control? | Not yet answered by this historical control: the six-/twelve-month flattened controls are worse on both frozen inner replays, but neural sequence modelling was not tested. |
| Is any new candidate independently confirmed? | No. |

## Statistical scope

The 2007-09, 2009-01, 2014-04, and 2014-12 h=1..7 periods have already
influenced development choices. They are not untouched confirmation evidence.
The two inner capacity replays overlap in five target calendar months and are
selection evidence only. Any later confirmation requires an independently
verified unused period and a recipe frozen before one evaluation.

## Stage C — availability-faithful compact neural comparison

This completed comparison supersedes the earlier statement that neural sequence modelling was untested. It used the predeclared sparse replay observation view, frozen B3/98 static context, deterministic training-only 252,000-row cap, prefix-only normalization, two initial seeds, and no Test prediction or submission. Inner candidates were static MLP, causal GRU, and causal TCN; every accepted fit recorded CUDA and passed the train-only learnability check.

Inner selection from `/kaggle/working/drought_runs/neural_sequence_inner_20260908T171500Z` chose the 12-slot GRU at epoch 1: mean raw RMSE **0.602960** over two origins and two seeds. This is not a B3 comparison and was frozen before outer fitting. The package is 22,576,764 bytes, SHA-256 `a8be8bb7b80cf590538f81d31aaecb790a2bc7efa5c5d367ffbf8a5314f89021`.

The availability-faithful outer comparison pairs the selected GRU's two-seed mean OOF with B3 by `(origin, sample_id)`. The initial job was kernel-killed while allocating the final origin; already-written immutable OOFs were recovered and only 2014-12 was rerun after a sequential-memory repair. The repair archive is `neural_sequence_outer_20260908T190000Z_repair_2014_12.zip`, 22,576,764 bytes, SHA-256 `194983f264b50f39666dd55f13d96c43400c60d3fb955bdff3c5d6a94f969e5c`. The joined analysis is at `/kaggle/working/drought_runs/neural_sequence_outer_20260908T190000Z_analysis_recovered`.

| Origin | B3 raw RMSE | GRU raw RMSE | GRU minus B3 | Equal blend minus B3 |
|---|---:|---:|---:|---:|
| 2007-09 | 0.528354 | 0.546294 | +0.017940 | +0.001220 |
| 2009-01 | 0.560377 | 0.590095 | +0.029718 | +0.006879 |
| 2014-04 | 0.582737 | 0.608675 | +0.025937 | +0.007775 |
| 2014-12 | 0.809393 | **0.804180** | **-0.005213** | **-0.006429** |
| all outer OOF | **0.595026** | 0.613169 | +0.018143 | +0.002334 |

The overall result covers 738,956 rows and intentionally has no one-number official weighted score because it includes an h=8--12 stress tail; competition weights apply only to h=1--7. Calendar-block paired bootstrap for neural-minus-B3 raw delta is +0.009815 to +0.026459 (2.5--97.5%); 5-degree geography bootstrap is +0.015793 to +0.020731. The predeclared equal blend interval is -0.001173 to +0.006248 with an adverse point estimate. Residual correlation is 0.957139.

The third seed at the frozen selected epoch had RMSE 0.612227 (2003-04) and 0.602575 (2004-04), mean 0.607401 versus initial two-seed mean 0.602960. Its archive is `neural_sequence_stability_20260908T220000Z.zip`, 36,018,558 bytes, SHA-256 `341d6e066a6b99273fabe7b17d1a50ce7e4d343aa71502500af78be47cd9cf1f`.

The predeclared history ablation hid only older raw sequence slots, retaining current source channels, their masks/ages/calendar fields, and B3 static context. It tests dependence on older temporal inputs, not an information-free model. Hiding history increased mean inner raw RMSE from 0.602960 to **0.608462** (+0.005503): 2003-04 +0.001433 and 2004-04 +0.009573 across two seeds. Thus the branch used some lawful historical information, but it did not translate into a robust B3 improvement. The ablation package is `neural_sequence_ablation_20260908T223000Z.zip`, 71,949,551 bytes, SHA-256 `d0b6d9c83a7bb82a821200968c88c5724c8709e4cb99fd4c03e6be6df1d75a8b`. Remote metric-only follow-up output is `/kaggle/working/drought_runs/neural_sequence_followup_analysis_20260908T223000Z/`; its JSON SHA-256 is `71afbb70fd1b76fd05df6ae2312db0663eef9b66c47188e55f80238832ed94f4`.

**Decision:** retain availability-faithful B3/98 as the strongest verified development baseline. Do not promote the compact GRU or equal B3/GRU blend; the lone 2014-12 gain is insufficient against three earlier regressions and dependence-aware uncertainty. These outer replays are development robustness evidence, not independent confirmation. No untouched confirmation, Test prediction, or submission was produced.
