# Hydrological history results

**Status:** Stage A and all fixed-capacity Stage B outer ablations (B0--B3) are
completed. The two frozen 2003-04/2004-04 inner replays have not yet selected a
candidate.
**Latest trajectory-code SHA:** `545fe9963fe6388eb89e065b9f39afcc233ba081`
**Execution pin:** pass the checked-out, pushed full `git rev-parse HEAD` value
to each runner's `--expected-commit`; the runner records it in its manifest.
**Environment:** the repaired Kaggle checkout and experiment root are writable
by `kaggle`; all fitting/scoring in this record is Kaggle-only.

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
completed in 784.37 s with a 13.422 GiB peak RSS. The B2 package is
`/kaggle/working/drought_runs/hydro_trajectory_b2_regional_20260908T070438Z.zip`
(22,101,247 bytes, SHA-256
`95b8c2e1230d7a46d0a80e1ab6a428fbfa2b0ce9eec1fd85e6acc09faf62715e`),
completed in 986.61 s with a 20.521 GiB peak RSS. At this checkpoint those two
full packages are remotely durable; local transfer is recorded separately only
after an exact checksum match.

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

Local verification at `545fe99`: `python -m pytest -q` passed **30 tests**;
syntax compilation passed for the trajectory runner. On the executing Kaggle
environment, `tests/test_hydro_trajectory.py` passed **4/4** after the
continuous-memory tracker was added. This is implementation verification, not
model-performance evidence.

## Unanswered questions

| Question | Status |
|---|---|
| Did Stage A reveal a feature-contract defect? | No: all 56 B0 features are available in Test; material temporal distribution shift remains. |
| Do local trajectories help B0? | Yes in these four development/stress replays: raw gain +0.002382 to +0.005994. |
| Do regional trajectories add beyond contemporaneous C1? | Yes in these four replays: raw gain +0.002250 to +0.008623. |
| Does C1/98 transfer to the declared outer replays? | Measured for all four: B0 raw RMSE 0.523809 / 0.559511 / 0.590070 / 0.811981. |
| Does combined local plus regional history improve all four replays? | Yes versus B0: raw gain +0.002610 to +0.007332; it is not the best single candidate on every replay. |
| Does a sequence model add beyond equivalent history inputs? | Not started; it remains gated on frozen inner-replay selection after Stage B. |
| Is any new candidate independently confirmed? | No. |

## Statistical scope

The 2007-09, 2009-01, 2014-04, and 2014-12 h=1..7 periods have already
influenced development choices. They are not untouched confirmation evidence.
The two inner capacity replays overlap in five target calendar months and are
selection evidence only. Any later confirmation requires an independently
verified unused period and a recipe frozen before one evaluation.
