# Hydrological history experiment plan

**Status:** predeclared 2026-09-08, before fitting any new candidate  
**Baseline source commit:** `9b0080cc9f60b4e765ef8b0e069cccc50defe84e`  
**Execution:** Kaggle only; no Test predictions, submission files, or Zindi submissions.

## Decision question

Can causally constructed local and regional hydrological trajectories improve the
availability-safe regional LightGBM model, and does a compact sequence model add
value beyond the same history supplied to a tree model?  All reported outer
periods are development/replay evidence unless an untouched period is explicitly
reserved and frozen before its one-time use.

## Contract and frozen comparator

The target is next **calendar** month's TWS.  For a source event at month `t`, a
feature may use only source observations at or before `t`, and its TWS state is
the legally visible anchor recorded in the replay ledger.  Missing calendar
months remain missing: no row-order shifting, future filling, interpolation,
smoothing, target-derived regional value, or compressed elapsed time is allowed.
The Test file is read only for schema, source-field availability, and its supplied
mask/template; it is never used for a target or a competition prediction.

The verified current comparison is combined 5-degree + 15-degree regional C1 at
the C0-matched 59 rounds (63 leaves, `min_data_in_leaf=1000`), with raw RMSE
0.527274, 0.567747, 0.599374, and 0.808313 at the 2007-09, 2009-01, 2014-04,
and 2014-12 h=1..7 replays. C1 selected 98 rounds on the two historical inner
capacity replays, but its outer 98-round score has not been run. The Stage B
pilot first establishes this `B0=C1/98` fixed-capacity comparator, then B1--B3
use the identical 98-round capacity. The inner replays are 2003-04 and 2004-04;
their five target months overlap and are selection evidence only.

## Stage A — feature availability and shift audit

Before the trajectory fit, construct the exact B0 source/ledger features for
historical replays and actual Test inputs, without labels.  Report horizon,
season, 5-degree geography, missingness, anchor age, covariate age, local and
regional support, calendar-window support, and current/anchor covariate shifts.

Fit a diagnostic-only, deterministic 100-iteration shallow histogram gradient
domain classifier twice: (1) an equal-size broad sample, and (2) a sample matched
by horizon, season, and 5-degree cell.  Use five spatial grouped folds by 5-degree
cell; report mean/standard deviation AUC and class prevalence.  This detects
separability, not a feature-selection objective.  It cannot remove the intrinsic
calendar-period confounding between historical Train and later Test, which must
be disclosed.  A missing field or different train/inference feature construction
is a contract defect and blocks every affected baseline until fixed and rerun.

## Stage B — causal trajectory ablation

The shared history builder receives only target-blind source covariates
`SPEI_01_t`, `SPEI_03_t`, `SPEI_06_t`, `SPEI_12_t`, and `SOIL_MOISTURE_t` and
the replay ledger.  For each variable it emits, where support exists, causal
3- and 6-calendar-month mean, standard deviation, min, max, elapsed-month slope,
current-minus-mean, recent change, change of slope, drying/recovery direction,
observed run length, window count, coverage, and age.  Values with inadequate
support remain missing with explicit support/age indicators.  The builder uses
calendar keys and never labels.

Regional trajectories use the existing contemporaneous 5-degree and 15-degree
context, restricted to source covariates available at that event.  They add
regional trend/variability/support, local-minus-regional level/trend, and
local/regional-disagreement indicators.  Regional aggregations must exclude
unavailable observations.

| Candidate | Features beyond B0 | First comparison |
|---|---|---|
| `B0` | none | first establish the inner-selected C1/98 comparator |
| `B1_local` | local trajectory block | B0 + local block at 98 rounds |
| `B2_regional` | regional trajectory block | B0 + regional block at 98 rounds |
| `B3_both` | local + regional trajectory blocks | B0 + both blocks at 98 rounds |

All four use identical folds, source/label cutoffs, sampled training rows,
weights, seed, and 98-round tree capacity.  The runner writes feature schema and
hashes, row/label/replay hashes, OOF keyed by exact `sample_id`, models, metrics,
and logs.  It first executes one 2007-09 `B0` timing pilot, records build/fit
duration and peak memory, then schedules only one candidate/job at a time.  The
prior comparable local-response run took about six minutes for three origins;
the pilot replaces that rough estimate before the full ablation.

An ablation may advance to inner capacity selection only when it has a consistent
direction on the two frozen inner replays and no raw-RMSE deterioration greater
than 0.003 on any of the four development/stress replays.  The chosen capacity,
if any, is selected only from the existing inner procedure and then frozen for
the four outer replays.  These gates limit effort; they do not turn prior outer
replays into independent confirmation.

## Stage C — direct history learning, conditional

Proceed only if the Stage B invariance tests pass, feature construction is
identical for training and inference, and an added-history tree control shows
that the legal history has non-negligible signal.  First compare a flattened/tree
history control and a simple regularized linear/MLP control using the same rows.
Only then fit one compact GRU with a static context branch, explicit masks and
elapsed-time channels.  Context spans are six and twelve calendar months; both
are fit with two predeclared seeds.  Use a direct signed anchor-delta output and
the same replay metric.  Normalization is fitted on each training prefix only.

The sequence branch is stopped after one justified repair if loss is unstable,
if controls show no history signal, or if it does not beat the equivalent tree
control across the frozen inner replays.  A second compact architecture is
conditional on that evidence; a large Transformer is out of scope without it.

## Tests, artifacts, and reporting

Required tests cover target blindness, changing hidden/future values not changing
earlier features, real calendar gaps, identical legal-state train/inference
features, unavailable-value exclusion from regional aggregates, prefix-only
normalization, sequence alignment/split boundaries, exact OOF alignment, and a
hand-calculated official metric.  Existing relevant tests must also pass.

Every material Kaggle run uses a clean committed/pushed SHA, unique run directory,
resolved config, package versions, dataset/schema fingerprints, flushed log,
exit status, elapsed time, and checksums.  It packages no raw challenge data.
Remote and local durability are recorded separately; a visible download action
is not evidence of a local artifact.

The decision report will give raw and official weighted RMSE, horizon and support
slices, model size/runtime, residual complementarity, block-aware uncertainty
limitations, failures, and the precise remaining work.  Any confirmation period
must be verified unused and evaluated once with the full recipe frozen.
