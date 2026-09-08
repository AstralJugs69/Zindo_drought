# Hydrological history results

**Status:** execution checkpoint — no new Stage A/B model fit has been run.  
**Latest implementation SHA:** `5f7274c6999d68917e2a982d5ca2b8908be33aca`  
**Reason:** the verified Kaggle SSH account can read but cannot write the
root-owned checkout or `/kaggle/working/drought_runs`; see `NEXT_ACTION.md`.

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

## Implemented, not yet measured

- `scripts/run_history_availability_audit.py` implements the Stage A target-blind
  B0-vs-Test availability/distribution audit, including broad and matched
  horizon/season/5-degree-cell domain diagnostics with spatial grouped folds.
- `src/hydro_trajectory.py` constructs causal local 3/6-calendar-month
  covariate summaries and regional trajectory/disagreement features. It has no
  TWS or target input.
- `scripts/run_hydrological_trajectory.py` implements the fixed-capacity paired
  B0, B1-local, B2-regional, B3-both ablation. It validates Test feature-schema
  parity but never predicts Test targets or creates a submission.
- `WINNER_METHOD_AUDIT.md` records source/code evidence and transfer limits.

Local verification at `5f7274c`: `python -m pytest -q` passed **29 tests**;
syntax compilation passed for both runners and the new feature module. This is
implementation verification, not model-performance evidence.

## Unanswered questions

| Question | Status |
|---|---|
| Did Stage A reveal a feature-contract defect? | Not yet executed. |
| Do local trajectories help B0? | Not yet measured. |
| Do regional trajectories add beyond contemporaneous C1? | Not yet measured. |
| Does C1/98 transfer to outer replays? | Not yet measured. |
| Does a sequence model add beyond equivalent history inputs? | Not started; correctly gated behind Stage B. |
| Is any new candidate independently confirmed? | No. |

## Statistical scope

The 2007-09, 2009-01, 2014-04, and 2014-12 h=1..7 periods have already
influenced development choices. They are not untouched confirmation evidence.
The two inner capacity replays overlap in five target calendar months and are
selection evidence only. Any later confirmation requires an independently
verified unused period and a recipe frozen before one evaluation.
