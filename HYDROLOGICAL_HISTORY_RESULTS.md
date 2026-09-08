# Hydrological history results

**Status:** Stage A availability audit completed; no Stage B model fit has been
run.
**Latest trajectory-code SHA:** `1810664d9145c07e2ebc7497555a5da63522a3a2`
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

## Implemented, not yet measured

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

Local verification at `5f7274c`: `python -m pytest -q` passed **29 tests**;
syntax compilation passed for both runners and the new feature module. On the
executing Kaggle environment, `tests/test_hydro_trajectory.py` passed **3/3**
before Stage A. This is implementation verification, not model-performance
evidence.

## Unanswered questions

| Question | Status |
|---|---|
| Did Stage A reveal a feature-contract defect? | No: all 56 B0 features are available in Test; material temporal distribution shift remains. |
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
