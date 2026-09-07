# Execution directive — local hydrological response and usable recent validation

Owner of Kaggle execution: existing task **Execute Kaggle rebuild workflow**. The lead task owns investigation and scientific direction. This directive supersedes the previous instruction to stop merely because no perfectly untouched h=1..7 scenario exists. No subagents; preserve the established local → GitHub branch → thin Kaggle runner workflow. Do not submit to Zindi.

## Confirmed investigation findings

1. `_scenario_training_rows` in `scripts/run_experiment.py` greedily selects non-overlapping complete 40-calendar-month Test-template windows. For **all three** evaluated cutoffs (September 2006, September 2007, January 2009), the only selected origin is **May 2002**. Thus R02/R03 see the same old sparse window, whereas R01 sees much more and newer training history. Their negative results are real for those particular recipes, but they do **not** isolate the effect of realistic observation masking. Correct the ledger's broader conclusion. Do not rerun unchanged R02/R03.
2. The December 2014–June 2015 block contains **109,349 h=1..7 rows and 90 h>7 rows**. The latter are **0.0822376%** of all rows. Its horizon counts are `{1:15637,2:15627,3:15607,4:15659,5:15638,6:15596,7:15585,8:46,9:21,10:8,11:5,12:5,13:5}`. The whole block need not be discarded. Define the h=1..7 slice using availability only, before model scores, and evaluate all candidates on identical IDs. Preserve/report the 90-row tail separately; never clip horizons or hide exclusions. Call this a **recent stress check**, not an untouched final test. Raw and horizon-weighted scores answer different questions and both must be shown.
3. R01/R04 changing the regression target does not test a fundamentally different response model. The next concrete hypothesis is that local hydrological response slopes differ by location and global trees underrepresent them. A regularized per-location model tests this directly. This is a hypothesis, not a promised improvement.

The lead inspected the notebook but executed no Kaggle cells. At inspection the draft session was off, the saved code/output was still pinned to `8c6edaf...`, and the local branch tip was `399dcf1...`. Re-check live state; do not assume prior artifacts survive.

## Files prepared locally by the lead

Three uncommitted/untracked prototype files now exist in the shared workspace:

- `src/local_response.py`
- `scripts/run_local_response.py`
- `tests/test_local_response.py`

The model standardizes the availability-safe features using training examples only, adds three simple horizon interactions, fits a global linear delta response, and estimates per-location coefficients with an L2 penalty toward that global prior. Alpha is fixed at **30** before seeing validation scores. Unknown locations fall back to the global model. The model serializes to a non-pickle NPZ file.

The runner compares five predeclared predictions: persistence, R01 LightGBM, global linear response, local response, and a fixed 50/50 local/LightGBM blend. It uses the same deterministic sampled training rows and weights for the learned candidates, LightGBM at fixed 173 rounds, and origins September 2007, January 2009, and December 2014 (the last with the declared h=1..7 slice). No early stopping or parameter selection on these outcomes. It generates no Test predictions.

The lead ran `python -m unittest discover -s tests -v`: **all nine tests passed**, including local slope recovery on synthetic opposite-response locations, unknown-location fallback, serialization identity, input permutation stability, and the existing simulator checks. No real-data model was fitted by the lead. Review the prototype yourself before running; passing a synthetic test is not full integration verification.

## Your immediate bounded task

1. Inspect `git status`, the three prototype files, and this directive. Preserve other untracked audit/prompt documents. Do not `git add .` or discard shared work. You now own edits to these prototype files; the lead will not concurrently edit them.
2. Review the runner for alignment, cutoff correctness, memory, and artifact completeness. Add the inexpensive production checks it still needs: expected clean commit, resolved configuration, fit parameters/seed/feature order, fold coverage/spec hashes, overall status, and a reliable flushed console log. Existing `_score` supplies per-fold hashes/slices; preserve them. Add the full recent block's persistence raw RMSE and the separate 90-row tail's count/error summary to the final report. Do not change the scientific comparison while doing plumbing.
3. Run syntax and unit checks locally. Commit only task-owned source/tests/directive and appropriate factual docs to the existing `codex/validation-rebuild` branch, then push normally. Do not merge main, force-push, or alter sharing.
4. Use the existing Kaggle notebook. Safely bootstrap/recover if needed; re-discover the mounted dataset; pin the new exact commit in both bootstrap and runner. Keep one selected-cell execution and streamed logs. Never use Run All or accidentally trigger Save & Run All.
5. Execute the new script through the thin runner, using a fresh unique output directory outside the clone:

   `python -u scripts/run_local_response.py --data-dir /kaggle/input/datasets/cashgenenator/drought --output-dir /kaggle/working/drought_runs/local_response_<unique_UTC_run_id> --rounds 173 --alpha 30`

   Use `sys.executable` in the notebook subprocess list. Review `--help` and the actual path first. Run only one experiment process at a time. Estimated several minutes, not hours; observe the actual timing. If it fails, repair the narrow cause and use a new run ID.
6. Persist per-fold model files, OOF predictions, manifest, metrics, and logs. Print a compact result table into the notebook. Package outputs without raw challenge data. Try the supported Kaggle Output download UI rather than repeatedly clicking the same ineffective FileLink. Verify durability rather than claiming it. At minimum preserve compact metrics/manifest text locally from visible output, even if large package transfer remains unavailable; do not invent a filesystem path for a browser download.
7. Return the three-origin table for all five candidates, by-horizon errors for the recent block, training row counts, any runtime/memory failures, artifact paths/checksums, exact commit, and a factual ledger update. Separate hypothesis outcomes: does global linear work, does localization help relative to it, does either beat R01, and does the fixed blend add value? Do not describe the best of these as independently selected/tested; all are now development/stress evidence.

## Decision rule and next assignments

Do not retune alpha after seeing this recent block and relabel it untouched. A useful initial signal is local response or its fixed blend beating R01 by at least .003 absolute RMSE on the recent stress check, with supporting improvement on an older replay and no severe horizon/region regression. This is a practical screen, not statistical significance. A local model may help by complementarity even when worse standalone; report both.

If the local branch fails, that is a clean informative result. Stop this bounded branch and return the artifacts to the lead for the next assignment; do not consume more runs searching arbitrary ridge penalties. If it succeeds, preserve the frozen recipe and report; the lead will choose a second recent origin and whether to test a causal local-residual booster or further regional structure. Do not generate a submission or run a parameter grid yet.

The separate future task is a **matched-data masking test**: replace the single earliest 40-month training replay with rolling shorter historical observation blocks that retain recent data, cap/normalize repeated target contributions, and compare with a legacy sampler constrained to matched source months and location/label coverage. That task requires a training-coverage table before fitting. Do not conflate it with the local-response experiment or claim the old R02 test already answered it.

Run the immediate task and return measured results. Do not terminate at another plan or because the historical periods have previously been inspected; the role of these runs is explicit development and stress diagnosis.
