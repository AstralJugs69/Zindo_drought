# Next action — compact neural sequence comparison closed

## 2026-09-08 — decision checkpoint

- The availability-faithful MLP/GRU/TCN comparison is complete. Inner selection chose GRU, 12 calendar slots, epoch 1 (mean raw RMSE 0.602960 across two frozen inner origins and two seeds). A third seed averaged 0.607401; it did not alter the frozen selection or rescue the result.
- The outer two-seed GRU is worse than B3/98 on 2007-09, 2009-01, and 2014-04; its all-outer raw delta is +0.018143. It helps only at 2014-12 (-0.005213), which is inadequate for promotion. The predeclared equal blend is also adverse overall (+0.002334). Retain **B3/98** as strongest verified development baseline; do not generate Test predictions or a submission from any neural candidate.
- The inner history ablation confirms older lawful sequence slots carry some signal (+0.005503 RMSE when hidden), but not enough to beat B3. This is a direction decision, not a claim that history is useless.
- The outer runner had a documented cgroup OOM on its final origin. Valid per-fit OOFs were retained and only 2014-12 was resumed after a sequential-memory repair. The paired recovered analysis, stability check, and ablation completed remotely. See `HYDROLOGICAL_HISTORY_RESULTS.md` for exact paths, hashes, and limits.
- Source commit for the metric follow-up analyzer: `c00687d4ae9506f37090fa85d413de3d8d02ad26`; `python -m pytest -q` passed 43 tests. Remote outputs remain under `/kaggle/working/drought_runs/`; do not download them merely for inspection.
- No neural follow-up is justified without a new independently motivated, availability-faithful hypothesis. A future confirmation must first verify a genuinely unused period and freeze its recipe before one evaluation.

# Earlier checkpoint — availability-faithful neural sequence comparison

## 2026-09-08 — Stage C direct-history controls (superseded as a neural gate)

- The completed Stage B runs used clean, pinned source
  `545fe9963fe6388eb89e065b9f39afcc233ba081` on `codex/validation-rebuild`.
  Do not reset the branch: the earlier Kaggle-only commits are intentional.
- The four B0 references and B1/B2 trajectory ablations completed on Kaggle
  without Test predictions. B1 and B2 improved raw RMSE in the favorable
  direction on every declared development/stress replay. The full metrics, run
  paths, checksums, memory, and statistical limits are in
  `HYDROLOGICAL_HISTORY_RESULTS.md`.
- B3's full outer ablation completed. It improved B0 on every replay, although
  B1 is the best single variant at 2009-01 and B2 at 2014-04. The remaining B3
  package is 13,993,051 bytes with SHA-256
  `3ba286a2e60221e906e26ea58772e716cf8a786e768397a83d8d71a277f36318`; it
  recorded `no_test_predictions=true`, a finite-or-missing 446-column Test
  contract, and no Test prediction.
- The frozen 2003-04/2004-04 inner selection completed from clean commit
  `c5591b428ea24bc960ddb172079de70fda957d16`. B3/98 is selected by the
  predeclared mean weighted-inner rule (0.568483) over B2/98 (0.568508), but the
  0.000025 margin is explicitly too small to call a practical B3-over-B2
  advantage. The full table and checksums are in
  `HYDROLOGICAL_HISTORY_RESULTS.md`.
- Stage C run `hydro_sequence_controls_20260908T120000Z` reproduced B3/98,
  then found both six-/twelve-month flattened history trees worse on both
  frozen inner replays; fixed prefix-normalized ridge controls were much worse.
  Its 78,524,335-byte remote archive hashes to
  `18ae60a07b4569dd341fbb5086555e3906146b065fe585322779be824f0ae25d` and
  records `no_test_predictions=true`.
- **Historical control result:** the compact flattened-tree/ridge controls did
  not add useful direct local-history signal beyond B3. **Neural sequence
  modelling was not tested** by that run. Its former no-GRU stop rule is
  superseded by the separately predeclared availability-faithful MLP/GRU/TCN
  comparison in `NEURAL_SEQUENCE_EXPERIMENT_PLAN.md`; that comparison retains
  the failed control result and does not reinterpret it as a neural negative.
  Do not generate Test predictions or a competition submission.

# Next action — local-response stress result

## 2026-09-08 — verified rerun at provenance-hardened commit

- Kaggle rerun `local_response_20260907T220239Z` completed in **355.293 s** from
  full commit `60376c0ac9ce9438f1c3e05ed7b8612cc73f33c3` on
  `codex/validation-rebuild`, using seed `20260907`, 173 rounds, and alpha=30.
  The result is deterministic and matches the table below; this is the current
  reproducibility anchor (the earlier 34b7e3b run remains historical).
- The no-raw-data archive is
  `/kaggle/working/drought_runs/local_response_20260907T220239Z.zip`,
  92,481,289 bytes, SHA-256
  `6ce1bd4b98ffa6e7f84422d66b5e09869e1e181c342924926a93d6adea2d8c52`.
  Kaggle's supported download action was invoked twice, including a retry after
  reopening the notebook; no matching local Downloads file was observed, so
  durability remains **unverified/session-local** and no repository copy could
  be made.

- A separate, earlier verified archive was subsequently downloaded and copied
  into the local repository at
  `artifacts/local_response_20260907T212420Z.zip`. It is 92,480,263 bytes with
  SHA-256
  `7a45c3a919b6a19f0defb58218643b0f912813c15f92d6bb0a28fbdf4bfde830`.
  The file is locally durable but remains Git-ignored under `artifacts/` and
  was not committed or pushed. This does not change the newer 220239Z run's
  session-local durability status.

- Branch: `codex/validation-rebuild`. The completed local-response Kaggle run
  used the clean, pinned source commit
  `34b7e3b132a463fb7d6cdd936ee188bc4f8b049d`; the local branch subsequently
  advanced to `d856959` with provenance-only runner hardening and was not
  re-run.
- Kaggle run: `local_response_20260907T212420Z`, completed in **358.909 s**.
  It used the fixed seed `20260907`, 173 LightGBM rounds, alpha=30, the same
  deterministic sampled training rows for all learned candidates, and no
  early stopping or parameter selection. The old submission-training cell,
  Run All, Save Version, Test prediction generation, and Zindi submission were
  not used.
- Training/validation rows by origin were `851,033 / 278,447` (Sep-2007),
  `1,100,055 / 273,200` (Jan-2009), and `1,827,255 / 109,349`
  (Dec-2014 h=1..7). The December block is a declared **recent stress check**,
  not an untouched confirmation test.

| Candidate | Sep-2007 raw RMSE | Jan-2009 raw RMSE | Dec-2014 raw RMSE | Decision signal |
|---|---:|---:|---:|---|
| Persistence | 0.625852 | 0.711862 | 0.861437 | reference only |
| R01 LightGBM | 0.537422 | 0.574104 | 0.822625 | incumbent comparison |
| Global linear response | 0.540809 | 0.587308 | **0.807727** | recent-stress gain vs R01, but regressions on both older replays |
| Local response, alpha=30 | 0.553143 | 0.585955 | 0.843479 | localization does not help relative to global linear or R01 |
| Fixed 50/50 local/R01 blend | **0.531417** | **0.567633** | 0.823734 | improves both older replays but is 0.001109 worse than R01 on recent stress |

- Result: do **not** promote the standalone local model or fixed blend. The
  blend clears the older-replay portion of the practical screen but fails its
  recent-stress requirement; the global model is promising only as an
  unselected development hypothesis, not a promoted recipe. No alpha or blend
  weight was retuned after these results.
- Recent h=1..7 slice: 109,349 rows. Its separate h>7 tail is 90 rows
  (h8=46, h9=21, h10=8, h11=5, h12=5, h13=5); tail persistence raw RMSE is
  1.228637, MAE 0.914803, bias -0.537166. Full-block persistence raw RMSE is
  0.861803 over 109,439 rows. Per-horizon R01/global/local/blend metrics and
  OOF files are in the artifact manifest/metrics rather than collapsed into a
  headline score.
- Persisted live artifacts: `/kaggle/working/drought_runs/local_response_20260907T212420Z/`
  contains config, manifest, metrics, flushed console log, three LightGBM
  models, three NPZ local models, fifteen OOF files, and the h>7 tail. The
  packaged output is `/kaggle/working/drought_runs/local_response_20260907T212420Z.zip`
  (92,480,263 bytes; SHA-256
  `7a45c3a919b6a19f0defb58218643b0f912813c15f92d6bb0a28fbdf4bfde830`).
  This is the archive copied to the local repository as noted above; the
  Kaggle source directory itself remains session-local.
- Ledger correction: R02/R03's observed regressions remain real for their
  exact recipes, but their schedule-training windows all collapsed to the same
  May-2002 origin. They do not isolate realistic observation masking and must
  not be treated as a general rejection of masking-aware training. Do not rerun
  those unchanged experiments.
- Next bounded assignment: design the predeclared matched-data masking test
  with a training-coverage table before fitting. Keep the local-response branch
  frozen unless a separately authorized experiment tests a new causal
  formulation; do not tune alpha, blend weight, or generate a submission.
# 2026-09-08 Spatial Phase A checkpoint (corrected)

The earlier pooled Phase A negative conclusion is **retracted**. It used a
bugged denominator that summed repeated per-rank/per-date supported SSE. The
repaired Kaggle run used commit `0f406fa058e0d7553a8c11796a365edd22e612cb` at
`/kaggle/working/drought_runs/spatial_phase_a_repaired_20260908T013000Z`.
It computes the finite OOF residual SSE once per origin, including unsupported
rows, and reports requested/actual basis rank separately. Synthetic regression
tests cover denominator invariance and rank provenance.

Corrected all-finite opportunity fractions (ranks 4/8/16/32) are: Sep-2007
`0.07212/0.11801/0.21368/0.33246`; Jan-2009
`0.06167/0.13130/0.21886/0.31247`; Dec-2014 h1-7
`0.11902/0.20431/0.29569/0.40116`. Supported and all-finite fractions now
match because the basis covers every scored location in these folds. These are
`ORACLE_DIAGNOSTIC` hindsight residual projections, not deployable forecast
skill. The spatial forecast stage and independent regional stage remain
eligible; no Test predictions were produced.

The repaired Phase A artifact remains Kaggle-session-local and was not
downloaded.

# 2026-09-08 Regional hydrology smoke checkpoint

The paired Kaggle smoke run completed successfully from commit
`3a8cb5b96f243d6b4c88eb9f0cb10081aac2e13c` at
`/kaggle/working/drought_runs/regional_capacity_20260908T021000Z`. It used the
same sampled rows, labels, weights and frozen 173-round LightGBM capacity for
C0 (R01 features) and C1 (5-degree contemporaneous regional means, local-minus-
regional deviations and presence flags). Raw/weighted RMSE was C0/C1:
Sep-2007 `0.537422/0.530604` (`0.537405/0.530589` weighted), Jan-2009
`0.574104/0.563443` (`0.573954/0.563318`), and Dec-2014 h1-7
`0.822625/0.818479` (`0.803957/0.801413`). This is encouraging paired smoke
evidence, not completion of the prescribed stage: width-15 features, inner
chronological capacity selection, full OOF/model artifact persistence, and the
additional recent origin remain pending. The run's regional hashes were shape
placeholders and are not treated as provenance; the runner has since been
repaired to hash feature bytes. No Test predictions or submissions were made.

The expanded paired width run completed both 5- and 15-degree blocks
(`regional_widths_20260908T031000Z`, commit
`fdee96573583f22cdaf4d17f2b9d58de9bb9a219`). Width-15 C1 raw RMSE was
`0.529463/0.559609/0.818389` for Sep-2007, Jan-2009 and Dec-2014 h1-7,
respectively, versus paired C0 `0.537422/0.574104/0.822625`. These remain
frozen-capacity information comparisons; inner capacity selection and transfer
evaluation are not yet complete.
