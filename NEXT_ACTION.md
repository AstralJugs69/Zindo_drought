# Next action — Stage B hydrological-history reference

## 2026-09-08 — Stage A completed; start B0 reference only

- The repaired Kaggle checkout is writable and is clean at
  `1810664d9145c07e2ebc7497555a5da63522a3a2` while this checkpoint is written.
  Do not reset it: the preceding three Kaggle-only commits are intentional.
- Stage A run `history_availability_audit_20260908T055537Z` completed in
  209.57 s. All 56 B0 features are available in Test, and no Test horizon,
  season, or five-degree cell is outside historical support. Broad and matched
  spatial domain AUCs are 0.936608 and 0.900016. Treat the latter as a
  temporal-shift warning, not as a reason to discard the causal B0 protocol.
- The compact remote bundle is
  `/kaggle/working/drought_runs/history_availability_audit_20260908T055537Z/audit_artifacts.zip`
  (SHA-256 `7f20174f4894a45c8ee1afd856f43efd6b6b08a24616a1c66e3c25a8d65d2788`);
  its checksum-matched local Git-ignored copy is
  `artifacts/history_availability_audit_20260908T055537Z.zip`.
- **Next single run:** establish the missing outer C1/98-capacity reference as
  B0 at one origin before any B1/B2/B3 candidate. It produces OOF diagnostics,
  feature-schema parity checks, and model artifacts only—never Test predictions
  or a submission.

  ```bash
  cd /kaggle/working/Zindo_drought
  git status --short --branch
  expected_commit=$(git rev-parse HEAD)

  python -u scripts/run_hydrological_trajectory.py \
    --data-dir /kaggle/input/datasets/cashgenenator/drought \
    --output-dir /kaggle/working/drought_runs/hydro_trajectory_b0_pilot_<UTC> \
    --expected-commit "$expected_commit" \
    --origins 2007-09 --candidates B0 --rounds 98
  ```

- Inspect source-map construction, elapsed time, peak memory, Test feature
  contract, OOF/model count, manifest and log. If the pilot finishes cleanly,
  run fixed 98-round B0 across the remaining declared origins before evaluating
  B1/B2/B3. Do not reinterpret the inherited C1/59 metrics as C1/98 metrics.

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
