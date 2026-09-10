# Next action — B3/98 full submission result recorded

## 2026-09-10 — submission-compliant B3-C rebuild completed; GLDAS gate blocked

The coordinate-free B3-C reference was rebuilt on the persistent `zindi-gcp`
VM from source commit `958060ffca757c50c936d678bdaaca2533bae780` in tmux
session `zindi` (12 LightGBM threads, frozen seed `20260908`, 98 rounds,
63 leaves, `min_data_in_leaf=1000`).  The historical 446-column B3 builder is
preserved for audit only; the new fitted model contract is an explicit ordered
444-column allowlist that removes raw `lat` and `lon`.  Coordinates remain
metadata-only for causal source indexing and hydrological regional/local
aggregation.  No cell ID, coordinate encoding, spatial embedding, or learned
location input is present in B3-C.  The allowlist and feature provenance are
saved in `b3c_feature_audit.json` for every run review.

The four required B3-C references completed (remote artifacts remain remote):

| Origin | h1--7 raw RMSE | Weighted proxy |
|---|---:|---:|
| 2003-04 | 0.585218 | 0.585130 |
| 2004-04 | 0.563164 | 0.563051 |
| 2014-04 | 0.588990 | n/a (h=4 absent) |
| 2014-12 | 0.803770 | 0.785391 |

Run path:
`/home/milli/zindi_drought_gcp/drought_runs/gcp_submission_compliant_b3c_20260910_v1/`.
It contains four saved B3-C models, four OOF ledgers, manifests, prefit
causality/schema checks, timing, and no Test predictions or submissions.  The
`/usr/bin/time -v` record reports 16:01.70 wall time, 26,356,236 kB maximum
resident set, and exit status 0.  The optional in-process `psutil` sampler was
not installed, so the external time record is the authoritative RAM evidence.

The one permitted external intervention was not run.  An official
GLDAS-2.1 Noah monthly 0.25-degree sample URL was checked: discovery/HEAD was
available, but the first data GET without credentials returned HTTP 401.  No Earthdata token,
`.netrc`, username, or credential file was present, no bulk acquisition began,
and no substitute product was used.  The ten-feature GLDAS contract and exact
causal/missingness rules are recorded in `external_access.json`; its status is
`blocked_http_401_no_noninteractive_earthdata_credentials`.

**Decision:** B3-C is the current submission-compliant reference; the
historical coordinate-bearing B3 is explicitly unsuitable for a compliant
submission.  The paired GLDAS inner gate is not evaluable while official data
access is blocked, recent external candidates and optional 2009-01 were not
run, and no further experiment is authorized in this sequence.  If a valid
Earthdata credential is later supplied, re-run the single sample/access gate
first rather than changing the feature contract or silently substituting a
dataset.

## Current runtime state — 2026-09-10

Execution has moved to the persistent CPU-only VM reached with `ssh zindi-gcp`.
Use `/home/milli/zindi_drought_gcp` on `codex/validation-rebuild`; the capacity
run was launched from source commit `a44bf70`.  Run long jobs only in tmux session
`zindi` and keep its `script -af /home/milli/zindi-session.log` logger alive.
The authoritative Train/Test/SampleSubmission files are hash-matched on the VM,
and surviving fold archives are recorded in `RECOVERY_MATRIX_20260910.md`.
The exact full B3 model, Dec-2014 D0 replay package, and matched A/B/C directory
were rebuilt on GCP and remain remote for inspection; do not copy the large
artifacts to Windows.  See `GCP_MIGRATION_RUNBOOK.md` for the environment and
resume gate.

The current matched replay is complete.  On the same 109,439-row legal replay,
the late D0 model (A) scores `0.789549` Test-horizon-weighted validation proxy, while
the full B3 model under the sparse legal view (B) scores `0.737579`; the
retrospective dense view (C) is exactly identical to B.  A reproduces its
saved OOF within `4.44e-16`, exposure counts are recorded, and corrected cell
SSE aggregation is conserved to `1.82e-12`.  The GCP booster is an environment-
drift rebuild (LightGBM 4.7.0), not a byte-for-byte replacement for the
historical Kaggle booster; row/label/weight hashes match the frozen contract.
The B/C result is an empty no-withholding control (`sparse_withheld_window_rows=0`,
zero changed features), not evidence that sparse covariates cannot matter.
No Test labels, predictions, calibration, or submission were created.

The old undefined recency-weighted follow-up is closed.  The fixed 98-vs-392
capacity comparison is now complete; see `CAPACITY_BOTTLENECK_RESULTS.md` for
the gated result and remote artifact path.

## 2026-09-10 — capacity bottleneck decision closed

The Train-only B3 comparison ran on `zindi-gcp` in tmux at commit
`a44bf7053b187133f5d26aea4b44fe03aef13ef4`, with the unchanged 446-feature,
63-leaf, `min_data_in_leaf=1000` recipe fit once to 392 rounds and scored
checkpoints 98 and 392 on identical paired rows.  At 2014-04, raw h1--7 RMSE
improved `0.581798 -> 0.567687` (h=4 is absent and was not imputed).  At the
complete 2014-12 origin it worsened `0.809378 -> 0.814031`; the complete
Test-horizon-weighted validation proxy worsened `0.789549 -> 0.791794`.  No recent month with at least 1,000
rows exceeded the `+0.02` regression guardrail, but both origins had to
improve, so `recent_gate_pass=false` and the optional 2009-01 transfer check
was not run.

**Decision:** close extra boosting rounds as a standalone remedy and retain
B3/98 as the development control.  Continue with representation/data-transfer
diagnostics only if a separately authorized hypothesis is supplied; do not
launch a larger capacity grid or create Test predictions/submissions.

Future GCP LightGBM work now uses a 12-thread baseline, with 24 or 48 threads
available only for an explicitly authorized isolated benchmark after checking
RSS headroom.  The capacity run itself remains the pre-change four-thread
measurement because it was not interrupted.

## 2026-09-10 — late-2015 spatial/reversal diagnostic

The Train-only diagnostic is complete on `zindi-gcp` in
`/home/milli/zindi_drought_gcp/drought_runs/gcp_late2015_spatial_reversal_20260910_v10`
from commit `c3575fb119751ef0925d748081b82ec743684efa`.  It read no Test data,
fit no model, and wrote no predictions or submission.  The exact-calendar
target identity holds on 1,977,398 rows with zero difference; 176,623
terminal/missing-next-month rows were excluded rather than bridged.

The failure is spatially coherent and temporally reversing: `d_t` RMSE is
1.012/1.189/1.182 in 2015-01/02/06 versus 0.504/0.524/0.542 in matched early
controls, while nearest-neighbour correlations remain 0.991/0.987/0.997 and
5-degree cell means explain 83.0%/69.1%/92.7% of d SSE.  Same-location
calendar controls have 1.130–1.426, 1.339–1.371, and 1.366–1.427 difference
RMSE for January/February/June.  Reversal becomes stronger in the suspicious
months, and historical-q90+ deltas account for 71–73% of Dec-replay error SSE.

The frozen 446-feature B3 schema contains focal `last_observed_TWS` and `h`,
300 hydro-only regional features, and no neighboring TWS state/date.  Under the
existing mask-block replay ledger, legal source-date neighbor state has 99.91%
support, median eight neighbors, and differs from the focal anchor on ~100% of
supported events.  This is a non-empty candidate, not evidence that the prior
zero-withholding B/C control was informative.

The specified intervention was then run exactly once as the gated
`NEIGHBOR_STATE_B3_RESULTS.md` experiment.  It added the legal eight-neighbour
mean, mean-minus-focal residual, support count, and median neighbor-anchor age
to frozen B3/98.  The two-origin inner screen passed (`+0.000680` and
`+0.002900` raw h1--7 RMSE gains), but the recent gate failed: 2014-04
regressed by `0.001061` while 2014-12 improved by `0.002575` (weighted proxy
`0.789549 -> 0.788071`).  April has no h=4 support and was evaluated with
present h1--7 rows; no weighted proxy was invented.  No source month with at
least 1,000 rows exceeded the `+0.02` regression guardrail, but both recent
origins had to improve, so the optional 2009-01 transfer check was not run.

**Decision:** reject the neighbor-state augmentation as a replacement and
retain dense B3/98.  Keep the remote models/OOF and prefit checks as audit
evidence; do not sweep additional neighbor features or create Test predictions
or a submission without separate authorization.  Full paths, hashes,
coverage, resource measurements, and per-horizon/source-month tables are in
`NEIGHBOR_STATE_B3_RESULTS.md`.

The diagnostic ran in 144.07 s at the 12-thread baseline, with 1.87 GiB peak
RSS and no swap.  The fixed cached inference benchmark supports explicit 24/48
thread escalation (median 0.1817/0.1201/0.0772 s at 12/24/48) but leaves 12 as
the safe default and forbids concurrent high-thread jobs.

## 2026-09-09 — leaderboard root-cause investigation complete

The read-only investigation is recorded in
`LEADERBOARD_ROOT_CAUSE_REPORT.md`. It did not fit a model, create a Test
prediction file, or upload anything. The saved B3/98 submission is internally
reproducible: Train/Test/SampleSubmission hashes and ID order match, the exact
next-calendar target join has 1,977,398 zero-difference matches (176,623
terminal rows are explicitly unmatched), and a deterministic saved-model Test
sample matches the submitted CSV to `4.44e-16`. Batch/shuffle order, masked-TWS
perturbation, and later-covariate causality checks all pass.

The measured cause of the public `0.750207364` gap is now ranked as late-period
target-process/transfer shift first, with historical support mismatch (sparse
Test months and increasing legal-anchor age) as a contributor. Train target
behaviour is materially harder in 2015 (target-current RMSE 1.012/1.189/1.182
for Jan/Feb/Jun versus 0.433--0.615 in matched earlier months), and the change
is dispersed across hundreds of 5-degree cells. Test hydro covariates are
finite and within Train ranges, so an export defect or simple numeric
extrapolation is not supported. Existing outer OOF remains development
evidence: pooled h1--7 weighted B3/persistence is `0.583339/0.689653`, while
the late Dec-2014 fold is `0.789549` weighted (`0.809688` raw).

**Historical proposal (closed):** the late-prefix, Test-schedule-matched
recency-weighted B3/98 fit described in the original 2026-09-09 note is not an
active next step.  July/August 2015 was already used as development evidence;
the bounded capacity comparison above replaced it and failed its two-origin
gate.  Do not run that recency proposal, a larger capacity grid, or create a
Test prediction/submission from this branch.

## 2026-09-09 — frozen full-training deployment

- The one authorized full-training deployment run completed on Kaggle from
  commit `8fa8956e3a27da544f5954292ea5ff6bc44c9c22` using the frozen dense
  B3/98 recipe. It fit exactly once (`fit_count=1`) on 1,976,942 legal sampled
  rows; the 446-column Test contract and exact replay integration check passed.
- The generated file is
  `artifacts/b3_full_submission_20260909T010000Z/submission_b3_dense_history_98_5bc9e52.csv`,
  with 280,961 rows and SHA-256
  `62f1876ee7a26dfc5289c68d724353e9b89185d146eb7e8335a827e923424e7c`.
  Only this CSV and small manifests were retrieved locally.  The large Kaggle
  artifacts were not independently recovered after the runtime loss; the
  historical path is retained as lost provenance, not as a live `ssh kaggle`
  source.
- The authenticated Zindi account currently shows two successful entries with
  this same filename: IDs `1JLvpsYJ` (2026-09-09 04:13:32.148Z) and
  `38KHQMrj` (2026-09-09 04:13:54.579Z). Both scored `0.750207364`; the
  leaderboard shows rank 230 for `astraljugs` with four total submissions.
  This duplicate platform state is recorded as observed; no additional upload
  was attempted.

## Safe continuation

1. Treat the B3/98 public score as a completed leaderboard observation, not as
   independent confirmation; do not fit another candidate or submit another
   file under this campaign.
2. Keep `/kaggle/working/drought_runs/b3_full_submission_20260909T010000Z/`
   as the historical audit path only; its post-restart contents are lost unless
   independently recovered.  Use `ssh zindi-gcp` for current artifacts and do
   not copy the model or large ledgers to Windows.
3. Do not reconnect to Kaggle for this campaign; current work stays on the
   persistent `zindi-gcp` VM in tmux session `zindi`.

# Next action — covariate-gap decision closed

## 2026-09-09 — final augmentation checkpoint

- The deterministic Test-shaped covariate-history gap experiment is complete on
  Kaggle. It used only Test source-date geometry to derive masks; no Test labels,
  predictions, or submission files were created, and no run artifacts were
  copied to Windows.
- The predeclared inner screen compared `D0_dense`, `D1_sparse`, and
  `D2_mixed` on 2003-04 and 2004-04. Both augmented recipes improved the
  seed-mean official h=1..7 weighted RMSE on both inner origins. `D1_sparse`
  won the equal-origin mean (0.566690 versus D2's 0.566941) and was therefore
  the only recipe taken to the four outer replays with seeds 20260909/10.
- Outer D1 improved 2007-09 by 0.006003 and 2009-01 by 0.004169, but regressed
  complete 2014-12 by 0.003792. 2014-04 has no h=4 rows, so its official gate
  is incomplete; the present-horizon diagnostic worsened by 0.006348. The
  analyzer records `official_gate_complete=false`, blocker `2014-04`, and
  `promote=false`.
- **Decision:** reject `D1_sparse` as a deployment/training replacement and
  retain dense **B3/98**. Keep the augmentation code and remote artifacts as
  audit evidence; do not tune this recipe further and do not generate Test
  predictions or a competition submission.
- Source and artifacts are pinned in `HYDROLOGICAL_HISTORY_RESULTS.md`.
  The final remote checkout is clean at
  `8154633d450ed9ca76b038f7c72b23ccc0cef621`; analysis JSONs remain under
  `/kaggle/working/drought_runs/`; those post-restart artifacts were not
  independently recovered and the path is retained as lost provenance.

## Safe continuation

1. Treat B3/98 as the current strongest verified development baseline while
   the bounded capacity diagnostic is evaluated.
2. If work resumes after that diagnostic, first identify and verify a genuinely
   unused confirmation period, freeze any independently motivated hypothesis
   and metric gate, and evaluate once on `zindi-gcp`. Do not use the four
   replay origins above as independent confirmation.
3. Do not treat the lost Kaggle runtime as a current artifact source; current
   work stays on the persistent `zindi-gcp` VM and its `zindi` tmux session.

# Next action — compact neural sequence comparison closed

## 2026-09-08 — decision checkpoint

- The availability-faithful MLP/GRU/TCN comparison is complete. Inner selection chose GRU, 12 calendar slots, epoch 1 (mean raw RMSE 0.602960 across two frozen inner origins and two seeds). A third seed averaged 0.607401; it did not alter the frozen selection or rescue the result.
- The outer two-seed GRU is worse than B3/98 on 2007-09, 2009-01, and 2014-04; its all-outer raw delta is +0.018143. It helps only at 2014-12 (-0.005213), which is inadequate for promotion. The predeclared equal blend is also adverse overall (+0.002334). Retain **B3/98** as strongest verified development baseline; do not generate Test predictions or a submission from any neural candidate.
- The inner history ablation confirms older lawful sequence slots carry some signal (+0.005503 RMSE when hidden), but not enough to beat B3. This is a direction decision, not a claim that history is useless.
- The outer runner had a documented cgroup OOM on its final origin. Valid per-fit OOFs were retained and only 2014-12 was resumed after a sequential-memory repair. The paired recovered analysis, stability check, and ablation completed remotely. See `HYDROLOGICAL_HISTORY_RESULTS.md` for exact paths, hashes, and limits.
- Source commit for the metric follow-up analyzer: `c00687d4ae9506f37090fa85d413de3d8d02ad26`; `python -m pytest -q` passed 43 tests. The historical outputs under `/kaggle/working/drought_runs/` were not independently recovered after restart; do not treat them as live or download them merely for inspection.
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
