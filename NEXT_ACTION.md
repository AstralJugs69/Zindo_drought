# Next action — validation rebuild

- Branch: `codex/validation-rebuild`; latest pushed validation-replay commit:
  `39d4fd7e6302f03ebc0160fcb127c4b837019cd1`. The local worktree now contains
  the next task-scoped R01--R03 runner change, not yet committed.
- Kaggle: running Draft Session. The notebook has only the safe idempotent
  bootstrap, verified dataset discovery, the thin named-runner cell, and one
  short artifact-packaging cell. The old submission-training cell was not run;
  Run All and Save Version were not used.
- Completed, verified Kaggle runs:
  - `preflight_20260907T140625Z`: non-training environment/input preflight.
  - `preflight_20260907T140714Z`: raw-only R00 stress block; h reached 13.
  - `preflight_20260907T141306Z`: failed closed on a template-horizon mismatch.
  - `preflight_20260907T143613Z`: corrected R00 complete-coverage replay.
- R00 corrected weighted persistence RMSE: Jan-2009 exact replay `0.711328`,
  Sep-2006 `0.638469`, Sep-2007 `0.625816`. The Dec-2014--Jun-2015 h=1..13
  confirmation block is raw-only `0.861803`; it is not promotion eligible.
- Last completed package (session-local until a verified download succeeds):
  `/kaggle/working/drought_runs/preflight_20260907T143613Z.zip`, 14,834,676
  bytes, SHA-256 `8d37c97dd6d5094ceb15410b8976e0004032a2261fa7f57f26d9a4abca21a867`.
- Current incumbent: R00 persistence only under corrected scenarios. No trained
  candidate is promoted; EXP005/EXP010 public scores remain historical records.
- Next command after committing/pushing the local R01--R03 implementation and
  updating `EXPECTED_HEAD` in the safe bootstrap/runner cell: run R01 with
  `python -u scripts/run_experiment.py --data-dir "$DATA_DIR" --output-dir
  /kaggle/working/drought_runs/<unique-run-id> --run-id <unique-run-id> --mode
  r01 --rounds 173 --seed 20260907`, then inspect/package before R02.
