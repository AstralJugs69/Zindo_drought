# Next action — validation rebuild

- Branch/commit: `codex/validation-rebuild` / `f02ffa4281c438f9e66f310e03798de14876c3da`.
- Kaggle session: running, verified checkout at that commit; notebook draft has safe
  bootstrap, schema discovery, and the thin runner set to R00.
- Completed Kaggle artifacts: `preflight_20260907T140625Z` and
  `preflight_20260907T140714Z` under `/kaggle/working/drought_runs/`.
- R00 observation: the initial December-to-June masked stress block reached true
  horizons through 20 because historical source months are absent. It reported raw
  RMSE only and is not eligible for weighted promotion scoring.
- Failed run: `preflight_20260907T141306Z` correctly stopped because the streaming
  Test-template replay did not reproduce the legacy transplanted horizons. The
  cause is incomplete per-location historical source-row coverage, not a model error.
- Next command after synchronizing the next commit: rerun `scripts/run_experiment.py`
  with `--mode r00`; it now freezes a complete-coverage location set per scenario,
  records excluded locations/rows, and asserts that the stream-derived horizons
  equal the Test template before scoring.
