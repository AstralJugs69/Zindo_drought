# Next action — validation rebuild

- Branch/commit: `codex/validation-rebuild` / `b3bad826113e7a47350e798ed4dbda197d1bea02`.
- Kaggle session: running, verified checkout at that commit; notebook draft has safe
  bootstrap, schema discovery, and the thin runner set to R00.
- Completed Kaggle artifacts: `preflight_20260907T140625Z` and
  `preflight_20260907T140714Z` under `/kaggle/working/drought_runs/`.
- R00 observation: the initial December-to-June masked stress block reached true
  horizons through 20 because historical source months are absent. It reported raw
  RMSE only and is not eligible for weighted promotion scoring.
- Next command after synchronizing the next commit: run `scripts/run_experiment.py`
  with `--mode r00` to score the exact-template and September-aligned replays, then
  package/download the resulting artifact before any model fit.
