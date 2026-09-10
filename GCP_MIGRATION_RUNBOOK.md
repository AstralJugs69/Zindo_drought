# Persistent GCP migration runbook

**State captured:** 2026-09-10 (Africa/Addis_Ababa)

The working execution host is the persistent VM reached with `ssh zindi-gcp`.
Kaggle/Pinggy is not part of this workflow.  All long-running commands must be
started in the detached tmux session `zindi` and observed there; the pane is
wrapped by `script -af /home/milli/zindi-session.log`.

## Reconnect and inspect

```bash
ssh zindi-gcp
tmux attach -t zindi
```

The checkout is `/home/milli/zindi_drought_gcp` and the isolated interpreter is
`/home/milli/zindi_drought_gcp/.venv/bin/python`.  A fresh shell should use:

```bash
cd /home/milli/zindi_drought_gcp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
```

Do not run model fitting or scoring from the Windows checkout.  Do not invoke a
Test-prediction/submission path without explicit authorization.  The VM is
CPU-only; no GPU/neural campaign is planned.

## Verified host and software state

The VM reported Ubuntu 22.04, 48 vCPUs, about 62 GiB RAM, and about 192 GiB
free on the 194 GiB root disk.  The checkout is on
`codex/validation-rebuild` at `09d52e3` after the diagnostic runner's schema
persistence fix, with no tracked changes.  The completed recovery diagnostics
were launched at the preceding clean commit `7e44744739065dfd6bf7a5e53ae3af28b3b79450`.

The environment is intentionally isolated.  Exact observed versions are kept
in `/home/milli/zindi_drought_gcp/environment.lock.txt`:

```text
Python 3.10.12
numpy 2.2.6
pandas 2.3.3
lightgbm 4.7.0
scikit-learn 1.7.2
pytest 9.1.1
```

The repository test suite was run in tmux with the four-thread caps and passed
`56 passed in 6.51s`.  This is a non-training parity check, not model-score
parity; package/hardware differences from historical Kaggle runs remain.

## Authoritative inputs and recovery

`Train.csv`, `Test.csv`, and `SampleSubmission.csv` were copied from the
Windows checkout to the VM and verified there with SHA-256:

| file | SHA-256 |
| --- | --- |
| `Train.csv` | `97ff1912b35871574a01900c24a792a9a418653e87f01dccc1788c6838d94b1f` |
| `Test.csv` | `314da7996fa947b30797d82ea8d0ea34240fe52683ecf102a21c37f00f61c196` |
| `SampleSubmission.csv` | `77881bc7257791d583c5a1f496a5639864d490a0237515338fc5305ec943db22` |

Surviving local archives are preserved at
`/home/milli/zindi_drought_gcp/recovered_archives/`; their verified hashes are
in `recovered_archives.sha256`.  They contain completed fold-level B0/B1/B2/B3
models and OOF files and compact manifests.  The previously missing exact
diagnostic state has now been rebuilt in three unique directories:

- `drought_runs/gcp_d0_b3_parity_20260910` — Sep-2007 D0/B3 parity, raw RMSE
  `0.5283539879552412` (a measured environment-drift trigger).
- `drought_runs/gcp_d0_dec2014_20260910` — Dec-2014 D0 package, SHA-256
  `f5065e604b84289f6c33ce8a99e620702a878810289c244547e0e04d7bb73db7`.
- `drought_runs/gcp_full_b3_train_only_20260910` — Train-only full B3 package,
  SHA-256 `4de3ec9d52865d1e45247d511cf7a374ac64c314ecb57d16eb4c2795122c4858`.
- `drought_runs/gcp_matched_comparison_20260910_v2` — completed matched A/B/C
  compact outputs; details SHA-256
  `764f7fe16dda19976126dd4eb0bdb3ff3a1bf49a60d933c708ecee94435ec9dc`.

The full B3 run used 1,976,942 sampled rows (2,154,021 source rows minus
177,079 missing-anchor rows), and the matched stage read no Test labels and
wrote no row-level prediction or submission file.  These are current GCP
diagnostic artifacts; their booster bytes are not claimed byte-identical to
the historical Kaggle model.

The persistent disk is not an independent backup.  Backup status is currently
**not configured/verified**; do not describe VM-local copies as backed up.

## Safe execution contract

Before a run, create a unique directory under `/home/milli/zindi_drought_gcp/
drought_runs/`, write config/data/code fingerprints and the intended scenario,
and use atomic finalization for manifests, metrics, models, OOF, diagnostics,
logs, exit status, timing, and peak RSS.  Keep target-blind feature caches
keyed by input/code/recipe/schema/cutoff/availability fingerprints; never mix a
dense/sparse/Test view.  Run at most two folds concurrently only after checking
the memory estimate and retaining a 12 GiB reserve.  Start with four threads;
benchmark feature construction, fitting, inference, and RSS separately before
trying 12 or 24 threads.

The frozen parity and matched diagnostic are complete.  Their key result is
documented in `LEADERBOARD_ROOT_CAUSE_REPORT.md`: on the same 109,439-row
replay, A/D0 scores `0.789549` official weighted RMSE while B/full-B3 scores
`0.737579`; C/dense is exactly equal to B.  Cell-SSE aggregation is conserved
to `1.82e-12`.  This does not authorize the one recommended next experiment:
a late-prefix, Test-schedule-matched recency-weighted B3/98 fit.  Keep that
experiment gated behind explicit authorization and do not create Test outputs.
