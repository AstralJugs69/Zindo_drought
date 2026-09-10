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
free on the 194 GiB root disk.  No training process was running during the
initial inspection.  The checkout is on `codex/validation-rebuild` at
`32e506b2412e62400933ef06496fe191b3184163` with no tracked changes.

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
`56 passed in 5.09s`.  This is a non-training parity check, not model-score
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
models and OOF files and compact manifests.  The exact full-trained B3 model,
the Dec-2014 D0 package, and the matched A/B/C directory were not present in
the VM or local archive inventory.  Historical metrics and commit references
must therefore remain labelled historical until an artifact's manifest and
checksum are independently verified.

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

The scientifically useful next step is one representative availability-faithful
Sep-2007 D0/B3 validation using the frozen historical recipe (98 rounds, 63
leaves, `min_data_in_leaf=1000`, 446 features) only after the runner's actual
parameters and artifact contract are verified on this VM.  Its historical raw
RMSE (`0.5283539879552412`) is a drift trigger, not a score to force.  After
parity, rebuild the late Dec-2014 D0 OOF and, only if required, the full-trained
B3 artifact on the VM; then run the matched A/B/C comparison on identical
origin/sample/horizon exposure.  Do not run that next experiment automatically
from this runbook.
