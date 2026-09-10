# Artifact recovery matrix — 2026-09-10

This matrix records what is independently available after the Kaggle runtime
loss and migration to `zindi-gcp`.  A historical path or metric is not treated
as a current artifact unless its bytes and manifest checksum are present.

| experiment/source | artifact available now | verification | status/action |
| --- | --- | --- | --- |
| B0 hydrological trajectory, commit `d21ccba` / `545fe996` | fold models + OOF for Sep-2007, Jan-2009, Apr-2014, Dec-2014 in two archives | archive SHA-256 and embedded manifests preserved under `recovered_archives/` | recoverable; inspect remotely, do not substitute for D0 |
| B1 local, commit `545fe996` | four fold models + OOF | archive SHA-256 and embedded manifest | recoverable historical evidence |
| B2 regional, commit `545fe996` | four fold models + OOF | archive SHA-256 and embedded manifest | recoverable historical evidence |
| B3 both, commit `545fe996` | Jan-2009, Apr-2014, Dec-2014 fold models + OOF | archive SHA-256 and embedded manifest | recoverable late-fold evidence; Sep-2007 is absent from this archive |
| full-trained dense B3/98 | available on `zindi-gcp` as Train-only package | manifest records 1,976,942 rows, exact row/label/weight hashes, model SHA-256 `3895b7e3...`; package SHA-256 `4de3ec9d...` | current GCP diagnostic artifact; environment-drift bytes, no Test rows |
| Dec-2014 D0 model/OOF/manifest | available on `zindi-gcp` | package `gcp_d0_dec2014_20260910.zip`, SHA-256 `f5065e60...`, 109,439 validation rows | rebuilt frozen late stress fold; raw RMSE `0.8096875765`, weighted `0.7895493322` |
| matched A/B/C package | available on `zindi-gcp` | `gcp_matched_comparison_20260910_v2/`, details SHA-256 `764f7fe1...`; all cell-SSE checks pass | exact paired replay complete; no row-level prediction/submission output |
| local-response archive | available (`local_response_20260907T212420Z.zip`) | local and VM SHA-256 `7a45c3a9...` (full hash in `recovered_archives.sha256`) | historical comparator only |

## Input data

The VM copies are authoritative and hash-matched to the Windows files:

- Train: `97ff1912b35871574a01900c24a792a9a418653e87f01dccc1788c6838d94b1f`
- Test: `314da7996fa947b30797d82ea8d0ea34240fe52683ecf102a21c37f00f61c196`
- SampleSubmission: `77881bc7257791d583c5a1f496a5639864d490a0237515338fc5305ec943db22`

No Test predictions, submission files, calibration, or uploads were created
during this migration.  Existing historical submission CSVs remain evidence,
not a new output.

## Resume gate

The minimum useful recovery state is complete: the VM, hash-matched inputs,
isolated environment, non-training tests, frozen Sep-2007 parity, late D0,
Train-only full B3, and matched A/B/C outputs are all present and inspectable
through `ssh zindi-gcp` in tmux session `zindi`.  The matched result shows
B/full-B3 and C/dense are identical because the intervention had
`sparse_withheld_window_rows=0` and changed zero features.  Treat this as a
no-treatment control, not evidence that sparse covariates cannot matter.  The
bounded next diagnostic is the fixed 98-vs-392-round capacity comparison; the
old undefined recency-weighted experiment is closed.  No reset to `origin` is
permitted; the branch's intentional commits must be preserved.
