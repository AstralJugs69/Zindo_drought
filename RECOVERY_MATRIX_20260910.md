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
| full-trained dense B3/98 | absent | only historical docs/metrics references remain | do not claim available; rebuild only after baseline parity and explicit run gate |
| Dec-2014 D0 model/OOF/manifest | absent | not in local archive inventory or fresh VM | exact matched diagnosis remains incomplete |
| matched A/B/C package | absent | no `leaderboard_failure_*` package recovered | restore exact artifacts or perform the authorized VM rebuild |
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

The minimum useful state is not yet complete: the VM, inputs, environment, and
non-training tests are ready, and fold-level recovery is available, but exact
D0/full-B3/matched A/B/C artifacts are missing.  The next authorized action is
to run one frozen Sep-2007 D0/B3 parity validation on the VM in tmux, persist a
complete manifest, and use its measured result to decide whether late-period
D0 and full-B3 rebuilds are scientifically justified.  No reset to `origin`
is permitted; the branch's intentional commits must be preserved.
