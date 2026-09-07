# Next action — local-response stress result

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
  The supported Kaggle Output download action was invoked once, but no matching
  local file appeared in the expected Downloads location; it is therefore not
  claimed as locally durable.
- Ledger correction: R02/R03's observed regressions remain real for their
  exact recipes, but their schedule-training windows all collapsed to the same
  May-2002 origin. They do not isolate realistic observation masking and must
  not be treated as a general rejection of masking-aware training. Do not rerun
  those unchanged experiments.
- Next bounded assignment: design the predeclared matched-data masking test
  with a training-coverage table before fitting. Keep the local-response branch
  frozen unless a separately authorized experiment tests a new causal
  formulation; do not tune alpha, blend weight, or generate a submission.
