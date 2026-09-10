# Legal neighbor-state B3/98 intervention — results

**Run date:** 2026-09-10  
**Host:** `zindi-gcp` (persistent `zindi` tmux session)  
**Remote artifact directory:**
`/home/milli/zindi_drought_gcp/drought_runs/gcp_neighbor_state_b3_20260910_v1`  
**Source commit:** `fb12f744dbcb3db58ea84e5d76faae6fd3349001`  
**Decision:** retain the frozen dense-history B3/98 control; do not promote C1.

## Question and frozen comparison

The intervention tested whether legal neighboring TWS state helps the frozen
446-feature B3/98 booster under the same availability-faithful replay ledger.
The control (C0) kept the original 446 columns.  C1 added exactly four
features:

* `neighbor_tws_mean`
* `neighbor_tws_minus_focal`
* `neighbor_tws_count`
* `neighbor_tws_median_age`

All LightGBM settings were frozen: seed `20260908`, 98 boosting rounds, 63
leaves, `min_data_in_leaf=1000`, and `num_threads=12`.  C1 therefore had 450
columns.  The fixed spatial geometry used the eight nearest *other* locations,
with a 500 km cap and no replacement.  The latest finite neighbor TWS was
taken at or before each event's focal `last_observed_date`; source month was
used only to compute observation age.  Summaries were left missing when fewer
than four neighbors were supported.

The inner screen used paired `2003-04` and `2004-04` origins.  Because it
passed, the predeclared recent screen used paired `2014-04` and `2014-12`
origins.  The optional `2009-01` transfer check was conditional on the recent
gate and was not run after that gate failed.

## Gate results

Raw h1--7 RMSE is the primary gate metric.  Positive `C1-C0` means a
regression; the displayed gain is `C0-C1`.

| Origin | C0 raw h1--7 | C1 raw h1--7 | C0-C1 gain | Test-horizon-weighted proxy | Gate evidence |
|---|---:|---:|---:|---:|---|
| 2003-04 (inner) | 0.580718638 | 0.580038530 | +0.000680108 | 0.580619774 -> 0.579943267 | strict improvement |
| 2004-04 (inner) | 0.560775988 | 0.557875951 | +0.002900037 | 0.560653095 -> 0.557747652 | strict improvement |
| 2014-04 (recent) | 0.581798304 | 0.582859573 | -0.001061269 | not defined | strict improvement failed |
| 2014-12 (recent) | 0.809377539 | 0.806802197 | +0.002575342 | 0.789549332 -> 0.788070604 | nonregression and improvement |

The inner gate passed because C1 strictly improved raw h1--7 RMSE on both
inner origins.  The outer gate failed because both recent origins were
required to improve and 2014-04 regressed.  April has no h=4 support, so its
present h1--7 rows were evaluated directly and no weighted proxy was invented.
The December weighted proxy nonregressed (improvement `+0.001478729`).  No
source month with at least 1,000 rows exceeded the `+0.02` raw-RMSE regression
guardrail; the largest was `+0.009075` at source month 2005-04 in the
2004-04 replay.  The optional transfer gate is recorded as
`status=not_run, reason=outer_gate_not_passed`.

Longer-horizon stress was mixed in the same paired OOF: 2014-04 h>7 changed
from `1.103301117` to `1.134140543` (97 rows), while 2014-12 h>7 changed from
`1.124936664` to `1.081612612` (90 rows).  These small-support stress rows do
not override the predeclared h1--7 recent gate.

## Availability and causality checks

The fixed geometry contains 15,715 locations.  The median and 90th-percentile
distance to the eighth-neighbor slot are 111.195 km and 155.815 km, well below
the 500 km cap.  Across all four origins, validation support was approximately
99.91%, the neighbor count median was eight, and the age median was 1 month on
the inner replays, 2 months at 2014-04, and 3 months at 2014-12.  The maximum
age quantiles reached 5 months for training/inner rows and 6 months for the
recent validation rows.  Unsupported summaries remained missing rather than
being interpolated.

The persisted `prefit_checks.json` reports `status=passed` for every required
check:

* target perturbation invariance;
* future-TWS perturbation invariance;
* masked-TWS perturbation invariance;
* missing months are not interpolated;
* self-exclusion; and
* longitude-wrap handling.

Every origin detail records zero cutoff excess and zero source-date excess.
The event cutoff is the focal last-observed date, not the event source date,
and target-time neighbor state is explicitly unused.

## Reproducibility and resource record

The run completed with exit status 0 in 1,487.7468 seconds (24:48.45 wall
time).  `/usr/bin/time -v` measured 2,446.68 user seconds, 122.80 system
seconds, 172% aggregate CPU, 25,581,984 kB maximum resident set (about
25.34 GiB), zero swap, and no major page faults.  The in-process resource
sampler could not import `psutil`, so its manifest fields are intentionally
null; the external tmux monitor and `/usr/bin/time` values above are the
authoritative resource record.  The 12-thread baseline was left unchanged and
no concurrent high-thread job was started.

There were eight candidate records.  The four inner C0/C1 fits and the two
recent C1 fits were newly fit; the recent C0 models were reused only after
their saved 446-column, 98-tree provenance matched the frozen control.
All models have exactly 98 trees.  The remote directory contains the models,
OOF ledgers, per-horizon and per-source-month tables, details, manifest,
configuration, and event log for full inspection without copying large
artifacts to Windows.

The manifest records Train SHA-256
`97ff1912b35871574a01900c24a792a9a418653e87f01dccc1788c6838d94b1f` and
geometry-only Test SHA-256
`314da7996fa947b30797d82ea8d0ea34240fe52683ecf102a21c37f00f61c196`.
Test geometry consisted of 280,961 rows and only
`ID,time,lat,lon,TWS_t,TWS_t_masked` were read.  `test_labels_read=false`,
`test_predictions_written=false`, and `submission_written=false`; no
competition Test prediction or submission was created.

## Decision and next step

Legal neighboring TWS is a real, causal, highly supported signal in the
replay, and it improves both historical inner origins.  It is not a robust
replacement for B3/98 under the recent transfer gate: the 2014-04 regression
outweighs the 2014-12 gain under the predeclared rule.  Retain C0/B3/98 as the
development control, retain this run as audit evidence, and do not generate
Test predictions, submissions, or a larger neighbor-feature sweep from this
result.  Any further representation or transfer hypothesis needs separate
authorization and a fresh frozen gate.
