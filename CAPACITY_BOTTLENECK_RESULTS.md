# B3 capacity bottleneck comparison — 2026-09-10

## Decision

The bounded comparison is complete on `zindi-gcp` at commit
`a44bf7053b187133f5d26aea4b44fe03aef13ef4`.  It fit the unchanged dense-history
B3 recipe once to 392 boosting rounds and scored checkpoints 98 and 392 on the
same paired rows.  The predeclared recent-origin gate **fails**:

- 2014-04 improves by `0.014111` raw h1--7 RMSE (`0.581798` to `0.567687`),
  although h=4 is absent from this replay and is not imputed.
- 2014-12 worsens by `0.004654` raw h1--7 RMSE (`0.809378` to `0.814031`).
- No recent source month with at least 1,000 rows regresses by more than the
  allowed `0.02` raw RMSE, but both recent origins were required to improve.

Therefore, extra boosting rounds are **not** a validated general remedy for
the late-period failure.  Retain B3/98 as the development control, close the
capacity hypothesis as a standalone explanation, and do not launch a larger
round grid.  The result is consistent with the existing late target-process /
transfer and validation-support diagnosis: training RMSE falls at 392 rounds,
but the complete late December replay does not improve.

No Test labels, Test predictions, calibration, submission, or upload were
created.

## Run contract and environment

Remote run directory:

```text
/home/milli/zindi_drought_gcp/drought_runs/gcp_b3_capacity_20260910/
```

The run was submitted in the persistent `zindi` tmux session, wrapped by
`script -af /home/milli/zindi-session.log`.  It completed in `1433.053592081`
seconds (~23m53s).  The VM has 48 vCPUs and about 62 GiB RAM; the isolated
environment is Python 3.10.12, NumPy 2.2.6, pandas 2.3.3, LightGBM 4.7.0,
and scikit-learn 1.7.2.  The run was explicitly launched with four LightGBM
threads before the later resource-policy change; the manifest records
`requested_num_threads=4` and `num_threads=4`.  `psutil` was not installed, so
the manifest's RSS tracker is unavailable; read-only process spot checks saw
RSS below roughly 22 GiB and ample VM headroom.

For future executions, the repository defaults now use a **12-thread baseline**
and allow explicitly authorized escalation to 24 or 48 threads.  The
already-running comparison was left uninterrupted and remains a valid
fixed-recipe result.

The intervention was only:

```text
98 versus 392 boosting rounds
seed=20260908, learning_rate=0.05, num_leaves=63,
min_data_in_leaf=1000, 446 features, force_col_wise=True,
same sampled IDs/targets/anchors/horizon weights, no early stopping
```

For each origin, features were built once and reused for both checkpoints;
training targets were strictly before the origin.  April's missing h=4 support
is explicit.  The optional 2009-01 transfer check was not run because the
recent gate failed, and Test was never read (`test_rows_read=0`,
`test_labels_read=false`).

## Validation results

Raw RMSE is over the indicated paired rows.  The weighted column is the
official h1--7 weighting where all seven horizons are present; it is
intentionally `n/a` for April.

| origin | validation rows | official rows | h1--7 horizons | iter98 raw | iter392 raw | delta (392−98) | iter98 weighted | iter392 weighted | stress h>7 98 → 392 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 2014-04 | 77,960 | 77,863 | 1,2,3,5,6,7 (h=4 absent) | 0.581798 | 0.567687 | −0.014111 | n/a | n/a | 1.103301 → 1.063767 |
| 2014-12 | 109,439 | 109,349 | 1,2,3,4,5,6,7 | 0.809378 | 0.814031 | +0.004654 | 0.789549 | 0.791794 | 1.124937 → 1.104837 |

The persistence raw h1--7 references are `0.766748` (2014-04) and `0.861437`
(2014-12).  The 392-round model still beats persistence, but the December
checkpoint is worse than its own 98-round control.  Training absolute-target
RMSE falls from `0.514955` to `0.439992` (April) and from `0.515843` to
`0.441685` (December), which is not accompanied by a late validation gain.

### Official source-month deltas

All rows in this table have at least 1,000 examples; the gate threshold is
`delta <= +0.02`.

| origin | source month | rows | iter98 RMSE | iter392 RMSE | delta |
|---|---|---:|---:|---:|---:|
| 2014-04 | 2014-04 | 15,644 | 0.535389 | 0.529106 | −0.006283 |
| 2014-04 | 2014-05 | 15,581 | 0.541138 | 0.526465 | −0.014673 |
| 2014-04 | 2014-06 | 15,577 | 0.569487 | 0.558633 | −0.010854 |
| 2014-04 | 2014-09 | 15,519 | 0.643561 | 0.617676 | −0.025885 |
| 2014-04 | 2014-10 | 15,542 | 0.612444 | 0.600943 | −0.011500 |
| 2014-12 | 2014-12 | 15,637 | 0.636306 | 0.633289 | −0.003018 |
| 2014-12 | 2015-01 | 15,651 | 0.843275 | 0.851830 | +0.008555 |
| 2014-12 | 2015-02 | 15,665 | 1.052076 | 1.049376 | −0.002700 |
| 2014-12 | 2015-03 | 15,665 | 0.540907 | 0.538250 | −0.002657 |
| 2014-12 | 2015-04 | 15,634 | 0.603411 | 0.605966 | +0.002555 |
| 2014-12 | 2015-05 | 15,558 | 0.674197 | 0.677413 | +0.003216 |
| 2014-12 | 2015-06 | 15,539 | 1.119835 | 1.139118 | +0.019283 |

The largest gated monthly regression is December-origin June 2015 at
`+0.019283`, below the `+0.02` guardrail.  This does not rescue the origin
gate: January, April, May, and June 2015 all regress, while other horizons and
months improve.

### Official horizon deltas

April has no h=4 rows.  The five April h=5 rows are retained in the audit but
are too small to carry a reliable aggregate conclusion.

| origin | h | rows | iter98 RMSE | iter392 RMSE | delta |
|---|---:|---:|---:|---:|---:|
| 2014-04 | 1 | 15,644 | 0.535389 | 0.529106 | −0.006283 |
| 2014-04 | 2 | 15,574 | 0.541232 | 0.526545 | −0.014687 |
| 2014-04 | 3 | 15,548 | 0.568534 | 0.558023 | −0.010511 |
| 2014-04 | 5 | 5 | 0.124897 | 0.136497 | +0.011600 |
| 2014-04 | 6 | 15,548 | 0.644285 | 0.618136 | −0.026148 |
| 2014-04 | 7 | 15,544 | 0.612412 | 0.600919 | −0.011493 |
| 2014-12 | 1 | 15,637 | 0.636306 | 0.633289 | −0.003018 |
| 2014-12 | 2 | 15,627 | 0.842080 | 0.850465 | +0.008385 |
| 2014-12 | 3 | 15,607 | 1.045686 | 1.043460 | −0.002226 |
| 2014-12 | 4 | 15,659 | 0.549182 | 0.546643 | −0.002539 |
| 2014-12 | 5 | 15,638 | 0.606719 | 0.608951 | +0.002231 |
| 2014-12 | 6 | 15,596 | 0.675010 | 0.677987 | +0.002977 |
| 2014-12 | 7 | 15,585 | 1.120588 | 1.139753 | +0.019165 |

### Five-degree spatial summaries

The complete additive cell table is retained remotely in
`validation_by_geo5.csv`.  Collapsing source months leaves 893 observed
5-degree cells per origin.  Counts below compare each cell's official h1--7
RMSE at the two checkpoints.

| origin | official rows | iter98 RMSE | iter392 RMSE | cells improved | cells worsened | largest cell regression | largest cell gain |
|---|---:|---:|---:|---:|---:|---|---|
| 2014-04 | 77,863 | 0.581798 | 0.567687 | 492 | 401 | (lat5=55, lon5=95) +0.158164 | (lat5=55, lon5=115) −0.205699 |
| 2014-12 | 109,349 | 0.809378 | 0.814031 | 402 | 491 | (lat5=40, lon5=−90) +0.155558 | (lat5=20, lon5=−105) −0.164164 |

The spatial split is mixed rather than a uniform capacity gain.  Aggregate
cell SSE is the same as the row-level official SSE; the full per-month,
per-cell table remains the audit source.

## Paired identity and availability hashes

These values are copied from the two remote `origin_*_details.json` manifests;
they prove that each checkpoint used the same paired support.

| origin | training IDs | training target | training delta | training weights | validation IDs | validation target | validation anchor | validation h |
|---|---|---|---|---|---|---|---|---|
| 2014-04 | `16ba0c013cde4de12146f11f387f4037690ceaac7e94f044720cd8598b796964` | `ae1b09b86c4b60b7b4fa591e45819bc807bd2d34c4a5a1f372b7be4ca0354fdc` | `85dc5339147b90baae15d75410424916385513711a8cf099ff908e6b23ee45c2` | `edb074b89aa514006c0319cc4acb9c6c3924ef8768c320c8c6dc4f9b9f5f7162` | `44dff5d7392acbdc2400d5e4f01598b40dc084c4bf97096483298063a0a6d840` | `3e2a8dc743ecee27770a212fc35a348a246e4739dc6122d1394e7bf8ff96f843` | `0190645fd835a59cbb18bdd28c293de05e14f007e8278d41f2310337efaf72da` | `6b228d7ee3e78ea5f6b80f09ed98bcc017a7dc81864744ad2603d300c408562a` |
| 2014-12 | `f04b731341b003426e5cad67d646ce2cdd4053298931816fbd512a3398e69ea3` | `8bb13a6f385120f333f4fb647d1bdfb928082a69a4e2cc60c79deefbe889ab5a` | `a243435a3e7fe0fd89e7f087d97fadf5a5c00893888406d1b8545f520b66a19e` | `833621c37a41f2ea670fbfd77b4b64e90dce08d0a976947162191dc6dd538331` | `9a8e1f492d38878933254a6999919d8a621fa505c02c33823c6f3e16d432cd82` | `3c9284bd222ab9c62307892bab042482ca1f715fdabc3595dadae1657436457f` | `379481c304a89920f5b005f464a5a458984e38a652670fb5475ab8eb86cea67d` | `7e86464b48e7a3e36cd2ca61109c9209c203303a0eff3a7d9a71283f4e5a9101` |

The feature schema hash is `a62c9ee7361ba8998b305a03a11955dcdb312b28da876d2455291734faf011ea`
for both origins.  Matrix/view fingerprints are:

| origin | training matrix SHA-256 | validation matrix SHA-256 | source/structural ID hash | source rows | withheld-window rows |
|---|---|---|---|---:|---:|
| 2014-04 | `093a40287cd24e511372090146560ad166ed031c12c29b7bfe267e7c09677e9c` | `56c6aa194d4b0e3ac6740e06c992ef0f34131a3b24d3acaada4d08fd9f6c5dc4` | `9d917cb70985ed7a7e7e0aeb7e0e343ff611c8b1381eec4fb03f923e4fabd52c` | 1,997,822 | 0 |
| 2014-12 | `557055027530b97da808b952e732d236b0b8f7b9e4934b81e68328041cfc1e37` | `6eca8e6ff3a2e4c509d60a0ce216db6f6c02cac403e57fc9f4bdf44113f17687` | `6042a34b3c51e42dbc8cd5af63918fff2a5038dc4abed1e5c3bdb301a9a03331` | 2,122,894 | 0 |

April training used 1,775,112 rows after 144,750 missing-anchor exclusions;
December used 1,827,911 rows after 169,911 exclusions.  The target cutoffs
were 2014-03 and 2014-11 respectively.

## Dec-2014 D0 checkpoint control

The independently rebuilt D0 control was available only for December.  Its
model SHA-256 is
`13afa9a7f1e41577442e954b35fdcba1566eba644d267788601fcea30c828da9` and its
OOF SHA-256 is
`fc223cb81559f2a4dc3b2e4808d49a99ed1499ccca756dbeb55c80cbd36f9bc0`.  The
feature schema has 446 columns, OOF IDs are equal, and the D0 prediction is
exactly equal to the capacity run's iteration-98 prediction (`max_abs=0` and
`prediction_within_1e-6=true`).  OOF target and anchor differences are below
`1.82e-7` and `1.19e-7` respectively, attributable to stored float precision.
No April D0 control was available.

## Durable artifact index

All large artifacts remain on the VM for remote inspection; none were copied to
Windows.  The run's compact manifest and summaries are:

```text
manifest.json                         e839dd7976285bb6230f1a16441e49463301fc3b12b64c6d8e6e9c5d9ef9c53b
gate.json                             952ef965f8d291e65de8cd5a31831fcf69c128c10b91b1dde02801ae4a9e9491
origin_2014-04_details.json           8eada447495f4cf1bf08fc1a2cc6bf1a25c0b305d9ad253dde73b4069e8648fb
origin_2014-12_details.json           b69cc18e40c15030a61754aaffae0771c47abc4b26755ed5b989a5c436cbf206
validation_overall.csv                92e2e8b32f961ecdf62d04461e8a35772f0db5aa8cc31ed489daf343960a643f
validation_by_source_month.csv        c5064c4ddb335dcf0f1106526f9280d9041b032dec00b53be8531fc1fd82ea21
validation_by_horizon.csv             9be50f0704f38c33030f2b65cfb77e5f3f7299fc0c1f5176d7c412c50b606ff3
validation_by_geo5.csv                bc72e10689123c904c996f09e2a7822207bdc293f51fbb27707d228eec421958
train_metrics.csv                     9d5a04bdc8295c0f7c9e52248b92d10c5db24714989491feb1bde5440ebfc3bc
```

Checkpoint model and OOF hashes are recorded in the same directory's
`origin_*_details.json` files.  There are no files whose names indicate Test
predictions or submissions, and the manifest records `status=completed`,
`submission_written=false`, `test_rows_read=0`, and `test_labels_read=false`.
