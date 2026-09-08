# Campaign results — 2026-09-08

Status: **in progress**. This file is the durable campaign report; entries are
added only after fresh Kaggle evidence is inspected.

## Requirement checklist

| Branch | State | Evidence / limitation |
|---|---|---|
| Corrected Phase A accounting | measured | Kaggle run `spatial_phase_a_repaired_20260908T013000Z`; commit `0f406fa...`; corrected values and retraction in `NEXT_ACTION.md` |
| Spatial B1 ranks 8/16 | measured-negative | Kaggle run completed; all origins regressed; per-rank OOF persisted |
| Spatial B2 ranks 8/16 | failed-runtime | First run reset Kaggle session; retry is currently blocked at session startup |
| Regional 5/15-degree paired features | measured-partial | Both widths completed at frozen-173 with per-fit OOF/model files; inner capacity selection and transfer checks pending |
| Inner capacity selection | pending | Chronological inner checks not yet run |
| Additional recent/transfer block | pending | Coverage must be established before scoring |
| Matched-data masking retest | pending | Not yet implemented |
| Downloaded artifacts | partial | Earlier local-response archive is durable locally; newer and campaign artifacts remain session-local until verified |

No Test predictions, submission files, or Zindi submissions are permitted in
this campaign.

## Regional width study (measured)

Kaggle run `regional_widths_20260908T031000Z` completed from commit
`fdee96573583f22cdaf4d17f2b9d58de9bb9a219`. The paired frozen-173 results
(C0/C1 raw RMSE) were:

| Origin | 5-degree C0 | 5-degree C1 | 15-degree C0 | 15-degree C1 |
|---|---:|---:|---:|---:|
| Sep-2007 | 0.537422 | 0.530604 | 0.537422 | 0.529463 |
| Jan-2009 | 0.574104 | 0.563443 | 0.574104 | 0.559609 |
| Dec-2014 h1–7 | 0.822625 | 0.818479 | 0.822625 | 0.818389 |

This measures added regional information at frozen capacity only. Inner
chronological capacity selection and the additional recent/transfer checks are
still pending. The run persisted per-fit OOF CSVs and LightGBM model files in
the Kaggle output directory; local download durability has not been verified.

## Spatial B1 screen (measured)

The causal B1 projection run `spatial_b1_20260908T041000Z` completed from commit
`342f84be92cf0a5ebc465149ef9d2522ebd59c2c`. It projected R01 predicted deltas
onto the prefix-fitted basis at ranks 8 and 16, preserving each legal anchor.
Coverage was 100% on all folds, but the screen regressed materially:

| Origin | R01 RMSE | B1 rank 8 | B1 rank 16 |
|---|---:|---:|---:|
| Sep-2007 | 0.537422 | 0.617560 | 0.606508 |
| Jan-2009 | 0.574104 | 0.694490 | 0.671714 |
| Dec-2014 h1–7 | 0.822625 | 0.880291 | 0.867192 |

This is a clean negative B1 result, not a reason to cancel B2; the dynamic
factor hypothesis remains separately required.

The first B2 attempt (`spatial_b2_20260908T050000Z`, commit
`45ae11f...`) did not produce a manifest: the Kaggle session reset during the
run. This is recorded as a runtime failure/diagnostic gap, not as a model
score. The runner now supports a scoped recent-origin smoke retry before the
full two-rank, three-origin execution.

The subsequent retry was not started because the reopened Kaggle notebook
remained in `Draft Session Starting` with execution controls disabled. This is a
separate infrastructure-state blocker; no claim is made about B2 predictive
performance.

## Executed evidence

See `NEXT_ACTION.md` for the corrected spatial gate, the paired regional smoke
metrics, and the downloaded artifact checksum. Oracle projections are labelled
`ORACLE_DIAGNOSTIC` and are not deployable forecasts.
