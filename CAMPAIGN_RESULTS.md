# Campaign results — 2026-09-08

Status: **in progress**. This file is the durable campaign report; entries are
added only after fresh Kaggle evidence is inspected.

## Requirement checklist

| Branch | State | Evidence / limitation |
|---|---|---|
| Corrected Phase A accounting | measured | Kaggle run `spatial_phase_a_repaired_20260908T013000Z`; commit `0f406fa...`; corrected values and retraction in `NEXT_ACTION.md` |
| Spatial B1 ranks 8/16 | pending | Requires causal forecast implementation and Kaggle run |
| Spatial B2 ranks 8/16 | pending | Requires causal temporal-factor implementation and Kaggle run |
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

## Executed evidence

See `NEXT_ACTION.md` for the corrected spatial gate, the paired regional smoke
metrics, and the downloaded artifact checksum. Oracle projections are labelled
`ORACLE_DIAGNOSTIC` and are not deployable forecasts.
