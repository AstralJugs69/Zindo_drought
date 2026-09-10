# Late-2015 spatial/reversal diagnostic

**Run:** `/home/milli/zindi_drought_gcp/drought_runs/gcp_late2015_spatial_reversal_20260910_v10`  
**Source commit:** `c3575fb119751ef0925d748081b82ec743684efa`  
**Input:** `Train.csv` SHA-256 `97ff1912b35871574a01900c24a792a9a418653e87f01dccc1788c6838d94b1f`  
**OOF inputs:** April iter98 `60b7d2ce81089e58cdcea186f3d4edf14b96e4dbb6844619b7f7469dc8857176`; December iter98 `0c411ccc1dca72f1d8aa890d0aac78654a947e74fd10aa1065a751a51e48c440`

## Decision

The late-2015 failure is concentrated in a spatially coherent, temporally
reversing TWS-change regime. It is not primarily a nearest-neighbour support
failure, a longitude-wrap/self-join bug, or purely local noise. A legal
source-date neighbour-TWS state is available and is not represented in the
446-column B3 schema, so one fixed B3/98 neighbour-state intervention is
justified as the next experiment. It was **specified but not launched** here.

All spatial/reversal quantities below use `d_t = target_t - TWS_t` and are
therefore `ORACLE_DIAGNOSTIC` descriptions, never predictor inputs.

## Calendar and identity contract

- 2,154,021 Train rows, 15,715 fixed coordinate locations, and 138 source
  calendar months were indexed by integer `(location_id, source_month)` keys.
- `target_month` is explicit and is always `source_month + 1`; no row shifting or
  adjacent-row join is used.
- 1,977,398 rows have an exact next-calendar source row and satisfy the target
  identity with maximum and mean absolute difference `0.0`; 176,623 terminal or
  missing-next-month rows are excluded from reversal pairs.
- The full canonical table is retained remotely as
  `canonical_train_only.csv.gz` (SHA-256
  `fa31fc8d110078d68058aae2e7a1a6f3084b8fd364a6650c0bbf04791dc7a578`).

## Spatial structure

The fixed coordinate index uses the eight nearest *other* locations, great-circle
distance on unit vectors (longitude-wrap safe), a 500 km cap, and a minimum of
four finite same-month neighbour values. Across all 15,715 locations, 125,604 of
125,720 neighbour slots are within the cap; distance quantiles are 12.59 km,
111.20 km, 155.81 km, and 497.24 km (0th/50th/90th/100th).

| source month | d RMSE | neighbour corr | sign agreement | roughness RMSE | 5-degree cell-mean SSE share | supported rows |
|---|---:|---:|---:|---:|---:|---:|
| 2009-01 | 0.504 | 0.990 | 0.969 | 0.073 | 0.842 | 15,634/15,648 |
| 2009-02 | 0.524 | 0.989 | 0.972 | 0.078 | 0.838 | 15,646/15,662 |
| 2009-06 | 0.542 | 0.990 | 0.969 | 0.078 | 0.803 | 15,548/15,565 |
| 2015-01 | 1.012 | 0.991 | 0.974 | 0.142 | 0.830 | 15,639/15,653 |
| 2015-02 | 1.189 | 0.987 | 0.965 | 0.209 | 0.691 | 15,654/15,670 |
| 2015-06 | 1.182 | 0.997 | 0.983 | 0.098 | 0.927 | 15,564/15,581 |

The full 2014–2015 month table is in `spatial_month_metrics.csv`; all months
remain near 99.9% neighbour support. The late-2015 amplitude increase is about
2.0x (January), 2.3x (February), and 2.2x (June) relative to the early controls.
The cell-mean/within-cell decomposition conserves raw SSE to at most
`1.93e-12`, including unequal cell counts. January and June are especially
broad-field events; February has more local roughness but still has 69.1% of
SSE in 5-degree cell means.

Same-location, same-calendar-month comparisons reinforce a regime change rather
than a sampling artifact. For 2015 versus the available 2009/2010/2012/2013
controls, paired `d` difference RMSE ranges 1.130–1.426 for January,
1.339–1.371 for February, and 1.366–1.427 for June; correlations are near zero
or negative.
2011 has no rows for these calendar months and is reported as an explicit
coverage exclusion, not imputed.

## Exact-calendar reversal

Only exact `(location, source_month + 1)` keys are joined; missing months never
bridge. Historical 2009–2013 absolute-`d` thresholds are q50=`0.2810`,
q75=`0.5453`, q90=`0.8805`.

| pair | corr(d_t,d_t+1) | slope | opposite-sign fraction | two-month-sum variance |
|---|---:|---:|---:|---:|
| 2009-01 → 2009-02 | −0.268 | −0.270 | 0.557 | 0.377 |
| 2009-02 → 2009-03 | −0.382 | −0.346 | 0.587 | 0.297 |
| 2009-06 → 2009-07 | −0.329 | −0.338 | 0.619 | 0.406 |
| 2015-01 → 2015-02 | −0.381 | −0.446 | 0.595 | 1.517 |
| 2015-02 → 2015-03 | −0.719 | −0.606 | 0.734 | 0.706 |
| 2015-06 → 2015-07 | −0.700 | −0.569 | 0.792 | 0.672 |

The reversal is stronger in the suspicious periods, especially in high-delta
rows. In the q90+ bin the opposite-sign fraction is 0.650/0.875/0.907 for
January/February/June, and the sum variance is 2.730/1.311/1.084. This is
descriptive evidence of a changed target process; shared TWS coupling means it
is not, by itself, causal proof.

## Saved B3/98 error attribution

The saved OOF rows were joined by exact `sample_id`, then verified against Train
source and target dates before any aggregation. Metrics retain additive `n`,
SSE, SAE, and error sums until final RMSE/MAE derivation.

- April 2014 replay: raw h1–7 RMSE `0.581798`; h=4 is absent, so the
  Test-horizon-weighted validation proxy is correctly unavailable (missing h=4).
- December 2014 replay: raw h1–7 RMSE `0.809378`; complete
  Test-horizon-weighted validation proxy `0.789549`.
- December replay source-month raw h1–7 RMSE is `0.843275` (2015-01),
  `1.052076` (2015-02), and `1.119835` (2015-06). h>7 rows are kept in a
  separate stress slice and do not drive these values.
- 5-degree cell-mean error SSE shares are 0.794/0.713/0.939 for
  January/February/June. Historical-q90+ delta rows account for 0.713/0.726/
  0.727 of error SSE while comprising 0.315/0.332/0.424 of rows.
- Focal anchor age progresses from mostly 1 month (January) to 2–3 months
  (February) to 4–6 months (June), with raw RMSE 0.842/1.049/1.120 on the
  dominant age bands. Age is therefore a plausible exposure amplifier, but the
  extreme-delta and broad-cell results show it is not the sole explanation.

The complete additive tables are `oof_error_metrics.csv`,
`oof_spatial_attribution.csv`, `oof_extreme_delta.csv`, and
`oof_anchor_age.csv`.

## B3 feature and replay audit

The verified B3 schema has 446 features, including 300 hydro-only regional
features. Its only TWS state feature is focal `last_observed_TWS` (with `h` as
the focal-anchor age/horizon metadata); there are no neighbour-TWS aggregate or
neighbour-observation-date features. Regional `reg*`, `dev*`, `count*`,
`coverage*`, and `regional*` features are hydrological context, not TWS state.

Using the existing mask-block replay visibility rule (full prefix through the
origin; later source rows TWS-hidden), legal source-date neighbour state has:

- 99.91% support on the April replay and 99.91% on December;
- median eight neighbours, with median neighbour-anchor age 2 and 3 months
  respectively (90th percentile 6 months);
- zero focal-anchor mismatch against the replay ledger;
- a neighbour mean different from the focal anchor on 99.99%/100% of supported
  events; and
- `view_withheld_window_rows=0` in both saved details, which is preserved as a
  limitation rather than treated as a sparse-versus-dense intervention.

This is a non-empty, legal source-date candidate. No target-time neighbour value
was used as a predictor; all target-bearing spatial/reversal work remains marked
`ORACLE_DIAGNOSTIC`.

## One next experiment (specified, not launched)

Keep the frozen B3/98 recipe, sampled IDs, weights, leaves, learning rate, and
98 rounds. Add exactly these source-date legal features per scored event:

1. `neighbor_tws_mean8_500`: mean of finite latest-visible TWS values from the
   fixed eight nearest other locations within 500 km, requiring at least four;
2. `neighbor_tws_minus_focal`: that mean minus focal `last_observed_TWS`;
3. `neighbor_tws_count8_500`: the finite-neighbour count; and
4. `neighbor_tws_median_age_months`: median of
   `source_month - neighbour_last_visible_month` over the same supported set.

The state is generated from the existing replay visibility ledger only: no dense
Train backfill inside a masked window, no target/future value, and no neighbour
target-time `d`. Unsupported events stay missing for native LightGBM handling.
The inner screen is one fixed paired B3/98-vs-intervention comparison on the
existing 2003-04 and 2004-04 Train-only replays (two fixed seeds, no grid). Take
the intervention forward only if the mean raw h1–7 RMSE improves on both inner
origins and neither regresses by more than 0.002. The outer stop gate is fixed
before execution: both recent replays must improve present-horizon raw h1–7
without imputing April h=4; December's complete weighted validation proxy must
not regress; any monthly raw-RMSE regression above 0.02 fails; 2009-01 is
reported as transfer evidence, not used to rescue a recent failure. Stop after
one failed gate. No Test predictions, scoring file, or submission is authorized
by this specification.

## Reproducibility and safety checks

- Run status is `completed`, exit status 0, elapsed 144.07 s, maximum RSS
  1.87 GiB, and no swaps; ~59.35 GiB remained available at completion.
- `invariance_checks.json` confirms no row-shift join, no self-neighbour,
  longitude-wrap-safe geometry, no missing-month reversal bridge, target-blind
  legal-state builder, future-TWS perturbation invariance, and additive cell SSE.
- Six readable SVG maps (2015-01/02/06 and 2009-01/02/06) use the identical
  scale `[-3.049319, +3.049319]`; see `maps/map_manifest.json`.
- No Test rows, Test labels, model fits, prediction files, submissions, or
  uploads were produced.

## Thread policy benchmark

The repository default is now 12 LightGBM threads. A fixed cached 200,000 × 446
B3/98 inference workload was run sequentially at 12/24/48 threads with matching
OMP/BLAS/MKL/NumExpr settings. Median inference times were 0.1817/0.1201/0.0772
seconds with identical prediction checksums and exit status 0; `/usr/bin/time`
reported peak RSS below 1.1 GiB and no swaps. This supports explicit 24/48
throughput escalation for isolated work, but does not silently change the safe
12-thread baseline or justify concurrent high-thread jobs. Results are under
`/home/milli/zindi_drought_gcp/drought_runs/gcp_thread_benchmark_20260910/`.
