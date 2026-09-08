# Availability-faithful compact neural sequence comparison

Status: predeclared before any MLP/GRU/TCN fit.  This plan answers a narrow
question: can a compact non-linear static or temporal model improve the verified
B3/98 hydrological-history baseline when both use the same Test-like historical
observation schedule?  It does not authorise Test prediction or submission.

## Information contract and paired baseline repair

Historical template replays previously built their trajectory maps from dense
Train covariates throughout the transplanted Test window.  That is not the
competition information set: a sparse Test supplies covariates only on its own
row schedule.  `src.availability.build_replay_observation_view` is therefore
the shared contract for every run below.  It retains all target-blind source rows
before a replay begins and exactly the replay ledger IDs thereafter; it blanks
hidden current-window TWS in the structural panel.  It never accepts a target.

Before neural selection, rerun only the affected paired B0 and B3/98 baselines
on the two inner origins (`2003-04`, `2004-04`) with that view.  After selection,
rerun B3/98 on all four predeclared outer origins (`2007-09`, `2009-01`,
`2014-04`, `2014-12`) with the same view.  These reruns replace neither the
original results nor their artifacts; they establish the fair comparator for
this experiment.

## Shared supervised task and preprocessing

All candidates predict the signed direct delta
`target - last_observed_TWS`; prediction is reconstructed by adding the legal
anchor.  Training labels are from rows whose target month is strictly before the
replay origin.  The fixed, horizon-rebalanced training-row builder is retained.
To make the compact GPU comparison practical on the 2.15M-row source panel, a
single deterministic, training-only stratified cap of 252,000 rows (up to 36,000
per h=1..7; data selection seed `20260908`) is used for every neural fit at an
origin.  Validation rows are never capped.

Each fit normalizes only its selected training prefix.  Static and sequence
normalizer arrays, schemas, selected IDs, model checkpoint, epoch curves, and a
reload-prediction equivalence check are exported with the fit.  Non-finite input,
loss, gradient, or prediction is a hard failure.  A 2,048-row train-only subset
must lower its weighted training loss during a 12-epoch learnability check before
the corresponding full fit is accepted.

The static context is the availability-faithful B3 block: legal current/anchor
hydrology, anchor age, h, latitude/longitude, calendar terms, contemporaneous
5/15-degree context, and causal B3 local/regional trajectory summaries.  It has
no hidden historical TWS, future covariates, target, or unobserved dense-window
row.  Temporal inputs use a real monthly grid ending at the source month:

- local, 5-degree regional, and 15-degree regional values for each of five
  hydrology variables;
- an observation mask and causal per-variable age for each of those 15 values;
- calendar sine/cosine and actual elapsed-month position for every slot.

No missing month is compressed.  A `span=6` or `span=12` model sees only that
many calendar slots.  The static MLP receives the B3 block only, serving as the
non-sequential capacity control.

## Fixed candidates and training recipe

All neural models use `AdamW(lr=0.002, weight_decay=0.0001)`, weighted MSE with
the existing horizon weights renormalized to mean one, batch size 1024, gradient
norm clip 5, dropout 0.10, and a maximum of 30 epochs on CUDA when available.
The random model seeds are `20260908` and `20260909`; deterministic data order is
fixed separately.  There is no hyperparameter search.

On Kaggle SSH launches, use `scripts/run_kaggle_neural_sequence.sh` rather than
calling the Python runner directly. The wrapper exports `/opt/bin` and the
NVIDIA/CUDA library paths before Python starts, because Kaggle recreates the
custom `kaggle` SSH user after a full kernel restart and does not preserve those
interactive-shell exports. The runner records the actual selected device for
each fit; it must never claim CUDA merely because the wrapper was invoked.

- **MLP control:** B3 static branch `446 -> 192 -> 96 -> 1` (the input width is
  inferred and recorded, not assumed).
- **GRU:** one 64-unit causal GRU over the shared channels; B3 static branch
  `-> 96`; concatenate, then `160 -> 96 -> 1`.
- **TCN:** two causal residual blocks with 64 channels, kernel 3, dilations 1 and
  2; same B3 static branch and fusion head as GRU.  Sequence pooling is the final
  source-month state, never a future/bi-directional operation.

There is one predeclared repair allowance: if the learnability check fails due to
a demonstrated numerical/shape defect, correct that defect once, record it in
the manifest, rerun the affected check, and do not alter model capacity,
optimizer, data split, or selection metric.  A weak flattened-tree control is
not a reason to omit any listed neural candidate.

## Inner selection, outer evaluation, and ablation

Inner origins are exactly `2003-04` and `2004-04`.  The planned inner fit count
is 4 static MLP fits (2 origins x 2 seeds) and 16 temporal fits (GRU/TCN x
span 6/12 x 2 origins x 2 seeds).  Curves are retained through epoch 30.  The
selected recipe is the architecture/span/epoch with the lowest equally weighted
mean raw RMSE across the two origins and two seeds; the B3 baseline is not part
of tuning.  The chosen epoch is frozen before outer fitting.

Only that recipe is then evaluated at all four outer origins, with both
predeclared seeds averaged per exact OOF ID.  Outer results are robustness
evidence and cannot change the selection.  A third seed (`20260910`) runs only
on the selected recipe at both inner origins as a stability check, not a new
selection opportunity.

The selected temporal recipe also receives an inner history ablation at both
seeds and both inner origins.  It replaces past sequence slots with missing
values/masks and retains the current slot, its current masks/ages/calendar
channels, and the unchanged declared B3 static context.  Thus it tests marginal
value from the learned raw past sequence, not a different static feature set.

Every OOF output retains exact IDs, origin, h, anchor age, calendar block, and
5-degree geography.  Reports include raw and per-horizon RMSE, coverage and
anchor-age strata, seed spread, runtime, peak memory, residual correlation with
availability-faithful B3, and the predeclared equal B3/neural average.  A
calendar-block and 5-degree block paired bootstrap reports uncertainty for the
neural-vs-B3 and blend-vs-B3 deltas.  No result is called confirmed without an
unused verification set.
