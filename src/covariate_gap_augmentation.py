"""Deterministic, target-blind Test-schedule covariate-gap augmentation.

The augmentation is deliberately limited to *training* covariate-history
exposure.  It never changes labels, sampled TWS anchors, horizons, or weights.
Each event receives a relative 12-month source-date pattern derived from the
unlabelled Test schedule; its current source row and legal anchor row are kept.
Regional summaries are rebuilt from the same masked date pattern rather than
post-hoc blanking feature columns.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from src.hydro_trajectory import trajectory_feature_names
from src.ml_features import build_hydro_gap_safe_feature_matrix
from src.neural_sequence import B3FeatureMaps
from src.regional_context import HYDRO_COLUMNS, attach_regional_features


HISTORY_MONTHS = 12


@dataclass(frozen=True)
class TestSchedulePatterns:
    """Unique Test-derived source-date masks, indexed by elapsed month backward."""

    masks: np.ndarray  # (pattern, offset 0=current .. 12=past), bool
    fingerprint: str


def _period_numbers(values: pd.Series) -> np.ndarray:
    return pd.to_datetime(values).dt.to_period("M").astype("int64").to_numpy(np.int64)


def derive_test_schedule_patterns(test: pd.DataFrame) -> TestSchedulePatterns:
    """Derive unique relative observation-date patterns from unlabelled Test rows."""
    if "time" not in test:
        raise ValueError("Test schedule requires time")
    months = np.unique(_period_numbers(test["time"]))
    if not len(months):
        raise AssertionError("Test schedule has no source months")
    observed = set(int(value) for value in months)
    masks = np.asarray(
        [[int(current - offset) in observed for offset in range(HISTORY_MONTHS + 1)] for current in months],
        dtype=bool,
    )
    masks[:, 0] = True
    unique = np.unique(masks, axis=0)
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(unique).tobytes())
    digest.update("|".join(str(int(value)) for value in months).encode("utf-8"))
    return TestSchedulePatterns(masks=unique, fingerprint=digest.hexdigest())


def _uniform(sample_ids: pd.Series, *, seed: int, salt: str) -> np.ndarray:
    key = sample_ids.astype(str) + f"__gap_{salt}_{int(seed)}"
    raw = pd.util.hash_pandas_object(key, index=False).to_numpy(dtype=np.uint64)
    return raw.astype(np.float64) / float(2**64)


def event_visibility(
    rows: pd.DataFrame,
    patterns: TestSchedulePatterns,
    *,
    recipe: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-event masks, selected pattern IDs, and sparse-assignment flags.

    D2's dense/sparse draw and sparse pattern draw use distinct deterministic
    hashes.  No target, covariate value, anchor value, or weight participates.
    """
    if recipe not in {"D0_dense", "D1_sparse", "D2_mixed"}:
        raise ValueError(f"unknown recipe {recipe}")
    required = {"sample_id", "source_date", "last_observed_date"}
    if missing := required.difference(rows.columns):
        raise ValueError(f"rows missing {sorted(missing)}")
    if not len(patterns.masks):
        raise AssertionError("no Test schedule patterns")
    n = len(rows)
    selected = np.floor(_uniform(rows.sample_id, seed=seed, salt="pattern") * len(patterns.masks)).astype(np.int16)
    sparse = np.ones(n, dtype=bool) if recipe == "D1_sparse" else (
        np.zeros(n, dtype=bool) if recipe == "D0_dense" else _uniform(rows.sample_id, seed=seed, salt="mixture") < .5
    )
    mask = np.ones((n, HISTORY_MONTHS + 1), dtype=bool)
    mask[sparse] = patterns.masks[selected[sparse]]
    # Current covariates and the legal anchor are explicitly retained.  This is
    # controlled augmentation, not a claim of exact row-level Test reproduction.
    offsets = _period_numbers(rows.source_date) - _period_numbers(rows.last_observed_date)
    eligible_anchor = (offsets >= 0) & (offsets <= HISTORY_MONTHS)
    mask[np.arange(n)[eligible_anchor], offsets[eligible_anchor]] = True
    if not mask[:, 0].all():
        raise AssertionError("augmentation removed a current source observation")
    return mask, selected, sparse


def _dense_regional_panel(source: pd.DataFrame, width: float) -> pd.DataFrame:
    x = source.loc[:, ["time", "lat", "lon", *HYDRO_COLUMNS]].copy()
    x["_period"] = _period_numbers(x.time)
    x["_lat_bin"] = np.floor(x.lat.astype(float) / width) * width
    x["_lon_bin"] = np.floor(x.lon.astype(float) / width) * width
    return x.groupby(["_period", "_lat_bin", "_lon_bin"], sort=False)[list(HYDRO_COLUMNS)].mean().reset_index()


def _lookup_history(
    rows: pd.DataFrame,
    source: pd.DataFrame,
    *,
    keys: list[str],
    masks: np.ndarray,
    force_anchor: bool,
) -> np.ndarray:
    """Look up one event-relative 0..12 history, then apply its visibility mask."""
    n = len(rows)
    left = rows.loc[:, keys].copy()
    left["_row"] = np.arange(n, dtype=np.int64)
    left["_period"] = _period_numbers(rows.source_date)
    expanded = left.loc[left.index.repeat(HISTORY_MONTHS + 1)].reset_index(drop=True)
    expanded["_offset"] = np.tile(np.arange(HISTORY_MONTHS + 1, dtype=np.int16), n)
    expanded["_lookup_period"] = expanded["_period"].to_numpy(np.int64) - expanded["_offset"].to_numpy(np.int64)
    right_columns = [*keys, *( ["time"] if "time" in source.columns else ["_period"] ), *HYDRO_COLUMNS]
    right = source.loc[:, right_columns].copy()
    if "time" in right:
        right["_lookup_period"] = _period_numbers(right.pop("time"))
    else:
        right = right.rename(columns={"_period": "_lookup_period"})
    merged = expanded.merge(right, how="left", on=[*keys, "_lookup_period"], validate="many_to_one", sort=False)
    values = merged.loc[:, HYDRO_COLUMNS].to_numpy(np.float64).reshape(n, HISTORY_MONTHS + 1, len(HYDRO_COLUMNS))
    visible = masks.copy()
    if force_anchor:
        offsets = _period_numbers(rows.source_date) - _period_numbers(rows.last_observed_date)
        eligible = (offsets >= 0) & (offsets <= HISTORY_MONTHS)
        visible[np.arange(n)[eligible], offsets[eligible]] = True
    return np.where(visible[:, :, None], values, np.nan)


def _trajectory_from_values(values: np.ndarray, *, prefix: str, dense: pd.DataFrame | None, sample_ids: pd.Series) -> pd.DataFrame:
    """Compute the established trajectory schema from event-relative raw values."""
    n = len(values)
    result: dict[str, np.ndarray] = {}
    offsets = np.arange(HISTORY_MONTHS + 1, dtype=np.float64)
    dense_indexed = None if dense is None else dense.assign(sample_id=dense.sample_id.astype(str)).set_index("sample_id")
    ids = sample_ids.astype(str).to_numpy()
    for variable_index, variable in enumerate(HYDRO_COLUMNS):
        raw = values[:, :, variable_index]
        finite = np.isfinite(raw)
        slopes: dict[int, np.ndarray] = {}
        for window in (3, 6):
            part, good = raw[:, :window], finite[:, :window]
            count = good.sum(axis=1).astype(np.float64)
            sum_y = np.nansum(part, axis=1)
            mean = np.divide(sum_y, count, out=np.full(n, np.nan), where=count > 0)
            centered = np.where(good, part - mean[:, None], 0.0)
            std = np.sqrt(np.divide(np.square(centered).sum(axis=1), count, out=np.full(n, np.nan), where=count > 0))
            minimum = np.where(good.any(axis=1), np.nanmin(part, axis=1), np.nan)
            maximum = np.where(good.any(axis=1), np.nanmax(part, axis=1), np.nan)
            x = -offsets[:window]
            sx = (good * x).sum(axis=1); sx2 = (good * x * x).sum(axis=1); sy = sum_y; sxy = np.nansum(np.where(good, part * x, np.nan), axis=1)
            denominator = count * sx2 - sx * sx
            slope = np.divide(count * sxy - sx * sy, denominator, out=np.full(n, np.nan), where=denominator > 0)
            stem = f"{prefix}{variable}_trail{window}_"
            result.update({f"{stem}mean": mean, f"{stem}std": std, f"{stem}min": minimum, f"{stem}max": maximum,
                           f"{stem}slope": slope, f"{stem}current_minus_mean": raw[:, 0] - mean,
                           f"{stem}count": count, f"{stem}coverage": count / float(window)})
            slopes[window] = slope
        prior = finite[:, 1:]
        has_prior = prior.any(axis=1)
        first = np.where(has_prior, prior.argmax(axis=1) + 1, -1)
        previous = np.full(n, np.nan); previous[has_prior] = raw[np.arange(n)[has_prior], first[has_prior]]
        elapsed = np.where(has_prior, first, np.nan).astype(np.float64)
        # Consecutive observed-run length truncates only when an augmentation or
        # real calendar gap appears; dense fallback restores runs longer than 12.
        broken = (~finite).argmax(axis=1)
        all_finite = finite.all(axis=1)
        run = broken.astype(np.float64); run[all_finite] = float(HISTORY_MONTHS + 1)
        age = np.where(finite.any(axis=1), finite.argmax(axis=1), np.nan).astype(np.float64)
        if dense_indexed is not None and all_finite.any():
            dense_values = dense_indexed.loc[ids[all_finite], f"{prefix}{variable}_observed_run_length"].to_numpy(np.float64)
            run[all_finite] = dense_values
        result.update({
            f"{prefix}{variable}_recent_change": raw[:, 0] - previous,
            f"{prefix}{variable}_recent_elapsed_months": elapsed,
            f"{prefix}{variable}_slope_change_3_6": slopes[3] - slopes[6],
            f"{prefix}{variable}_drying_recovery_sign": np.sign(raw[:, 0] - previous),
            f"{prefix}{variable}_observed_run_length": run,
            f"{prefix}{variable}_age_months": age,
        })
    return pd.DataFrame({name: result[name].astype(np.float32) for name in trajectory_feature_names(prefix)})


def _matrix_with_masks(
    ledger: pd.DataFrame,
    source: pd.DataFrame,
    structural: pd.DataFrame,
    maps: B3FeatureMaps,
    masks: np.ndarray,
) -> pd.DataFrame:
    base = build_hydro_gap_safe_feature_matrix(ledger, source, structural)
    current = attach_regional_features(ledger.sample_id, maps.regional)
    local_values = _lookup_history(ledger, source, keys=["lat", "lon"], masks=masks, force_anchor=True)
    local = _trajectory_from_values(local_values, prefix="local_", dense=maps.local_trajectory, sample_ids=ledger.sample_id)
    blocks: list[pd.DataFrame] = [base, current, local]
    for width in (5.0, 15.0):
        regional = _dense_regional_panel(source, width)
        regional_rows = ledger.copy()
        regional_rows["_lat_bin"] = np.floor(regional_rows.lat.astype(float) / width) * width
        regional_rows["_lon_bin"] = np.floor(regional_rows.lon.astype(float) / width) * width
        regional_values = _lookup_history(regional_rows, regional, keys=["_lat_bin", "_lon_bin"], masks=masks, force_anchor=False)
        tag = f"{int(width)}"
        # Dense map carries the exact established long-run fallback for each row.
        reg = _trajectory_from_values(regional_values, prefix=f"regional{tag}_", dense=maps.regional_trajectory, sample_ids=ledger.sample_id)
        blocks.append(reg)
        for variable in HYDRO_COLUMNS:
            for window in (3, 6):
                local_mean = local[f"local_{variable}_trail{window}_mean"].to_numpy(np.float32)
                local_slope = local[f"local_{variable}_trail{window}_slope"].to_numpy(np.float32)
                reg_mean = reg[f"regional{tag}_{variable}_trail{window}_mean"].to_numpy(np.float32)
                reg_slope = reg[f"regional{tag}_{variable}_trail{window}_slope"].to_numpy(np.float32)
                blocks.append(pd.DataFrame({
                    f"local_minus_reg{tag}_{variable}_trail{window}_mean": local_mean - reg_mean,
                    f"local_minus_reg{tag}_{variable}_trail{window}_slope": local_slope - reg_slope,
                    f"local_reg{tag}_{variable}_trail{window}_direction_disagree": np.where(np.isfinite(local_slope) & np.isfinite(reg_slope), (np.sign(local_slope) != np.sign(reg_slope)).astype(np.float32), np.nan),
                }))
    out = pd.concat([part.reset_index(drop=True) for part in blocks], axis=1)
    if out.columns.duplicated().any() or np.isinf(out.to_numpy(np.float32)).any():
        raise AssertionError("augmented B3 matrix is invalid")
    return out.astype(np.float32)


def build_augmented_b3_matrix(
    ledger: pd.DataFrame,
    source: pd.DataFrame,
    structural: pd.DataFrame,
    maps: B3FeatureMaps,
    patterns: TestSchedulePatterns,
    *,
    recipe: str,
    seed: int,
    chunk_rows: int = 60_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build B3 context with event-relative augmented history exposure.

    Feature construction chunks rows only for memory.  Masks are assigned once
    before chunking, so IDs receive identical target-blind treatment regardless
    of chunk boundary.
    """
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    masks, pattern_ids, sparse = event_visibility(ledger, patterns, recipe=recipe, seed=seed)
    parts = [
        _matrix_with_masks(ledger.iloc[first:last].reset_index(drop=True), source, structural, maps, masks[first:last])
        for first in range(0, len(ledger), chunk_rows)
        for last in [min(len(ledger), first + chunk_rows)]
    ]
    out = pd.concat(parts, ignore_index=True)
    metadata = {
        "recipe": recipe, "augmentation_seed": int(seed), "pattern_fingerprint": patterns.fingerprint,
        "pattern_count": int(len(patterns.masks)), "sparse_rows": int(sparse.sum()),
        "dense_rows": int((~sparse).sum()), "selected_pattern_counts": {str(key): int(value) for key, value in pd.Series(pattern_ids[sparse]).value_counts().sort_index().items()},
        "forced_anchor_retained_rows": int(((~masks[:, 1:]).any(axis=1) & (np.isfinite(_period_numbers(ledger.source_date) - _period_numbers(ledger.last_observed_date)))).sum()),
    }
    return out, metadata
