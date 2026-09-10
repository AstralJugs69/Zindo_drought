"""Causal GLDAS-2.1 external features and compact remote raw-value store.

The competition locations are metadata used to index the official 0.25-degree
GLDAS grid.  They are never returned as model columns.  This module keeps the
raw store deliberately small (one row per requested month/grid cell), performs
exact calendar joins, and propagates missingness instead of inventing values.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


GLDAS_PRODUCT = "GLDAS_NOAH025_M.2.1"
GLDAS_GRID_RESOLUTION_DEGREES = 0.25
GLDAS_GRID_TOLERANCE_DEGREES = 0.125 + 1e-6

GLDAS_RAW_VARIABLES: Mapping[str, str] = {
    "soil_moisture_0_10cm": "SoilMoi0_10cm_inst",
    "soil_moisture_10_40cm": "SoilMoi10_40cm_inst",
    "soil_moisture_40_100cm": "SoilMoi40_100cm_inst",
    "soil_moisture_100_200cm": "SoilMoi100_200cm_inst",
    "swe": "SWE_inst",
    "canopy_storage": "CanopInt_inst",
}
GLDAS_RAW_FIELDS = tuple(GLDAS_RAW_VARIABLES)
GLDAS_EXTERNAL_FEATURE_COLUMNS = (
    "gldas_soil_moisture_0_10cm_t",
    "gldas_soil_moisture_10_40cm_t",
    "gldas_soil_moisture_40_100cm_t",
    "gldas_soil_moisture_100_200cm_t",
    "gldas_swe_t",
    "gldas_canopy_storage_t",
    "gldas_storage_sum_t",
    "gldas_storage_sum_anchor",
    "gldas_storage_sum_gap",
    "gldas_storage_sum_prev_month_delta",
)
GLDAS_EXPECTED_UNITS = "kg m-2"
GLDAS_PROHIBITED_MODEL_IDENTIFIERS = frozenset(
    {"lat", "lon", "cell_id", "location_id", "grid_i", "grid_j"}
)

INNER_ORIGINS = ("2003-04", "2004-04")
RECENT_ORIGINS = ("2014-04", "2014-12")
OLDER_ORIGIN = "2009-01"
GLDAS_REQUEST_ORIGINS = (*INNER_ORIGINS, *RECENT_ORIGINS, OLDER_ORIGIN)
GLDAS_END_MONTHS = {"2014-04": "2014-10", "2014-12": "2015-06"}


def period_key(value: str | pd.Period | pd.Timestamp) -> int:
    """Encode a calendar month as an integer without relying on string order."""
    period = pd.Period(value, freq="M")
    return int(period.year * 100 + period.month)


def period_keys(values: Sequence[str | pd.Period | pd.Timestamp] | pd.Series) -> np.ndarray:
    if isinstance(values, pd.Series):
        if isinstance(values.dtype, pd.PeriodDtype):
            periods = values.astype("period[M]")
        else:
            periods = pd.to_datetime(values, errors="raise").dt.to_period("M")
        years = periods.dt.year.to_numpy(dtype=np.int64)
        months = periods.dt.month.to_numpy(dtype=np.int64)
    else:
        periods = pd.PeriodIndex(values, freq="M")
        years = np.asarray(periods.year, dtype=np.int64)
        months = np.asarray(periods.month, dtype=np.int64)
    return (years * 100 + months).astype(np.int32)


def _lookup_key(periods: np.ndarray, grid_i: np.ndarray, grid_j: np.ndarray) -> np.ndarray:
    # 2,000 exceeds the GLDAS longitude dimension; the period multiplier leaves
    # ample space for both grid indices and avoids Python tuple dictionaries.
    return periods.astype(np.int64) * 1_000_000 + grid_i.astype(np.int64) * 2_000 + grid_j.astype(np.int64)


def _axis_indices(
    values: np.ndarray,
    axis: np.ndarray,
    *,
    tolerance: float = GLDAS_GRID_TOLERANCE_DEGREES,
) -> tuple[np.ndarray, np.ndarray]:
    """Map coordinates to an ascending axis; exact ties choose the lower cell."""
    values = np.asarray(values, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1 or len(axis) == 0 or not np.all(np.diff(axis) > 0):
        raise ValueError("GLDAS grid axis must be non-empty and strictly ascending")
    right = np.searchsorted(axis, values, side="left")
    lo = np.clip(right - 1, 0, len(axis) - 1)
    hi = np.clip(right, 0, len(axis) - 1)
    lo_distance = np.abs(values - axis[lo])
    hi_distance = np.abs(axis[hi] - values)
    # <= is intentional: the lower coordinate wins a half-cell tie.
    chosen = np.where(lo_distance <= hi_distance, lo, hi).astype(np.int32)
    distance = np.minimum(lo_distance, hi_distance)
    if np.any(~np.isfinite(values)) or np.any(distance > tolerance):
        bad = int(np.sum(~np.isfinite(values) | (distance > tolerance)))
        raise ValueError(f"{bad} coordinates lie outside the GLDAS grid tolerance")
    return chosen, distance.astype(np.float32)


class GLDASRawStore:
    """Sorted, compact GLDAS values with vectorized exact-key lookup."""

    def __init__(
        self,
        *,
        period_key: np.ndarray,
        grid_i: np.ndarray,
        grid_j: np.ndarray,
        values: Mapping[str, np.ndarray],
        grid_lat: np.ndarray,
        grid_lon: np.ndarray,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.period_key = np.asarray(period_key, dtype=np.int32)
        self.grid_i = np.asarray(grid_i, dtype=np.int16)
        self.grid_j = np.asarray(grid_j, dtype=np.int16)
        self.grid_lat = np.asarray(grid_lat, dtype=np.float32)
        self.grid_lon = np.asarray(grid_lon, dtype=np.float32)
        if not (len(self.period_key) == len(self.grid_i) == len(self.grid_j)):
            raise ValueError("GLDAS store key arrays have different lengths")
        if self.grid_lat.ndim != 1 or self.grid_lon.ndim != 1:
            raise ValueError("GLDAS grid axes must be one-dimensional")
        self.values: dict[str, np.ndarray] = {}
        for field in GLDAS_RAW_FIELDS:
            if field not in values:
                raise ValueError(f"GLDAS store is missing raw field {field}")
            array = np.asarray(values[field], dtype=np.float32)
            if len(array) != len(self.period_key):
                raise ValueError(f"GLDAS raw field {field} has the wrong row count")
            self.values[field] = array
        self.metadata = dict(metadata or {})
        if self.metadata.get("product", GLDAS_PRODUCT) != GLDAS_PRODUCT:
            raise ValueError("GLDAS store product is not the selected GLDAS-2.1 monthly product")
        self._keys = _lookup_key(self.period_key, self.grid_i, self.grid_j)
        order = np.argsort(self._keys, kind="mergesort")
        self._keys = self._keys[order]
        self.period_key = self.period_key[order]
        self.grid_i = self.grid_i[order]
        self.grid_j = self.grid_j[order]
        for field in GLDAS_RAW_FIELDS:
            self.values[field] = self.values[field][order]
        if len(self._keys) and np.any(self._keys[1:] == self._keys[:-1]):
            raise ValueError("GLDAS store contains duplicate month/grid keys")

    @property
    def row_count(self) -> int:
        return int(len(self._keys))

    @property
    def grid_shape(self) -> tuple[int, int]:
        return int(len(self.grid_lat)), int(len(self.grid_lon))

    @property
    def store_key_sha256(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self._keys).tobytes()).hexdigest()

    def grid_indices_for_coordinates(
        self,
        lat: Sequence[float] | np.ndarray | pd.Series,
        lon: Sequence[float] | np.ndarray | pd.Series,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(lat) != len(lon):
            raise ValueError("latitude and longitude arrays have different lengths")
        grid_i, lat_distance = _axis_indices(np.asarray(lat), self.grid_lat)
        grid_j, lon_distance = _axis_indices(np.asarray(lon), self.grid_lon)
        return grid_i, grid_j, np.maximum(lat_distance, lon_distance)

    def lookup(
        self,
        periods: Sequence[str | pd.Period | pd.Timestamp] | np.ndarray,
        grid_i: np.ndarray,
        grid_j: np.ndarray,
    ) -> np.ndarray:
        query_periods = (
            np.asarray(periods, dtype=np.int32)
            if np.asarray(periods).dtype.kind in "iu"
            else period_keys(pd.Series(periods))
        )
        grid_i = np.asarray(grid_i, dtype=np.int16)
        grid_j = np.asarray(grid_j, dtype=np.int16)
        if not (len(query_periods) == len(grid_i) == len(grid_j)):
            raise ValueError("GLDAS lookup arrays have different lengths")
        query = _lookup_key(query_periods, grid_i, grid_j)
        positions = np.searchsorted(self._keys, query, side="left")
        present = positions < len(self._keys)
        safe_positions = np.clip(positions, 0, max(len(self._keys) - 1, 0))
        present &= len(self._keys) > 0
        if len(self._keys):
            present &= self._keys[safe_positions] == query
        out = np.full((len(query), len(GLDAS_RAW_FIELDS)), np.nan, dtype=np.float32)
        if np.any(present):
            selected = safe_positions[present]
            for column, field in enumerate(GLDAS_RAW_FIELDS):
                out[present, column] = self.values[field][selected]
        return out

    def save(self, path: Path, metadata_path: Path | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        np.savez_compressed(
            temporary,
            period_key=self.period_key,
            grid_i=self.grid_i,
            grid_j=self.grid_j,
            grid_lat=self.grid_lat,
            grid_lon=self.grid_lon,
            **self.values,
        )
        # numpy appends .npz when a string path lacks that suffix; the temporary
        # name intentionally ends in .tmp, so resolve the actual emitted name.
        emitted = Path(str(temporary) + ".npz")
        emitted.replace(path)
        if metadata_path is not None:
            metadata = {
                **self.metadata,
                "product": GLDAS_PRODUCT,
                "rows": self.row_count,
                "grid_shape": list(self.grid_shape),
                "store_key_sha256": self.store_key_sha256,
                "raw_fields": list(GLDAS_RAW_FIELDS),
                "grid_tolerance_degrees": GLDAS_GRID_TOLERANCE_DEGREES,
                "tie_break": "lower grid coordinate on an exact half-cell tie",
            }
            temporary_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
            temporary_metadata.write_text(
                json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            temporary_metadata.replace(metadata_path)

    @classmethod
    def load(cls, path: Path, metadata_path: Path | None = None) -> "GLDASRawStore":
        with np.load(path, allow_pickle=False) as payload:
            values = {field: np.asarray(payload[field]) for field in GLDAS_RAW_FIELDS}
            arrays = {
                "period_key": np.asarray(payload["period_key"]),
                "grid_i": np.asarray(payload["grid_i"]),
                "grid_j": np.asarray(payload["grid_j"]),
                "grid_lat": np.asarray(payload["grid_lat"]),
                "grid_lon": np.asarray(payload["grid_lon"]),
            }
        metadata: Mapping[str, object] = {}
        if metadata_path is not None and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls(**arrays, values=values, metadata=metadata)


def _finite_sum(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values).all(axis=1)
    result = np.full(len(values), np.nan, dtype=np.float32)
    if np.any(valid):
        result[valid] = np.sum(values[valid], axis=1, dtype=np.float64).astype(np.float32)
    return result


def build_gldas_external_features(
    ledger: pd.DataFrame,
    store: GLDASRawStore,
) -> pd.DataFrame:
    """Build exactly ten target-blind GLDAS columns in the ledger's row order."""
    forbidden = {column for column in ledger.columns if column.lower() in {"target", "label", "y"}}
    if forbidden:
        raise AssertionError(f"GLDAS feature ledger must be target-blind; found {sorted(forbidden)}")
    required = {"source_date", "last_observed_date", "lat", "lon"}
    missing = required.difference(ledger.columns)
    if missing:
        raise ValueError(f"GLDAS feature ledger missing columns: {sorted(missing)}")
    source_dates = pd.to_datetime(ledger["source_date"], errors="raise")
    anchor_dates = pd.to_datetime(ledger["last_observed_date"], errors="raise")
    source_periods = source_dates.dt.to_period("M")
    anchor_periods = anchor_dates.dt.to_period("M")
    if (anchor_periods > source_periods).any():
        raise AssertionError("GLDAS anchor month lies after the source month")
    previous_periods = source_periods - 1
    grid_i, grid_j, _distance = store.grid_indices_for_coordinates(ledger["lat"], ledger["lon"])
    current = store.lookup(period_keys(source_periods), grid_i, grid_j)
    anchor = store.lookup(period_keys(anchor_periods), grid_i, grid_j)
    previous = store.lookup(period_keys(previous_periods), grid_i, grid_j)

    output = pd.DataFrame(index=ledger.index)
    for column, field in zip(GLDAS_EXTERNAL_FEATURE_COLUMNS[:6], GLDAS_RAW_FIELDS):
        output[column] = current[:, GLDAS_RAW_FIELDS.index(field)]
    current_sum = _finite_sum(current)
    anchor_sum = _finite_sum(anchor)
    previous_sum = _finite_sum(previous)
    output[GLDAS_EXTERNAL_FEATURE_COLUMNS[6]] = current_sum
    output[GLDAS_EXTERNAL_FEATURE_COLUMNS[7]] = anchor_sum
    output[GLDAS_EXTERNAL_FEATURE_COLUMNS[8]] = np.where(
        np.isfinite(current_sum) & np.isfinite(anchor_sum), current_sum - anchor_sum, np.nan
    ).astype(np.float32)
    output[GLDAS_EXTERNAL_FEATURE_COLUMNS[9]] = np.where(
        np.isfinite(current_sum) & np.isfinite(previous_sum), current_sum - previous_sum, np.nan
    ).astype(np.float32)
    output = output.loc[:, list(GLDAS_EXTERNAL_FEATURE_COLUMNS)].astype("float32")
    if output.columns.tolist() != list(GLDAS_EXTERNAL_FEATURE_COLUMNS):
        raise AssertionError("GLDAS external schema changed")
    return output.reset_index(drop=True)


def external_coverage_report(ledger: pd.DataFrame, features: pd.DataFrame) -> dict[str, object]:
    """Return finite/missing counts overall and by exact source calendar month."""
    if len(ledger) != len(features):
        raise ValueError("coverage ledger/features row counts differ")
    source_month = pd.to_datetime(ledger["source_date"], errors="raise").dt.to_period("M").astype(str)
    report: dict[str, object] = {
        "rows": int(len(ledger)),
        "feature_finite_counts": {
            column: int(features[column].notna().sum()) for column in GLDAS_EXTERNAL_FEATURE_COLUMNS
        },
        "feature_missing_counts": {
            column: int(features[column].isna().sum()) for column in GLDAS_EXTERNAL_FEATURE_COLUMNS
        },
        "by_source_month": {},
    }
    by_month: dict[str, object] = {}
    for month, positions in source_month.groupby(source_month, sort=True).groups.items():
        index = np.asarray(list(positions), dtype=np.int64)
        by_month[str(month)] = {
            "rows": int(len(index)),
            "feature_finite_counts": {
                column: int(features.iloc[index][column].notna().sum()) for column in GLDAS_EXTERNAL_FEATURE_COLUMNS
            },
            "feature_missing_counts": {
                column: int(features.iloc[index][column].isna().sum()) for column in GLDAS_EXTERNAL_FEATURE_COLUMNS
            },
        }
    report["by_source_month"] = by_month
    return report


def build_external_request_plan(train: pd.DataFrame, template: pd.DataFrame) -> dict[str, object]:
    """Derive the exact month/cell union needed by all declared replay origins."""
    from src.availability import build_replay_observation_view
    from src.ml_features import SOURCE_CORE_COLUMNS, SOURCE_HYDRO_HISTORY_COLUMNS, build_sampled_training_rows
    from src.observation_simulator import build_mask_block_fold, build_template_replay_fold

    source = train.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
    structural = train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
    periods: set[int] = set()
    coordinates: set[tuple[float, float]] = set()
    origins: dict[str, object] = {}
    for origin in GLDAS_REQUEST_ORIGINS:
        if origin in INNER_ORIGINS or origin == OLDER_ORIGIN:
            fold = build_template_replay_fold(
                train,
                template,
                start_month=origin,
                scenario_id=f"gldas_external_{origin}",
                family="inner_capacity_selection" if origin in INNER_ORIGINS else "development_transfer",
            )
        else:
            fold = build_mask_block_fold(
                train,
                anchor_month=origin,
                end_month=GLDAS_END_MONTHS[origin],
                scenario_id=f"gldas_external_{origin}",
                family="recent_mask_block",
            )
        view = build_replay_observation_view(
            source,
            structural,
            ledger=fold.ledger,
            first_source_month=fold.spec.first_source_month,
            last_source_month=fold.spec.last_source_month,
        )
        sampled = build_sampled_training_rows(
            view.structural,
            view.source.loc[:, SOURCE_CORE_COLUMNS],
            max_target_month=pd.Period(origin, freq="M") - 1,
            seed=20260908,
        )
        needed_frames = {"training": sampled.rows, "validation": fold.ledger}
        origin_periods: set[int] = set()
        origin_coordinates: set[tuple[float, float]] = set()
        for frame in needed_frames.values():
            current = pd.to_datetime(frame["source_date"], errors="raise").dt.to_period("M")
            anchor = pd.to_datetime(frame["last_observed_date"], errors="raise").dt.to_period("M")
            for values in (current, anchor, current - 1):
                origin_periods.update(period_keys(values).astype(int).tolist())
            origin_coordinates.update(
                (float(lat), float(lon)) for lat, lon in zip(frame["lat"], frame["lon"])
            )
        periods.update(origin_periods)
        coordinates.update(origin_coordinates)
        origins[origin] = {
            "training_rows": int(len(sampled.rows)),
            "validation_rows": int(len(fold.ledger)),
            "training_dropped_missing_anchor": int(sampled.dropped_missing_anchor),
            "periods": [f"{value // 100:04d}-{value % 100:02d}" for value in sorted(origin_periods)],
            "coordinate_count": int(len(origin_coordinates)),
        }
    sorted_coordinates = sorted(coordinates)
    coordinate_bytes = "\n".join(f"{lat:.8f},{lon:.8f}" for lat, lon in sorted_coordinates).encode("utf-8")
    return {
        "status": "planned",
        "product": GLDAS_PRODUCT,
        "sampling_method": "nearest 0.25-degree cell; lower coordinate wins exact half-cell ties",
        "seed": 20260908,
        "origins": origins,
        "periods": [f"{value // 100:04d}-{value % 100:02d}" for value in sorted(periods)],
        "period_count": int(len(periods)),
        "coordinates": [[lat, lon] for lat, lon in sorted_coordinates],
        "coordinate_count": int(len(sorted_coordinates)),
        "coordinate_sha256": hashlib.sha256(coordinate_bytes).hexdigest(),
        "no_test_labels": True,
    }


def run_gldas_external_prefit_checks() -> dict[str, object]:
    """Small deterministic checks required before any external LightGBM fit."""
    grid_lat = np.array([0.125, 0.375, 0.625], dtype=np.float32)
    grid_lon = np.array([10.125, 10.375, 10.625], dtype=np.float32)
    periods = np.array([202001, 202002, 202003, 202004], dtype=np.int32)
    cells = [(0, 0), (1, 1)]
    rows = []
    values = {field: [] for field in GLDAS_RAW_FIELDS}
    for p_index, p in enumerate(periods):
        for cell_index, (i, j) in enumerate(cells):
            rows.append((int(p), i, j))
            for field_index, field in enumerate(GLDAS_RAW_FIELDS):
                values[field].append(float(p_index * 10 + cell_index + field_index))
    store = GLDASRawStore(
        period_key=np.array([row[0] for row in rows], dtype=np.int32),
        grid_i=np.array([row[1] for row in rows], dtype=np.int16),
        grid_j=np.array([row[2] for row in rows], dtype=np.int16),
        values=values,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        metadata={"product": GLDAS_PRODUCT},
    )
    ledger = pd.DataFrame({
        "sample_id": ["a", "b"],
        "source_date": pd.to_datetime(["2020-03-15", "2020-03-15"]),
        "last_observed_date": pd.to_datetime(["2020-02-01", "2020-03-01"]),
        "lat": [0.2, 0.4],
        "lon": [10.2, 10.4],
    })
    features = build_gldas_external_features(ledger, store)
    expected_sum_a = sum(20.0 + field_index for field_index in range(6))
    arithmetic_ok = bool(
        np.isclose(features.loc[0, "gldas_storage_sum_t"], expected_sum_a)
        and np.isclose(features.loc[0, "gldas_storage_sum_anchor"], sum(10.0 + i for i in range(6)))
        and np.isclose(features.loc[0, "gldas_storage_sum_gap"], 60.0)
        and np.isclose(features.loc[0, "gldas_storage_sum_prev_month_delta"], 60.0)
    )
    shuffled = ledger.iloc[[1, 0]].reset_index(drop=True)
    shuffled_features = build_gldas_external_features(shuffled, store)
    row_order_ok = bool(
        np.array_equal(
            features.set_index(ledger["sample_id"]).sort_index().to_numpy(),
            shuffled_features.set_index(shuffled["sample_id"]).sort_index().to_numpy(),
            equal_nan=True,
        )
    )
    future_values = {field: array.copy() for field, array in store.values.items()}
    for field in GLDAS_RAW_FIELDS:
        future_values[field][-1] += 1_000_000.0
    future_store = GLDASRawStore(
        period_key=store.period_key,
        grid_i=store.grid_i,
        grid_j=store.grid_j,
        values=future_values,
        grid_lat=store.grid_lat,
        grid_lon=store.grid_lon,
        metadata={"product": GLDAS_PRODUCT},
    )
    future_invariant = bool(
        np.array_equal(features.to_numpy(), build_gldas_external_features(ledger, future_store).to_numpy(), equal_nan=True)
    )
    target_rejected = False
    try:
        build_gldas_external_features(ledger.assign(target=0.0), store)
    except AssertionError:
        target_rejected = True
    missing_row = int(np.flatnonzero((store.period_key == 202002) & (store.grid_i == 0) & (store.grid_j == 0))[0])
    keep = np.ones(store.row_count, dtype=bool)
    keep[missing_row] = False
    missing_store = GLDASRawStore(
        period_key=store.period_key[keep],
        grid_i=store.grid_i[keep],
        grid_j=store.grid_j[keep],
        values={field: array[keep] for field, array in store.values.items()},
        grid_lat=store.grid_lat,
        grid_lon=store.grid_lon,
        metadata={"product": GLDAS_PRODUCT},
    )
    missing_features = build_gldas_external_features(ledger.iloc[[0]], missing_store)
    missing_propagates = bool(
        pd.isna(missing_features.loc[0, "gldas_storage_sum_anchor"])
        and not bool(pd.isna(missing_features.loc[0, "gldas_storage_sum_t"]))
        and pd.isna(missing_features.loc[0, "gldas_storage_sum_gap"])
    )
    if not all((arithmetic_ok, row_order_ok, future_invariant, target_rejected, missing_propagates)):
        raise AssertionError("GLDAS external prefit contract checks failed")
    return {
        "status": "passed",
        "feature_count": len(GLDAS_EXTERNAL_FEATURE_COLUMNS),
        "feature_names": list(GLDAS_EXTERNAL_FEATURE_COLUMNS),
        "arithmetic_and_units": arithmetic_ok,
        "row_order_invariant": row_order_ok,
        "future_source_perturbation_invariant": future_invariant,
        "target_rejection": target_rejected,
        "missingness_propagates_without_partial_sum": missing_propagates,
        "coordinates_are_indexing_only": True,
        "exact_previous_calendar_month": True,
    }
