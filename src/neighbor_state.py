"""Causal nearest-neighbour TWS state features.

The feature builder in this module is deliberately separate from the oracle
spatial diagnostics.  It receives only a structural panel whose ``TWS_t``
values already encode the event's legal visibility.  Each event supplies its
own focal ``last_observed_date`` cutoff; values after that cutoff are never
consulted, even when they are present elsewhere in the panel.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd


NEIGHBOR_K = 8
NEIGHBOR_RADIUS_KM = 500.0
NEIGHBOR_MIN_SUPPORT = 4
NEIGHBOR_FEATURE_COLUMNS = [
    "neighbor_tws_mean",
    "neighbor_tws_minus_focal",
    "neighbor_tws_count",
    "neighbor_tws_median_age",
]
EARTH_RADIUS_KM = 6371.0088


def _month_number(values: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(values)
    return (dt.dt.year.to_numpy(dtype=np.int32) * 12 + dt.dt.month.to_numpy(dtype=np.int32))


def _assert_target_blind(frame: pd.DataFrame, name: str) -> None:
    forbidden = {column for column in frame.columns if column.lower() in {"target", "y", "label"}}
    if forbidden:
        raise AssertionError(f"{name} must be target-blind; found {sorted(forbidden)}")


def _unit_vectors(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(lat.astype(np.float64))
    lon_rad = np.deg2rad(lon.astype(np.float64))
    cos_lat = np.cos(lat_rad)
    return np.column_stack((
        cos_lat * np.cos(lon_rad),
        cos_lat * np.sin(lon_rad),
        np.sin(lat_rad),
    ))


@dataclass(frozen=True)
class NeighborFeatureResult:
    """Features plus auditable support/causality diagnostics."""

    features: pd.DataFrame
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class NeighborStateIndex:
    """Fixed geometry and as-of TWS state for one legal structural panel."""

    locations: pd.DataFrame
    periods: np.ndarray
    neighbor_ids: np.ndarray
    neighbor_distances_km: np.ndarray
    latest_values: np.ndarray
    latest_periods: np.ndarray
    _location_index: pd.MultiIndex

    @property
    def location_count(self) -> int:
        return int(len(self.locations))

    def build_features(self, events: pd.DataFrame) -> NeighborFeatureResult:
        """Build four causal features for events in their original row order.

        ``events`` must contain ``source_date`` (used only for the age feature),
        ``last_observed_date`` (the as-of cutoff), focal coordinates, and the
        existing focal ``last_observed_TWS``.  The returned values are native
        float32 columns with missing mean/difference/median-age values whenever
        fewer than four of the fixed eight neighbours have a legal finite state.
        """
        _assert_target_blind(events, "events")
        required = {
            "source_date", "last_observed_date", "lat", "lon", "last_observed_TWS",
        }
        missing = required.difference(events.columns)
        if missing:
            raise ValueError(f"events missing required columns: {sorted(missing)}")
        if events.empty:
            raise ValueError("events cannot be empty")

        left = pd.MultiIndex.from_frame(events.loc[:, ["lat", "lon"]])
        location_ids = self._location_index.get_indexer(left)
        if (location_ids < 0).any():
            raise AssertionError("event contains a location absent from the structural panel")

        source_period = _month_number(events["source_date"])
        cutoff_period = _month_number(events["last_observed_date"])
        if (cutoff_period > source_period).any():
            raise AssertionError("focal cutoff cannot be after the event source month")
        if events["last_observed_TWS"].isna().any():
            raise AssertionError("events contain a missing focal TWS anchor")

        # Search to the right, then step back one column, so a cutoff that falls
        # between observed source months still receives the latest legal month.
        cutoff_column = np.searchsorted(self.periods, cutoff_period, side="right") - 1
        has_prior_period = cutoff_column >= 0
        safe_column = np.clip(cutoff_column, 0, len(self.periods) - 1)
        selected_neighbors = self.neighbor_ids[location_ids]
        safe_neighbors = np.where(selected_neighbors >= 0, selected_neighbors, 0)

        values = self.latest_values[safe_neighbors, safe_column[:, None]].copy()
        observation_periods = self.latest_periods[safe_neighbors, safe_column[:, None]].copy()
        valid = np.isfinite(values)
        valid &= selected_neighbors >= 0
        valid &= has_prior_period[:, None]
        values[~valid] = np.nan
        observation_periods[~valid] = -1

        # These checks are the central causal contract.  They also catch an
        # accidental source-date lookup if this function is changed later.
        cutoff_matrix = np.broadcast_to(cutoff_period[:, None], valid.shape)
        source_matrix = np.broadcast_to(source_period[:, None], valid.shape)
        if valid.any() and (observation_periods[valid] > cutoff_matrix[valid]).any():
            raise AssertionError("a neighbour observation is newer than the focal cutoff")
        if valid.any() and (observation_periods[valid] > source_matrix[valid]).any():
            raise AssertionError("a neighbour observation is newer than the event source month")

        count = valid.sum(axis=1).astype(np.float32)
        supported = count >= NEIGHBOR_MIN_SUPPORT
        sums = np.nansum(values, axis=1, dtype=np.float64)
        means = np.full(len(events), np.nan, dtype=np.float32)
        means[supported] = (sums[supported] / count[supported]).astype(np.float32)
        focal = events["last_observed_TWS"].to_numpy(dtype=np.float32)
        differences = np.full(len(events), np.nan, dtype=np.float32)
        differences[supported] = means[supported] - focal[supported]

        ages = np.full(values.shape, np.nan, dtype=np.float32)
        age_values = source_matrix - observation_periods
        ages[valid] = age_values[valid].astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median_age_all = np.nanmedian(ages, axis=1).astype(np.float32)
        median_age = np.full(len(events), np.nan, dtype=np.float32)
        median_age[supported] = median_age_all[supported]

        output = pd.DataFrame({
            "neighbor_tws_mean": means,
            "neighbor_tws_minus_focal": differences,
            "neighbor_tws_count": count,
            "neighbor_tws_median_age": median_age,
        })
        if output.columns.tolist() != NEIGHBOR_FEATURE_COLUMNS:
            raise AssertionError("neighbor feature schema changed unexpectedly")
        if np.isinf(output.to_numpy(dtype=np.float32)).any():
            raise AssertionError("neighbor features contain infinite values")

        finite_ages = ages[np.isfinite(ages)]
        diagnostics: dict[str, object] = {
            "rows": int(len(events)),
            "neighbor_slots": int(len(events) * NEIGHBOR_K),
            "finite_neighbor_values": int(valid.sum()),
            "supported_rows": int(supported.sum()),
            "support_share": float(supported.mean()),
            "count_q50": float(np.quantile(count, 0.5)),
            "count_q90": float(np.quantile(count, 0.9)),
            "age_q50": float(np.quantile(finite_ages, 0.5)) if finite_ages.size else None,
            "age_q90": float(np.quantile(finite_ages, 0.9)) if finite_ages.size else None,
            "cutoff_is_event_source_rows": int((cutoff_period == source_period).sum()),
            "used_observation_max_cutoff_excess_months": int(
                max(0, int((observation_periods[valid] - cutoff_matrix[valid]).max()))
                if valid.any() else 0
            ),
            "used_observation_max_source_excess_months": int(
                max(0, int((observation_periods[valid] - source_matrix[valid]).max()))
                if valid.any() else 0
            ),
            "unsupported_rows": int((~supported).sum()),
        }
        return NeighborFeatureResult(output.reset_index(drop=True), diagnostics)


def _neighbor_geometry(locations: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed nearest-other IDs and distances, preserving the radius cap."""
    from scipy.spatial import cKDTree

    lat = locations["lat"].to_numpy(dtype=np.float64)
    lon = locations["lon"].to_numpy(dtype=np.float64)
    n_locations = len(locations)
    neighbor_ids = np.full((n_locations, NEIGHBOR_K), -1, dtype=np.int32)
    distances = np.full((n_locations, NEIGHBOR_K), np.nan, dtype=np.float32)
    if n_locations <= 1:
        return neighbor_ids, distances

    vectors = _unit_vectors(lat, lon)
    query_k = min(n_locations, NEIGHBOR_K + 1)
    chord, candidate_ids = cKDTree(vectors).query(vectors, k=query_k, workers=1)
    chord = np.atleast_2d(chord)
    candidate_ids = np.atleast_2d(candidate_ids)
    if chord.shape[0] != n_locations:
        chord = chord.T
        candidate_ids = candidate_ids.T
    for row in range(n_locations):
        selected: list[tuple[int, float]] = []
        for raw_distance, raw_id in zip(chord[row], candidate_ids[row]):
            candidate = int(raw_id)
            if candidate == row or candidate < 0:
                continue
            arc = 2.0 * np.arcsin(np.clip(float(raw_distance) / 2.0, 0.0, 1.0))
            distance_km = float(EARTH_RADIUS_KM * arc)
            selected.append((candidate, distance_km))
            if len(selected) == NEIGHBOR_K:
                break
        for slot, (candidate, distance_km) in enumerate(selected):
            if distance_km <= NEIGHBOR_RADIUS_KM:
                neighbor_ids[row, slot] = candidate
                distances[row, slot] = np.float32(distance_km)
    return neighbor_ids, distances


def build_neighbor_state_index(structural: pd.DataFrame) -> NeighborStateIndex:
    """Build geometry and as-of state from a target-blind structural panel."""
    _assert_target_blind(structural, "structural")
    required = {"time", "lat", "lon", "TWS_t"}
    missing = required.difference(structural.columns)
    if missing:
        raise ValueError(f"structural missing required columns: {sorted(missing)}")
    if structural.empty:
        raise ValueError("structural cannot be empty")

    panel = structural.loc[:, ["time", "lat", "lon", "TWS_t"]].copy()
    panel["state_period"] = _month_number(panel["time"])
    if panel.duplicated(["lat", "lon", "state_period"]).any():
        raise AssertionError("structural panel must be unique by location/calendar month")
    locations = (
        panel.loc[:, ["lat", "lon"]]
        .drop_duplicates()
        .sort_values(["lat", "lon"], kind="mergesort")
        .reset_index(drop=True)
    )
    locations["location_id"] = np.arange(len(locations), dtype=np.int32)
    location_index = pd.MultiIndex.from_frame(locations.loc[:, ["lat", "lon"]])
    panel_location_ids = location_index.get_indexer(pd.MultiIndex.from_frame(panel.loc[:, ["lat", "lon"]]))
    if (panel_location_ids < 0).any():
        raise AssertionError("structural panel contains an unmapped location")

    periods = np.sort(panel["state_period"].unique()).astype(np.int32)
    state = np.full((len(locations), len(periods)), np.nan, dtype=np.float32)
    period_ids = np.searchsorted(periods, panel["state_period"].to_numpy(dtype=np.int32))
    tws = panel["TWS_t"].to_numpy(dtype=np.float32)
    finite = np.isfinite(tws)
    state[panel_location_ids[finite], period_ids[finite]] = tws[finite]

    finite_state = np.isfinite(state)
    index_grid = np.arange(len(periods), dtype=np.int32)[None, :]
    last_index = np.maximum.accumulate(np.where(finite_state, index_grid, -1), axis=1)
    safe_index = np.maximum(last_index, 0)
    latest_values = np.take_along_axis(state, safe_index, axis=1)
    latest_periods = periods[safe_index]
    latest_values[last_index < 0] = np.nan
    latest_periods[last_index < 0] = -1
    del state

    neighbor_ids, distances = _neighbor_geometry(locations)
    return NeighborStateIndex(
        locations=locations,
        periods=periods,
        neighbor_ids=neighbor_ids,
        neighbor_distances_km=distances,
        latest_values=latest_values,
        latest_periods=latest_periods,
        _location_index=location_index,
    )
