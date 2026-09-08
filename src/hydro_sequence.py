"""Causal calendar-grid hydrology sequences for the Stage C controls.

The output has one row per requested source sample and fixed calendar slots from
``span - 1`` months before the source month through the source month itself.
Missing source months are represented explicitly by value masks; they are never
compressed into adjacent time steps.  The elapsed channel is the true calendar
offset of each slot, so a downstream sequence model need not infer elapsed time
from row adjacency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.regional_context import HYDRO_COLUMNS


def _assert_target_blind(frame: pd.DataFrame, name: str) -> None:
    forbidden = {column for column in frame.columns if column.lower() in {"target", "y", "label"}}
    if forbidden:
        raise AssertionError(f"{name} must be target-blind; found {sorted(forbidden)}")


def sequence_feature_names(span: int, *, prefix: str = "seq_") -> list[str]:
    """Return the stable flattened sequence schema for one calendar span."""
    if span < 1:
        raise ValueError("span must be positive")
    names: list[str] = []
    for offset in range(span - 1, -1, -1):
        step = f"{prefix}tminus{offset}_"
        names.extend(f"{step}{variable}" for variable in HYDRO_COLUMNS)
        names.extend(f"{step}{variable}_observed" for variable in HYDRO_COLUMNS)
        names.append(f"{step}elapsed_months")
    return names


def build_hydro_sequence_features(
    source: pd.DataFrame,
    sample_ids: pd.Series,
    *,
    span: int,
    prefix: str = "seq_",
) -> pd.DataFrame:
    """Build fixed-span local hydrology sequences without targets or future rows.

    ``sample_ids`` must identify rows in the supplied source panel.  A requested
    row at calendar month ``t`` receives only lookups at ``t`` and earlier
    calendar months.  The procedure is invariant to source order and to edits
    in future months.
    """
    _assert_target_blind(source, "source")
    if span < 1:
        raise ValueError("span must be positive")
    required = {"sample_id", "time", "lat", "lon", *HYDRO_COLUMNS}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"source missing required columns: {sorted(missing)}")
    if source.sample_id.duplicated().any():
        raise AssertionError("source sample_id must be unique")

    request = pd.DataFrame({"sample_id": sample_ids.astype(str).to_numpy()})
    if request.sample_id.duplicated().any():
        raise AssertionError("requested sample_ids must be unique")
    request["_order"] = np.arange(len(request), dtype=np.int64)
    panel = source.loc[:, ["sample_id", "time", "lat", "lon", *HYDRO_COLUMNS]].copy()
    panel["sample_id"] = panel.sample_id.astype(str)
    panel["_period"] = pd.to_datetime(panel.time).dt.to_period("M").astype("int64")
    if panel.duplicated(["lat", "lon", "_period"]).any():
        raise AssertionError("source has duplicate location/calendar observations")
    request = request.merge(
        panel.loc[:, ["sample_id", "lat", "lon", "_period"]],
        on="sample_id", how="left", validate="one_to_one", sort=False,
    )
    if request[["lat", "lon", "_period"]].isna().any().any():
        raise AssertionError("requested sequence sample_id is absent from source")

    expanded = request.loc[request.index.repeat(span)].reset_index(drop=True)
    offsets = np.tile(np.arange(span - 1, -1, -1, dtype=np.int16), len(request))
    expanded["_offset"] = offsets
    expanded["_lookup_period"] = expanded["_period"].to_numpy(dtype=np.int64) - offsets
    lookup = panel.loc[:, ["lat", "lon", "_period", *HYDRO_COLUMNS]].rename(columns={"_period": "_lookup_period"})
    expanded = expanded.merge(
        lookup, on=["lat", "lon", "_lookup_period"], how="left", validate="many_to_one", sort=False,
    )
    values = expanded.loc[:, HYDRO_COLUMNS].to_numpy(dtype=np.float32).reshape(
        len(request), span, len(HYDRO_COLUMNS)
    )
    masks = np.isfinite(values).astype(np.float32)
    names = sequence_feature_names(span, prefix=prefix)
    out = np.empty((len(request), len(names)), dtype=np.float32)
    cursor = 0
    for index in range(span):
        out[:, cursor:cursor + len(HYDRO_COLUMNS)] = values[:, index, :]
        cursor += len(HYDRO_COLUMNS)
        out[:, cursor:cursor + len(HYDRO_COLUMNS)] = masks[:, index, :]
        cursor += len(HYDRO_COLUMNS)
        out[:, cursor] = float(span - 1 - index)
        cursor += 1
    result = pd.DataFrame(out, columns=names)
    if np.isinf(result.to_numpy(dtype=np.float32)).any():
        raise AssertionError("sequence matrix contains infinite values")
    return result


@dataclass(frozen=True)
class PrefixNormalizer:
    """Missing-aware normalization fitted only on one training prefix."""

    fill_values: np.ndarray
    centers: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "PrefixNormalizer":
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("matrix must be two-dimensional")
        finite = np.isfinite(values)
        fill = np.zeros(values.shape[1], dtype=np.float64)
        centers = np.zeros(values.shape[1], dtype=np.float64)
        scales = np.ones(values.shape[1], dtype=np.float64)
        for column in range(values.shape[1]):
            observed = values[finite[:, column], column]
            if len(observed):
                fill[column] = float(np.median(observed))
                centers[column] = float(np.mean(observed))
                scale = float(np.std(observed))
                scales[column] = scale if np.isfinite(scale) and scale > 1e-6 else 1.0
        return cls(fill, centers, scales)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.fill_values):
            raise ValueError("matrix shape differs from fitted prefix normalizer")
        filled = np.where(np.isfinite(values), values, self.fill_values)
        normalized = (filled - self.centers) / self.scales
        if not np.isfinite(normalized).all():
            raise AssertionError("prefix normalization produced a non-finite value")
        return normalized.astype(np.float32)
