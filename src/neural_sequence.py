"""Availability-faithful tensors and compact neural delta models.

The module deliberately contains no competition-Test prediction pathway.  Its
inputs are a target-blind historical source panel plus a replay ledger, and every
tensor lookup is causal in calendar time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.hydro_trajectory import (
    attach_trajectory_features,
    build_hydro_trajectory_map,
    build_regional_trajectory_map,
)
from src.ml_features import build_hydro_gap_safe_feature_matrix
from src.regional_context import (
    DEFAULT_WIDTHS,
    HYDRO_COLUMNS,
    attach_regional_features,
    build_regional_context,
)


def _target_blind(frame: pd.DataFrame, name: str) -> None:
    bad = {column for column in frame.columns if column.lower() in {"target", "y", "label"}}
    if bad:
        raise AssertionError(f"{name} must be target-blind; found {sorted(bad)}")


@dataclass(frozen=True)
class B3FeatureMaps:
    """Causal maps derived only from one legal source availability view."""

    regional: pd.DataFrame
    local_trajectory: pd.DataFrame
    regional_trajectory: pd.DataFrame


def build_b3_feature_maps(source: pd.DataFrame) -> B3FeatureMaps:
    _target_blind(source, "source")
    regional_source = source.loc[:, ["sample_id", "time", "lat", "lon", *HYDRO_COLUMNS]].copy()
    regional = build_regional_context(regional_source)
    local = build_hydro_trajectory_map(regional_source)
    regional_trajectory = build_regional_trajectory_map(regional_source, regional, local)
    return B3FeatureMaps(regional, local, regional_trajectory)


def build_b3_matrix(
    ledger: pd.DataFrame,
    source: pd.DataFrame,
    structural: pd.DataFrame,
    maps: B3FeatureMaps,
) -> pd.DataFrame:
    """Build the exact B3 static context from a legal availability view."""
    _target_blind(ledger, "ledger")
    base = build_hydro_gap_safe_feature_matrix(ledger, source, structural)
    blocks = [
        base.reset_index(drop=True),
        attach_regional_features(ledger.sample_id, maps.regional).reset_index(drop=True),
        attach_trajectory_features(ledger.sample_id, maps.local_trajectory).reset_index(drop=True),
        attach_trajectory_features(ledger.sample_id, maps.regional_trajectory).reset_index(drop=True),
    ]
    result = pd.concat(blocks, axis=1)
    if result.columns.duplicated().any():
        raise AssertionError("B3 static context has duplicate feature names")
    if np.isinf(result.to_numpy(dtype=np.float32)).any():
        raise AssertionError("B3 static context contains infinite values")
    return result.astype(np.float32).reset_index(drop=True)


def deterministic_horizon_cap(
    rows: pd.DataFrame,
    *,
    per_horizon: int = 36_000,
    seed: int = 20260908,
) -> pd.DataFrame:
    """Take a fixed training-only balanced cap while retaining source order."""
    if per_horizon < 1:
        raise ValueError("per_horizon must be positive")
    if "h" not in rows:
        raise ValueError("rows must contain h")
    picked: list[np.ndarray] = []
    for horizon in sorted(rows.h.unique()):
        positions = np.flatnonzero(rows.h.to_numpy() == horizon)
        if len(positions) > per_horizon:
            rng = np.random.default_rng(seed + int(horizon))
            positions = rng.choice(positions, size=per_horizon, replace=False)
        picked.append(positions)
    selection = np.sort(np.concatenate(picked)) if picked else np.empty(0, dtype=np.int64)
    if not len(selection):
        raise AssertionError("deterministic horizon cap selected no rows")
    return rows.iloc[selection].reset_index(drop=True)


def _regional_value_panel(source: pd.DataFrame, width: float) -> pd.DataFrame:
    """One regional aggregate per actually supplied source-bin/month."""
    x = source.loc[:, ["time", "lat", "lon", *HYDRO_COLUMNS]].copy()
    x["_period"] = pd.to_datetime(x.time).dt.to_period("M").astype("int64")
    x["_lat_bin"] = np.floor(x.lat.astype(float) / width) * width
    x["_lon_bin"] = np.floor(x.lon.astype(float) / width) * width
    keys = ["_period", "_lat_bin", "_lon_bin"]
    out = x.groupby(keys, sort=False)[list(HYDRO_COLUMNS)].mean().reset_index()
    return out


def _age_within_span(masks: np.ndarray, span: int) -> np.ndarray:
    """Causal age in slots, using ``span`` as the >=span/no-history sentinel."""
    out = np.empty_like(masks, dtype=np.float32)
    previous = np.full(masks.shape[0:1] + masks.shape[2:3], float(span), dtype=np.float32)
    for step in range(masks.shape[1]):
        observed = masks[:, step, :] > 0.5
        previous = np.where(observed, 0.0, np.minimum(previous + 1.0, float(span)))
        out[:, step, :] = previous
    return out


@dataclass(frozen=True)
class TemporalTensor:
    values: np.ndarray
    channels: tuple[str, ...]
    span: int


def temporal_channel_names() -> tuple[str, ...]:
    names: list[str] = []
    for scope in ("local", "regional5", "regional15"):
        names.extend(f"{scope}_{column}" for column in HYDRO_COLUMNS)
    for scope in ("local", "regional5", "regional15"):
        names.extend(f"{scope}_{column}_observed" for column in HYDRO_COLUMNS)
    for scope in ("local", "regional5", "regional15"):
        names.extend(f"{scope}_{column}_age_months" for column in HYDRO_COLUMNS)
    names.extend(["calendar_sin", "calendar_cos", "elapsed_months"])
    return tuple(names)


def build_temporal_tensor(
    source: pd.DataFrame,
    sample_ids: pd.Series,
    *,
    span: int,
    widths: Iterable[float] = DEFAULT_WIDTHS,
) -> TemporalTensor:
    """Create local/5-degree/15-degree causal monthly sequence channels.

    The source panel itself defines availability.  An absent local source row
    cannot be recovered from dense Train, while an aggregate may still exist when
    another supplied row occupies the same regional bin/month.
    """
    _target_blind(source, "source")
    widths = tuple(widths)
    if span < 1:
        raise ValueError("span must be positive")
    if widths != (5.0, 15.0):
        raise ValueError("temporal tensor currently requires 5 and 15 degree regions")
    needed = {"sample_id", "time", "lat", "lon", *HYDRO_COLUMNS}
    if missing := needed.difference(source.columns):
        raise ValueError(f"source missing required sequence columns: {sorted(missing)}")
    if source.sample_id.duplicated().any():
        raise AssertionError("source sample IDs must be unique")
    requested = pd.DataFrame({"sample_id": sample_ids.astype(str).to_numpy()})
    if requested.sample_id.duplicated().any():
        raise AssertionError("sequence requests must be unique")
    requested["_order"] = np.arange(len(requested), dtype=np.int64)
    panel = source.loc[:, ["sample_id", "time", "lat", "lon", *HYDRO_COLUMNS]].copy()
    panel["sample_id"] = panel.sample_id.astype(str)
    panel["_period"] = pd.to_datetime(panel.time).dt.to_period("M").astype("int64")
    if panel.duplicated(["lat", "lon", "_period"]).any():
        raise AssertionError("source has duplicate local calendar rows")
    request = requested.merge(
        panel.loc[:, ["sample_id", "lat", "lon", "_period"]],
        on="sample_id", how="left", validate="one_to_one", sort=False,
    )
    if request[["lat", "lon", "_period"]].isna().any().any():
        raise AssertionError("requested sequence ID is absent from availability source")
    expanded = request.loc[request.index.repeat(span)].reset_index(drop=True)
    offsets = np.tile(np.arange(span - 1, -1, -1, dtype=np.int16), len(request))
    expanded["_offset"] = offsets
    expanded["_lookup_period"] = expanded["_period"].to_numpy(dtype=np.int64) - offsets
    local = panel.loc[:, ["lat", "lon", "_period", *HYDRO_COLUMNS]].rename(columns={"_period": "_lookup_period"})
    joined = expanded.merge(local, how="left", on=["lat", "lon", "_lookup_period"], validate="many_to_one", sort=False)
    local_values = joined.loc[:, HYDRO_COLUMNS].to_numpy(dtype=np.float32).reshape(len(request), span, len(HYDRO_COLUMNS))
    regional_values: list[np.ndarray] = []
    for width in widths:
        tag = f"{float(width):g}"
        regional = _regional_value_panel(panel, width)
        keys = ["_lookup_period", "_lat_bin", "_lon_bin"]
        right = regional.rename(columns={"_period": "_lookup_period"})
        query = expanded.loc[:, ["_lookup_period", "lat", "lon"]].copy()
        query["_lat_bin"] = np.floor(query.lat.astype(float) / width) * width
        query["_lon_bin"] = np.floor(query.lon.astype(float) / width) * width
        query["_order"] = np.arange(len(query), dtype=np.int64)
        attached = query.merge(right, how="left", on=keys, validate="many_to_one", sort=False).sort_values("_order", kind="mergesort")
        regional_values.append(attached.loc[:, HYDRO_COLUMNS].to_numpy(dtype=np.float32).reshape(len(request), span, len(HYDRO_COLUMNS)))
        del tag
    values = np.concatenate([local_values, *regional_values], axis=2)
    masks = np.isfinite(values).astype(np.float32)
    ages = _age_within_span(masks, span)
    calendar_month = ((expanded["_lookup_period"].to_numpy(dtype=np.int64) % 12) + 1).astype(np.float32)
    calendar = np.stack([
        np.sin(2.0 * np.pi * calendar_month / 12.0),
        np.cos(2.0 * np.pi * calendar_month / 12.0),
        expanded["_offset"].to_numpy(dtype=np.float32),
    ], axis=1).reshape(len(request), span, 3).astype(np.float32)
    output = np.concatenate([values, masks, ages, calendar], axis=2).astype(np.float32)
    if output.shape[2] != len(temporal_channel_names()):
        raise AssertionError("temporal channel schema drift")
    if np.isinf(output).any():
        raise AssertionError("temporal tensor contains infinite values")
    return TemporalTensor(output, temporal_channel_names(), span)


@dataclass(frozen=True)
class ArrayNormalizer:
    fill_values: np.ndarray
    centers: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ArrayNormalizer":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim < 2:
            raise ValueError("normalizer needs at least two dimensions")
        flat = array.reshape(-1, array.shape[-1])
        finite = np.isfinite(flat)
        fill = np.zeros(flat.shape[1], dtype=np.float64)
        center = np.zeros(flat.shape[1], dtype=np.float64)
        scale = np.ones(flat.shape[1], dtype=np.float64)
        for column in range(flat.shape[1]):
            observed = flat[finite[:, column], column]
            if len(observed):
                fill[column] = float(np.median(observed))
                center[column] = float(np.mean(observed))
                std = float(np.std(observed))
                scale[column] = std if np.isfinite(std) and std > 1e-6 else 1.0
        return cls(fill, center, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != len(self.fill_values):
            raise ValueError("normalizer column count differs from fitted state")
        filled = np.where(np.isfinite(array), array, self.fill_values)
        result = (filled - self.centers) / self.scales
        if not np.isfinite(result).all():
            raise AssertionError("normalizer produced non-finite values")
        return result.astype(np.float32)

    def save(self, path: Path, *, columns: Iterable[str]) -> None:
        np.savez_compressed(path, fill_values=self.fill_values, centers=self.centers, scales=self.scales,
                            columns=np.asarray(tuple(columns), dtype="U"))

    @classmethod
    def load(cls, path: Path) -> tuple["ArrayNormalizer", tuple[str, ...]]:
        with np.load(path, allow_pickle=False) as data:
            return cls(data["fill_values"], data["centers"], data["scales"]), tuple(data["columns"].tolist())


def make_model(architecture: str, *, static_dim: int, sequence_dim: int | None = None):
    """Construct a compact PyTorch model without any temporal lookahead."""
    import torch
    from torch import nn

    class StaticBranch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(static_dim, 192), nn.ReLU(), nn.Dropout(0.10), nn.Linear(192, 96), nn.ReLU())
        def forward(self, x):
            return self.layers(x)

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__(); self.branch = StaticBranch(); self.head = nn.Linear(96, 1)
        def forward(self, static, sequence=None):
            return self.head(self.branch(static)).squeeze(1)

    class GRU(nn.Module):
        def __init__(self) -> None:
            super().__init__(); self.branch = StaticBranch(); self.gru = nn.GRU(sequence_dim, 64, batch_first=True); self.head = nn.Sequential(nn.Linear(160, 96), nn.ReLU(), nn.Dropout(0.10), nn.Linear(96, 1))
        def forward(self, static, sequence):
            _, hidden = self.gru(sequence); return self.head(torch.cat([self.branch(static), hidden[-1]], dim=1)).squeeze(1)

    class CausalBlock(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__(); self.pad = dilation * 2; self.conv1 = nn.Conv1d(channels, channels, 3, dilation=dilation); self.conv2 = nn.Conv1d(channels, channels, 3, dilation=dilation); self.act = nn.ReLU(); self.dropout = nn.Dropout(0.10)
        def forward(self, x):
            residual = x
            x = self.conv1(nn.functional.pad(x, (self.pad, 0))); x = self.dropout(self.act(x))
            x = self.conv2(nn.functional.pad(x, (self.pad, 0))); return self.act(x + residual)

    class TCN(nn.Module):
        def __init__(self) -> None:
            super().__init__(); self.branch = StaticBranch(); self.input = nn.Conv1d(sequence_dim, 64, 1); self.blocks = nn.Sequential(CausalBlock(64, 1), CausalBlock(64, 2)); self.head = nn.Sequential(nn.Linear(160, 96), nn.ReLU(), nn.Dropout(0.10), nn.Linear(96, 1))
        def forward(self, static, sequence):
            x = self.blocks(self.input(sequence.transpose(1, 2)))[:, :, -1]
            return self.head(torch.cat([self.branch(static), x], dim=1)).squeeze(1)

    if architecture == "mlp":
        return MLP()
    if sequence_dim is None:
        raise ValueError("temporal architecture requires sequence_dim")
    if architecture == "gru":
        return GRU()
    if architecture == "tcn":
        return TCN()
    raise ValueError(f"unknown architecture: {architecture}")
