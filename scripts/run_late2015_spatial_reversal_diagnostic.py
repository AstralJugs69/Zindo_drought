"""Train-only spatial/reversal diagnostic for the late-2015 B3 failure.

This runner deliberately does not open Test.csv, fit a model, create predictions,
or write a submission.  It constructs the exact-calendar Train table
``d_t = target_t - TWS_t``, computes fixed-coordinate nearest-neighbour and
calendar-reversal summaries, attributes the saved B3/98 replay errors, and audits
the frozen 446-column B3 schema against the legal replay visibility state.

The spatial/reversal tables are descriptive (the target enters ``d_t``), and are
therefore marked ``ORACLE_DIAGNOSTIC``.  The only candidate-state audit uses TWS
observations at or before the replay source month and is target-blind.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import TEST_H_WEIGHTS

try:  # ``resource`` is available on the VM; keep imports testable on Windows.
    import resource as _resource
except ImportError:  # pragma: no cover - exercised only on Windows.
    _resource = None


EARTH_RADIUS_KM = 6371.0088
NEIGHBOR_K = 8
NEIGHBOR_RADIUS_KM = 500.0
MIN_NEIGHBORS = 4
CELL_DEGREES = 5.0
SUSPICIOUS_MONTHS = ("2015-01", "2015-02", "2015-06")
HISTORICAL_YEARS = tuple(range(2009, 2014))
TRAIN_COLUMNS = ["sample_id", "time", "lat", "lon", "TWS_t", "target"]
OOF_COLUMNS = [
    "sample_id", "source_date", "target_date", "last_observed_date",
    "last_observed_TWS", "h", "lat", "lon", "target", "prediction",
    "model",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_values(values: np.ndarray | pd.Series | Iterable[str]) -> str:
    if isinstance(values, pd.Series):
        raw = values.astype(str).str.cat(sep="\n").encode("utf-8")
    elif isinstance(values, np.ndarray):
        raw = np.ascontiguousarray(values).tobytes()
    else:
        raw = "\n".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_frame(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256("\n".join(columns).encode("utf-8"))
    for column in columns:
        values = frame[column]
        if pd.api.types.is_datetime64_any_dtype(values):
            raw = values.astype("int64").to_numpy().tobytes()
        elif pd.api.types.is_numeric_dtype(values):
            raw = np.ascontiguousarray(values.to_numpy()).tobytes()
        else:
            raw = values.astype(str).str.cat(sep="\n").encode("utf-8")
        digest.update(raw)
    return digest.hexdigest()


def month_ordinal(values: pd.Series | pd.DatetimeIndex | Iterable[Any]) -> np.ndarray:
    dt = pd.to_datetime(values)
    if isinstance(dt, pd.Series):
        return (dt.dt.year.to_numpy(dtype=np.int32) * 12 + dt.dt.month.to_numpy(dtype=np.int32))
    return dt.year.to_numpy(dtype=np.int32) * 12 + dt.month.to_numpy(dtype=np.int32)


def ordinal_to_month(value: int) -> str:
    year, month0 = divmod(int(value) - 1, 12)
    return f"{year:04d}-{month0 + 1:02d}"


def month_label(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series)
    return dt.dt.strftime("%Y-%m")


def finite_quantiles(values: np.ndarray, probabilities: tuple[float, ...]) -> dict[str, float | None]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {str(p): None for p in probabilities}
    return {str(p): float(np.quantile(x, p)) for p in probabilities}


def metric_sums(error: np.ndarray) -> dict[str, float | int | None]:
    x = np.asarray(error, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {
            "rows": 0, "sse": 0.0, "sae": 0.0, "error_sum": 0.0,
            "raw_rmse": None, "mae": None, "bias": None,
        }
    sse = float(np.sum(np.square(x), dtype=np.float64))
    sae = float(np.sum(np.abs(x), dtype=np.float64))
    return {
        "rows": int(len(x)), "sse": sse, "sae": sae,
        "error_sum": float(np.sum(x, dtype=np.float64)),
        "raw_rmse": float(math.sqrt(sse / len(x))),
        "mae": float(sae / len(x)),
        "bias": float(np.sum(x, dtype=np.float64) / len(x)),
    }


def cell_sse_decomposition(
    values: np.ndarray | pd.Series,
    cell_ids: np.ndarray | pd.Series,
) -> dict[str, float | int]:
    """Decompose raw SSE around zero into cell-mean and within-cell SSE.

    For unequal cell sizes this uses ``sum(n_c * mean_c**2)`` and therefore
    conserves ``sum(values**2)`` exactly up to floating-point roundoff.
    """
    x = np.asarray(values, dtype=np.float64)
    c = np.asarray(cell_ids)
    finite = np.isfinite(x)
    x = x[finite]
    c = c[finite]
    if not len(x):
        return {
            "rows": 0, "cells": 0, "sse_total": 0.0,
            "sse_between_cell_mean": 0.0, "sse_within_cell": 0.0,
            "sse_conservation_abs_error": 0.0,
        }
    frame = pd.DataFrame({"value": x, "cell": c})
    grouped = frame.groupby("cell", sort=False)["value"]
    count = grouped.size().to_numpy(dtype=np.float64)
    means = grouped.mean().to_numpy(dtype=np.float64)
    total = float(np.sum(np.square(x), dtype=np.float64))
    between = float(np.sum(count * np.square(means), dtype=np.float64))
    within = float(np.sum(np.square(frame["value"].to_numpy() - frame["cell"].map(grouped.mean()).to_numpy()), dtype=np.float64))
    return {
        "rows": int(len(x)), "cells": int(len(means)),
        "sse_total": total,
        "sse_between_cell_mean": between,
        "sse_within_cell": within,
        "sse_conservation_abs_error": abs(total - between - within),
    }


def _cell_key(lat: np.ndarray | pd.Series, lon: np.ndarray | pd.Series) -> np.ndarray:
    lat_bin = np.floor(np.asarray(lat, dtype=np.float64) / CELL_DEGREES).astype(np.int32)
    lon_bin = np.floor(np.asarray(lon, dtype=np.float64) / CELL_DEGREES).astype(np.int32)
    return (lat_bin.astype(np.int64) * 1000 + lon_bin.astype(np.int64)).astype(np.int64)


def build_neighbor_index(
    lat: np.ndarray | pd.Series,
    lon: np.ndarray | pd.Series,
    *,
    k: int = NEIGHBOR_K,
    radius_km: float = NEIGHBOR_RADIUS_KM,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-coordinate nearest *other* locations and great-circle km.

    The unit-vector KD-tree naturally handles longitude wrap.  The explicit
    self-filter below is retained even though a unique coordinate set is expected.
    """
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = np.asarray(lon, dtype=np.float64)
    if len(lat_arr) != len(lon_arr) or len(lat_arr) <= k:
        raise ValueError("Need more than k unique locations")
    if not (np.isfinite(lat_arr).all() and np.isfinite(lon_arr).all()):
        raise ValueError("Coordinates must be finite")
    lat_rad = np.deg2rad(lat_arr)
    lon_rad = np.deg2rad(lon_arr)
    xyz = np.column_stack([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    ])
    tree = cKDTree(xyz)
    distances, indices = tree.query(xyz, k=min(len(xyz), k + 1), workers=1)
    if indices.ndim == 1:
        indices = indices[:, None]
        distances = distances[:, None]
    out_idx = np.full((len(xyz), k), -1, dtype=np.int32)
    out_dist = np.full((len(xyz), k), np.nan, dtype=np.float32)
    for row in range(len(xyz)):
        write = 0
        for candidate, chord in zip(indices[row], distances[row], strict=True):
            candidate = int(candidate)
            if candidate == row or candidate < 0:
                continue
            arc = 2.0 * math.asin(min(1.0, float(chord) / 2.0)) * EARTH_RADIUS_KM
            if arc <= radius_km + 1e-9:
                out_idx[row, write] = candidate
                out_dist[row, write] = arc
                write += 1
                if write == k:
                    break
    if np.any(out_idx == np.arange(len(xyz), dtype=np.int32)[:, None]):
        raise AssertionError("self location leaked into nearest-neighbour index")
    return out_idx, out_dist


def neighbor_state_for_rows(
    location_ids: np.ndarray,
    values: np.ndarray,
    *,
    n_locations: int,
    neighbor_index: np.ndarray,
    min_neighbors: int = MIN_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute leave-one-location-out neighbor mean/count for one calendar group."""
    state = np.full(n_locations, np.nan, dtype=np.float64)
    loc = np.asarray(location_ids, dtype=np.int64)
    val = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(val)
    if len(np.unique(loc[finite])) != int(finite.sum()):
        raise AssertionError("duplicate location/month rows in spatial state")
    state[loc[finite]] = val[finite]
    safe = np.where(neighbor_index >= 0, neighbor_index, 0)
    neighbor_values = state[safe]
    neighbor_values[neighbor_index < 0] = np.nan
    count = np.isfinite(neighbor_values).sum(axis=1).astype(np.int16)
    total = np.nansum(neighbor_values, axis=1)
    means = np.divide(total, count, out=np.full(n_locations, np.nan), where=count >= min_neighbors)
    return means[loc], count[loc], state


def spatial_month_metrics(
    frame: pd.DataFrame,
    *,
    n_locations: int,
    neighbor_index: np.ndarray,
    neighbor_distances: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    values = frame["d"].to_numpy(dtype=np.float64)
    loc = frame["location_id"].to_numpy(dtype=np.int64)
    means, counts, _ = neighbor_state_for_rows(
        loc, values, n_locations=n_locations, neighbor_index=neighbor_index,
    )
    valid = np.isfinite(values)
    supported = valid & np.isfinite(means)
    support_values = values[supported]
    support_means = means[supported]
    roughness = support_values - support_means
    nonzero = supported & (values != 0.0) & (means != 0.0)
    sign_agree = float(np.mean(np.sign(values[nonzero]) == np.sign(means[nonzero]))) if nonzero.any() else None
    corr = float(np.corrcoef(support_values, support_means)[0, 1]) if len(support_values) > 1 and np.std(support_values) > 0 and np.std(support_means) > 0 else None
    valid_values = values[valid]
    cell_ids = _cell_key(frame["lat"].to_numpy(), frame["lon"].to_numpy())
    cell_decomp = cell_sse_decomposition(values, cell_ids)
    finite_counts = counts[valid]
    finite_neighbor_dist = neighbor_distances[neighbor_index >= 0]
    distance_stats = finite_quantiles(finite_neighbor_dist, (0.0, 0.5, 0.9, 1.0))
    result: dict[str, Any] = {
        "source_month": str(frame["source_month"].iloc[0]),
        "oracle_diagnostic": True,
        "rows": int(len(frame)),
        "valid_d_rows": int(valid.sum()),
        "neighbor_supported_rows": int(supported.sum()),
        "neighbor_support_share": float(supported.sum() / valid.sum()) if valid.any() else None,
        "neighbor_count_q0": int(np.min(finite_counts)) if len(finite_counts) else None,
        "neighbor_count_q50": float(np.quantile(finite_counts, 0.5)) if len(finite_counts) else None,
        "neighbor_count_q90": float(np.quantile(finite_counts, 0.9)) if len(finite_counts) else None,
        "neighbor_count_q100": int(np.max(finite_counts)) if len(finite_counts) else None,
        "neighbor_distance_q0_km": distance_stats["0.0"],
        "neighbor_distance_q50_km": distance_stats["0.5"],
        "neighbor_distance_q90_km": distance_stats["0.9"],
        "neighbor_distance_q100_km": distance_stats["1.0"],
        "neighbor_corr": corr,
        "neighbor_sign_agreement": sign_agree,
        "d_centered_variance": float(np.var(valid_values)) if len(valid_values) else None,
        "neighbor_squared_bias": float(np.mean(np.square(support_means))) if len(support_means) else None,
        "d_rmse": float(math.sqrt(np.mean(np.square(valid_values)))) if len(valid_values) else None,
        "roughness_rmse": float(math.sqrt(np.mean(np.square(roughness)))) if len(roughness) else None,
        **cell_decomp,
    }
    cell_summary = (
        frame.assign(cell_id=cell_ids)
        .loc[valid, ["source_month", "cell_id", "d"]]
        .groupby(["source_month", "cell_id"], sort=True)["d"]
        .agg(n="size", cell_mean="mean", cell_sse=lambda x: float(np.sum(np.square(x.to_numpy(dtype=np.float64)))))
        .reset_index()
    )
    cell_summary["cell_within_sse"] = (
        frame.assign(cell_id=cell_ids)
        .loc[valid, ["cell_id", "d"]]
        .groupby("cell_id", sort=True)["d"]
        .apply(lambda x: float(np.sum(np.square(x.to_numpy(dtype=np.float64) - x.mean()))))
        .reindex(cell_summary["cell_id"])
        .to_numpy()
    )
    return result, cell_summary


def build_exact_calendar_reversal(canonical: pd.DataFrame) -> pd.DataFrame:
    """Join d_t to d_(t+1) only on exact location/month keys."""
    x = canonical.loc[:, ["location_id", "source_ord", "d"]].copy()
    x["key"] = x["location_id"].astype(np.int64) * 1000 + x["source_ord"].astype(np.int64)
    valid = x.loc[np.isfinite(x["d"].to_numpy())].sort_values("key", kind="mergesort")
    keys = valid["key"].to_numpy(dtype=np.int64)
    vals = valid["d"].to_numpy(dtype=np.float64)
    query = canonical.loc[:, ["location_id", "source_ord", "d"]].copy()
    query["target_ord"] = query["source_ord"].astype(np.int32) + 1
    query["next_key"] = query["location_id"].astype(np.int64) * 1000 + query["target_ord"].astype(np.int64)
    positions = np.searchsorted(keys, query["next_key"].to_numpy(dtype=np.int64))
    found = (positions < len(keys)) & (keys[np.minimum(positions, max(len(keys) - 1, 0))] == query["next_key"].to_numpy(dtype=np.int64)) if len(keys) else np.zeros(len(query), dtype=bool)
    next_d = np.full(len(query), np.nan, dtype=np.float64)
    if found.any():
        next_d[found] = vals[positions[found]]
    query["d_next"] = next_d
    query["next_valid"] = found
    query["source_month"] = query["source_ord"].map(ordinal_to_month)
    query["target_month"] = query["target_ord"].map(ordinal_to_month)
    out = query.loc[np.isfinite(query["d"].to_numpy()) & np.isfinite(query["d_next"].to_numpy())].copy()
    out["sum_two_months"] = out["d"].to_numpy(dtype=np.float64) + out["d_next"].to_numpy(dtype=np.float64)
    out["opposite_sign"] = (out["d"].to_numpy(dtype=np.float64) * out["d_next"].to_numpy(dtype=np.float64)) < 0.0
    return out.reset_index(drop=True)


def reversal_metric_rows(pairs: pd.DataFrame, thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    bins: list[dict[str, Any]] = []
    for month, group in pairs.groupby("source_month", sort=True):
        d = group["d"].to_numpy(dtype=np.float64)
        nxt = group["d_next"].to_numpy(dtype=np.float64)
        sums = group["sum_two_months"].to_numpy(dtype=np.float64)
        opp = group["opposite_sign"].to_numpy(dtype=bool)
        denom = float(np.sum(np.square(d), dtype=np.float64))
        slope = float(np.sum(d * nxt, dtype=np.float64) / denom) if denom > 0 else None
        intercept = float(np.mean(nxt) - slope * np.mean(d)) if slope is not None else None
        metrics.append({
            "source_month": str(month), "target_month": str(group["target_month"].iloc[0]),
            "oracle_diagnostic": True, "rows": int(len(group)),
            "corr": float(np.corrcoef(d, nxt)[0, 1]) if len(group) > 1 and np.std(d) > 0 and np.std(nxt) > 0 else None,
            "slope_next_on_current": slope, "intercept": intercept,
            "opposite_sign_fraction": float(np.mean(opp)) if len(opp) else None,
            "sum_mean": float(np.mean(sums)) if len(sums) else None,
            "sum_variance": float(np.var(sums)) if len(sums) else None,
            "d_abs_q50": float(np.quantile(np.abs(d), 0.5)) if len(d) else None,
            "d_abs_q90": float(np.quantile(np.abs(d), 0.9)) if len(d) else None,
        })
        abs_d = np.abs(d)
        edges = [(-np.inf, thresholds["q50"]), (thresholds["q50"], thresholds["q75"]), (thresholds["q75"], thresholds["q90"]), (thresholds["q90"], np.inf)]
        names = ["low_lt_q50", "q50_to_q75", "q75_to_q90", "high_ge_q90"]
        for name, (lo, hi) in zip(names, edges, strict=True):
            mask = (abs_d >= lo) & (abs_d < hi if np.isfinite(hi) else np.ones(len(abs_d), dtype=bool))
            if not mask.any():
                continue
            bins.append({
                "source_month": str(month), "target_month": str(group["target_month"].iloc[0]),
                "delta_bin": name, "rows": int(mask.sum()),
                "corr": float(np.corrcoef(d[mask], nxt[mask])[0, 1]) if mask.sum() > 1 and np.std(d[mask]) > 0 and np.std(nxt[mask]) > 0 else None,
                "slope_next_on_current": float(np.sum(d[mask] * nxt[mask], dtype=np.float64) / np.sum(np.square(d[mask]), dtype=np.float64)) if np.sum(np.square(d[mask])) > 0 else None,
                "opposite_sign_fraction": float(np.mean(opp[mask])),
                "sum_mean": float(np.mean(sums[mask])), "sum_variance": float(np.var(sums[mask])),
            })
    return pd.DataFrame(metrics), pd.DataFrame(bins)


def weighted_proxy(group: pd.DataFrame) -> dict[str, Any]:
    h17 = group.loc[group["h"].between(1, 7)]
    horizons = sorted(int(v) for v in h17["h"].unique())
    required = list(range(1, 8))
    if horizons != required:
        return {"weighted_rmse_proxy": None, "weighted_complete": False, "weighted_missing_horizons": sorted(set(required) - set(horizons))}
    weighted_mse = 0.0
    for horizon in required:
        err = h17.loc[h17["h"] == horizon, "error"].to_numpy(dtype=np.float64)
        weighted_mse += float(TEST_H_WEIGHTS[horizon]) * float(np.mean(np.square(err), dtype=np.float64))
    return {"weighted_rmse_proxy": float(math.sqrt(weighted_mse)), "weighted_complete": True, "weighted_missing_horizons": []}


def oof_metric_tables(
    oof: pd.DataFrame,
    *,
    historical_delta_q90: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    extreme_rows: list[dict[str, Any]] = []
    age_rows: list[dict[str, Any]] = []
    oof = oof.copy()
    oof["source_month"] = month_label(oof["source_date"])
    oof["error"] = oof["prediction"].to_numpy(dtype=np.float64) - oof["target"].to_numpy(dtype=np.float64)
    oof["anchor_age_months"] = oof["source_ord"].to_numpy(dtype=np.int32) - month_ordinal(oof["last_observed_date"])
    oof["delta_extreme_q90_hist"] = np.abs(oof["d"].to_numpy(dtype=np.float64)) >= historical_delta_q90
    oof["cell_id"] = _cell_key(oof["lat"], oof["lon"])
    for origin, origin_frame in oof.groupby("origin", sort=True):
        for source_month, group in origin_frame.groupby("source_month", sort=True):
            scopes = {
                "all": group,
                "raw_h1_7": group.loc[group["h"].between(1, 7)],
                "h_gt7": group.loc[group["h"] > 7],
            }
            proxy = weighted_proxy(group)
            if proxy["weighted_complete"]:
                scopes["weighted_h1_7_proxy"] = group.loc[group["h"].between(1, 7)]
            for scope, sub in scopes.items():
                row = {"origin": origin, "source_month": source_month, "scope": scope, **metric_sums(sub["error"].to_numpy())}
                if scope == "weighted_h1_7_proxy":
                    row["weighted_rmse_proxy"] = proxy["weighted_rmse_proxy"]
                else:
                    row["weighted_rmse_proxy"] = None
                row["weighted_complete"] = bool(proxy["weighted_complete"]) if scope == "weighted_h1_7_proxy" else None
                row["weighted_missing_horizons"] = ",".join(map(str, proxy["weighted_missing_horizons"]))
                metric_rows.append(row)
            for horizon, hgroup in group.groupby("h", sort=True):
                metric_rows.append({"origin": origin, "source_month": source_month, "scope": "raw_h1_7_by_horizon" if int(horizon) <= 7 else "h_gt7_by_horizon", "horizon": int(horizon), **metric_sums(hgroup["error"].to_numpy()), "weighted_rmse_proxy": None, "weighted_complete": None, "weighted_missing_horizons": ""})
            for scope, sub in scopes.items():
                values = sub["error"].to_numpy(dtype=np.float64)
                cells = sub["cell_id"].to_numpy()
                decomp = cell_sse_decomposition(values, cells)
                spatial_rows.append({"origin": origin, "source_month": source_month, "scope": scope, **decomp})
            for flag, sub in group.groupby("delta_extreme_q90_hist", sort=True):
                for scope, scoped in {
                    "all": sub,
                    "raw_h1_7": sub.loc[sub["h"].between(1, 7)],
                    "h_gt7": sub.loc[sub["h"] > 7],
                }.items():
                    extreme_rows.append({"origin": origin, "source_month": source_month, "scope": scope, "delta_extreme_q90_hist": bool(flag), **metric_sums(scoped["error"].to_numpy())})
            age_bins = pd.cut(group["anchor_age_months"], bins=[-np.inf, 0, 1, 3, 6, np.inf], labels=["0", "1", "2-3", "4-6", "7+"])
            for age, sub in group.assign(anchor_age_bin=age_bins).groupby("anchor_age_bin", observed=False, sort=True):
                for scope, scoped in {
                    "all": sub,
                    "raw_h1_7": sub.loc[sub["h"].between(1, 7)],
                    "h_gt7": sub.loc[sub["h"] > 7],
                }.items():
                    age_rows.append({"origin": origin, "source_month": source_month, "scope": scope, "anchor_age_bin": str(age), **metric_sums(scoped["error"].to_numpy())})
    return pd.DataFrame(metric_rows), pd.DataFrame(spatial_rows), pd.DataFrame(extreme_rows), pd.DataFrame(age_rows)


def _svg_color(value: float, vmin: float, vmax: float) -> str:
    if not np.isfinite(value):
        return "#bdbdbd"
    t = float(np.clip((value - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0))
    if t < 0.5:
        u = t * 2.0
        r, g, b = int(40 + 180 * u), int(90 + 130 * u), 220
    else:
        u = (t - 0.5) * 2.0
        r, g, b = 220, int(220 - 130 * u), int(220 - 180 * u)
    return f"#{r:02x}{g:02x}{b:02x}"


def write_svg_map(frame: pd.DataFrame, path: Path, *, title: str, vmin: float, vmax: float) -> dict[str, Any]:
    width, height = 960, 520
    margin_x, margin_y = 55, 48
    plot_w, plot_h = width - 2 * margin_x, height - 2 * margin_y
    lon = frame["lon"].to_numpy(dtype=np.float64)
    lat = frame["lat"].to_numpy(dtype=np.float64)
    values = frame["d"].to_numpy(dtype=np.float64)
    x = margin_x + (lon + 180.0) / 360.0 * plot_w
    y = margin_y + (90.0 - lat) / 180.0 * plot_h
    clipped = int(np.sum((values < vmin) | (values > vmax)))
    circles = "\n".join(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="2.1" fill="{_svg_color(vv, vmin, vmax)}" />' for xx, yy, vv in zip(x, y, values, strict=True) if np.isfinite(vv))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin_x}" y="27" font-family="sans-serif" font-size="18" font-weight="bold">{title}</text>
<rect x="{margin_x}" y="{margin_y}" width="{plot_w}" height="{plot_h}" fill="#f7f7f7" stroke="#777" stroke-width="1"/>
{circles}
<text x="{margin_x}" y="{height-12}" font-family="sans-serif" font-size="11">longitude (fixed coordinates; wrap-safe distance)</text>
<text x="8" y="{margin_y+12}" font-family="sans-serif" font-size="11">90°N</text><text x="8" y="{height-margin_y}" font-family="sans-serif" font-size="11">90°S</text>
<text x="{width-180}" y="{height-22}" font-family="sans-serif" font-size="11">scale [{vmin:.3f}, {vmax:.3f}]</text>
</svg>\n'''
    path.write_text(svg, encoding="utf-8")
    return {"path": str(path), "rows": int(len(frame)), "clipped_rows": clipped, "vmin": vmin, "vmax": vmax}


def build_legal_anchor_state(train: pd.DataFrame, *, origin_ord: int) -> tuple[np.ndarray, np.ndarray]:
    """Latest visible TWS per location for a mask-block replay.

    Only rows at or before the replay source month are visible.  This function
    never receives target and therefore provides a target-blind invariance hook.
    """
    visible = train.loc[
        (train["source_ord"].to_numpy(dtype=np.int32) <= int(origin_ord))
        & np.isfinite(train["TWS_t"].to_numpy(dtype=np.float64)),
        ["location_id", "source_ord", "TWS_t"],
    ].sort_values(["location_id", "source_ord"], kind="mergesort")
    latest = visible.drop_duplicates("location_id", keep="last")
    tws = np.full(int(train["location_id"].max()) + 1, np.nan, dtype=np.float64)
    ords = np.full(len(tws), -10_000, dtype=np.int32)
    ids = latest["location_id"].to_numpy(dtype=np.int64)
    tws[ids] = latest["TWS_t"].to_numpy(dtype=np.float64)
    ords[ids] = latest["source_ord"].to_numpy(dtype=np.int32)
    return tws, ords


def replay_neighbor_support(
    oof: pd.DataFrame,
    train: pd.DataFrame,
    *,
    origin: str,
    origin_ord: int,
    neighbor_index: np.ndarray,
    neighbor_distances: np.ndarray,
    details: dict[str, Any],
) -> dict[str, Any]:
    tws, anchor_ord = build_legal_anchor_state(train, origin_ord=origin_ord)
    rows = oof.loc[oof["origin"] == origin].copy()
    loc = rows["location_id"].to_numpy(dtype=np.int64)
    safe = np.where(neighbor_index[loc] >= 0, neighbor_index[loc], 0)
    values = tws[safe]
    values[neighbor_index[loc] < 0] = np.nan
    count = np.isfinite(values).sum(axis=1)
    mean = np.divide(np.nansum(values, axis=1), count, out=np.full(len(rows), np.nan), where=count >= MIN_NEIGHBORS)
    source_ord = rows["source_ord"].to_numpy(dtype=np.int32)
    ages = (source_ord[:, None] - anchor_ord[safe]).astype(np.float64)
    ages[(neighbor_index[loc] < 0) | ~np.isfinite(values)] = np.nan
    supported = count >= MIN_NEIGHBORS
    focal = rows["last_observed_TWS"].to_numpy(dtype=np.float64)
    anchor_mismatch = float(np.max(np.abs(focal - tws[loc]))) if len(rows) else 0.0
    differs = supported & np.isfinite(focal) & (np.abs(mean - focal) > 1e-6)
    dists = neighbor_distances[loc][neighbor_index[loc] >= 0]
    return {
        "origin": origin,
        "state_kind": "LEGAL_SOURCE_DATE_TWS_NEIGHBOR_DIAGNOSTIC",
        "source_visibility_rule": "all Train rows with source_month <= origin; rows after origin masked",
        "ledger_details_path": details.get("path"),
        "view_withheld_window_rows": details.get("feature_fingerprint", {}).get("view_withheld_window_rows"),
        "rows": int(len(rows)),
        "focal_anchor_max_abs_mismatch": anchor_mismatch,
        "neighbor_supported_rows": int(supported.sum()),
        "neighbor_support_share": float(supported.mean()) if len(rows) else None,
        "neighbor_count_q50": float(np.quantile(count, 0.5)) if len(count) else None,
        "neighbor_count_q90": float(np.quantile(count, 0.9)) if len(count) else None,
        "neighbor_distance_q50_km": float(np.nanquantile(dists, 0.5)) if len(dists) else None,
        "neighbor_distance_q90_km": float(np.nanquantile(dists, 0.9)) if len(dists) else None,
        "neighbor_age_months_q50": float(np.nanquantile(ages, 0.5)) if np.isfinite(ages).any() else None,
        "neighbor_age_months_q90": float(np.nanquantile(ages, 0.9)) if np.isfinite(ages).any() else None,
        "neighbor_state_differs_from_focal_anchor_rows": int(differs.sum()),
        "neighbor_state_differs_from_focal_anchor_share_supported": float(differs.sum() / supported.sum()) if supported.any() else None,
        "candidate_is_nonempty": bool(supported.any() and differs.any()),
        "oracle_target_time_neighbor_used": False,
    }


def write_event(event_path: Path, phase: str, **payload: Any) -> None:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "phase": phase, **payload}
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")


def read_memory() -> dict[str, float | None]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0])
    except Exception:
        return {"available_gib": None}
    return {"available_gib": values.get("MemAvailable", 0) / 1024**2}


def max_rss_gib() -> float | None:
    if _resource is None:
        return None
    # Linux reports ru_maxrss in KiB (the execution host); macOS reports bytes.
    raw = float(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
    return raw / 1024**3 if sys.platform == "darwin" else raw / 1024**2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--capacity-dir", type=Path, required=True)
    parser.add_argument("--feature-schema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()

    train_path = (args.data_dir / "Train.csv").resolve()
    capacity_dir = args.capacity_dir.resolve()
    feature_schema_path = args.feature_schema.resolve()
    output_dir = args.output_dir.resolve()
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "maps").mkdir()
    events_path = output_dir / "events.jsonl"
    started = time.perf_counter()
    script_hash = sha256_file(Path(__file__).resolve())
    head = os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip()
    branch = os.popen(f"git -C {ROOT} branch --show-current").read().strip()
    if head != args.expected_commit:
        raise RuntimeError({"expected_commit": args.expected_commit, "actual_commit": head})
    config = {
        "train_only": True,
        "no_test_read": True,
        "no_model_fit": True,
        "no_predictions_or_submission": True,
        "suspicious_months": list(SUSPICIOUS_MONTHS),
        "historical_years": list(HISTORICAL_YEARS),
        "neighbor_k": NEIGHBOR_K,
        "neighbor_radius_km": NEIGHBOR_RADIUS_KM,
        "min_neighbors": MIN_NEIGHBORS,
        "cell_degrees": CELL_DEGREES,
        "oracle_diagnostic_spatial_reversal": True,
        "thread_policy": {"baseline": 12, "benchmarks_authorized": [12, 24, 48], "no_nested_oversubscription": True},
    }
    atomic_json(output_dir / "config.json", config)
    manifest: dict[str, Any] = {
        "status": "running", "stage": "LATE2015_SPATIAL_REVERSAL_DIAGNOSTIC",
        "started_at": datetime.now(timezone.utc).isoformat(), "commit": head,
        "branch": branch, "code_sha256": script_hash, "config": config,
        "input": {"train_path": str(train_path), "train_sha256": sha256_file(train_path)},
        "input_oof": {}, "resource_start": {"memory": read_memory(), "cpu_count": os.cpu_count()},
    }
    atomic_json(output_dir / "manifest.json", manifest)

    try:
        write_event(events_path, "train_read_start", path=str(train_path))
        train = pd.read_csv(train_path, usecols=TRAIN_COLUMNS)
        if train["sample_id"].duplicated().any() or train.duplicated(["lat", "lon", "time"]).any():
            raise AssertionError("Train sample IDs and location/calendar rows must be unique")
        train["source_date"] = pd.to_datetime(train["time"])
        train["source_ord"] = month_ordinal(train["source_date"])
        train["target_ord"] = train["source_ord"].to_numpy(dtype=np.int32) + 1
        train["source_month"] = month_label(train["source_date"])
        coords = train.loc[:, ["lat", "lon"]].drop_duplicates().sort_values(["lat", "lon"], kind="mergesort").reset_index(drop=True)
        coords["location_id"] = np.arange(len(coords), dtype=np.int32)
        train = train.merge(coords, on=["lat", "lon"], how="left", validate="many_to_one", sort=False)
        train["location_id"] = train["location_id"].astype(np.int32)
        train["d"] = (train["target"].to_numpy(dtype=np.float64) - train["TWS_t"].to_numpy(dtype=np.float64)).astype(np.float32)
        train["valid_d"] = np.isfinite(train["d"].to_numpy(dtype=np.float64))
        if train["valid_d"].sum() == 0:
            raise AssertionError("No finite Train d_t values")
        write_event(events_path, "train_read_complete", rows=len(train), locations=len(coords), months=int(train["source_ord"].nunique()), memory=read_memory())

        # Exact next-calendar target identity using integer location/month keys.
        keys_all = train["location_id"].to_numpy(dtype=np.int64) * 1000 + train["source_ord"].to_numpy(dtype=np.int64)
        order = np.argsort(keys_all, kind="mergesort")
        sorted_keys = keys_all[order]
        sorted_tws = train["TWS_t"].to_numpy(dtype=np.float64)[order]
        if np.any(np.diff(sorted_keys) == 0):
            raise AssertionError("duplicate location/source-month key")
        next_keys = train["location_id"].to_numpy(dtype=np.int64) * 1000 + train["target_ord"].to_numpy(dtype=np.int64)
        pos = np.searchsorted(sorted_keys, next_keys)
        next_exists = (pos < len(sorted_keys)) & (sorted_keys[np.minimum(pos, len(sorted_keys) - 1)] == next_keys)
        next_tws = np.full(len(train), np.nan, dtype=np.float64)
        next_tws[next_exists] = sorted_tws[pos[next_exists]]
        identity_diff = train["target"].to_numpy(dtype=np.float64) - next_tws
        train["next_source_exists"] = next_exists
        train["next_tws"] = next_tws.astype(np.float32)
        train["identity_abs_diff"] = np.abs(identity_diff).astype(np.float32)
        identity_mask = next_exists & np.isfinite(identity_diff)
        identity_summary = {
            "rows": int(len(train)), "next_source_exists": int(next_exists.sum()),
            "next_source_missing": int((~next_exists).sum()),
            "matched_target_identity_rows": int(identity_mask.sum()),
            "max_abs_diff": float(np.max(np.abs(identity_diff[identity_mask]))) if identity_mask.any() else None,
            "mean_abs_diff": float(np.mean(np.abs(identity_diff[identity_mask]))) if identity_mask.any() else None,
            "exact_within_1e-6": bool(np.all(np.abs(identity_diff[identity_mask]) <= 1e-6)) if identity_mask.any() else None,
        }
        atomic_json(output_dir / "exact_calendar_identity.json", identity_summary)

        canonical_columns = ["source_month", "source_ord", "target_ord", "location_id", "lat", "lon", "TWS_t", "target", "d", "next_source_exists", "next_tws", "identity_abs_diff"]
        canonical_path = output_dir / "canonical_train_only.csv.gz"
        train.loc[:, canonical_columns].to_csv(canonical_path, index=False, compression="gzip")
        month_support = train.groupby("source_month", sort=True).agg(
            rows=("source_month", "size"), locations=("location_id", "nunique"),
            valid_d_rows=("valid_d", "sum"), next_source_rows=("next_source_exists", "sum"),
            identity_max_abs_diff=("identity_abs_diff", "max"),
        ).reset_index()
        month_support["source_year"] = month_support["source_month"].str[:4].astype(int)
        month_support["calendar_month"] = month_support["source_month"].str[5:7].astype(int)
        month_support.to_csv(output_dir / "month_support.csv", index=False)

        neighbor_index, neighbor_distances = build_neighbor_index(coords["lat"].to_numpy(), coords["lon"].to_numpy())
        neighbor_pairs = []
        for focal in range(len(coords)):
            for rank, (other, distance) in enumerate(zip(neighbor_index[focal], neighbor_distances[focal], strict=True), start=1):
                if other >= 0:
                    neighbor_pairs.append({"location_id": focal, "neighbor_rank": rank, "neighbor_location_id": int(other), "distance_km": float(distance)})
        pd.DataFrame(neighbor_pairs).to_csv(output_dir / "fixed_neighbor_index.csv", index=False)
        neighbor_geometry = {
            "locations": int(len(coords)), "k": NEIGHBOR_K, "radius_km": NEIGHBOR_RADIUS_KM,
            "self_excluded": True, "longitude_wrap": "unit-vector spherical coordinates",
            "slots_within_radius": int(np.isfinite(neighbor_distances).sum()),
            "slots_total": int(neighbor_distances.size),
            "distance_quantiles_km": finite_quantiles(neighbor_distances, (0.0, 0.5, 0.9, 1.0)),
        }
        atomic_json(output_dir / "neighbor_geometry.json", neighbor_geometry)

        write_event(events_path, "spatial_start", locations=len(coords), months=int(train["source_ord"].nunique()), memory=read_memory())
        spatial_rows: list[dict[str, Any]] = []
        cell_frames: list[pd.DataFrame] = []
        for month, group in train.groupby("source_month", sort=True):
            result, cells = spatial_month_metrics(group, n_locations=len(coords), neighbor_index=neighbor_index, neighbor_distances=neighbor_distances)
            spatial_rows.append(result)
            cell_frames.append(cells)
        pd.DataFrame(spatial_rows).to_csv(output_dir / "spatial_month_metrics.csv", index=False)
        pd.concat(cell_frames, ignore_index=True).to_csv(output_dir / "spatial_cell_summary.csv", index=False)
        write_event(events_path, "spatial_complete", rows=len(spatial_rows), memory=read_memory())

        # Same-location, same-calendar-month paired controls.  No row shifting.
        pair_rows: list[dict[str, Any]] = []
        for current_month in SUSPICIOUS_MONTHS:
            current_ord = int(pd.Period(current_month, freq="M").ordinal + 1 + 12 * 0)  # replaced below for clarity
            current_dt = pd.Period(current_month, freq="M")
            current_ord = int(current_dt.year * 12 + current_dt.month)
            current = train.loc[train["source_ord"] == current_ord, ["location_id", "d", "valid_d"]].rename(columns={"d": "d_current", "valid_d": "current_valid"})
            for year in HISTORICAL_YEARS:
                hist_month = f"{year:04d}-{current_dt.month:02d}"
                hist_ord = year * 12 + current_dt.month
                history = train.loc[train["source_ord"] == hist_ord, ["location_id", "d", "valid_d"]].rename(columns={"d": "d_history", "valid_d": "history_valid"})
                joined = current.merge(history, on="location_id", how="outer", validate="one_to_one", sort=False)
                both = joined["d_current"].notna() & joined["d_history"].notna()
                diff = joined.loc[both, "d_current"].to_numpy(dtype=np.float64) - joined.loc[both, "d_history"].to_numpy(dtype=np.float64)
                pair_rows.append({
                    "current_month": current_month, "history_month": hist_month,
                    "current_rows": int(len(current)), "history_rows": int(len(history)),
                    "paired_valid_rows": int(both.sum()), "current_only_rows": int((joined["d_current"].notna() & ~joined["d_history"].notna()).sum()),
                    "history_only_rows": int((~joined["d_current"].notna() & joined["d_history"].notna()).sum()),
                    "difference_rmse": float(math.sqrt(np.mean(np.square(diff)))) if len(diff) else None,
                    "difference_bias": float(np.mean(diff)) if len(diff) else None,
                    "difference_corr": float(np.corrcoef(joined.loc[both, "d_current"], joined.loc[both, "d_history"])[0, 1]) if len(diff) > 1 and joined.loc[both, "d_current"].std() > 0 and joined.loc[both, "d_history"].std() > 0 else None,
                })
        pd.DataFrame(pair_rows).to_csv(output_dir / "same_location_calendar_pairs.csv", index=False)

        pairs = build_exact_calendar_reversal(train)
        hist_pairs = pairs.loc[pairs["source_month"].str[:4].astype(int).isin(HISTORICAL_YEARS)]
        abs_hist = np.abs(hist_pairs["d"].to_numpy(dtype=np.float64))
        thresholds = {
            "q50": float(np.quantile(abs_hist, 0.50)) if len(abs_hist) else 0.0,
            "q75": float(np.quantile(abs_hist, 0.75)) if len(abs_hist) else 0.0,
            "q90": float(np.quantile(abs_hist, 0.90)) if len(abs_hist) else 0.0,
            "estimated_from_source_years": list(HISTORICAL_YEARS),
            "historical_reversal_rows": int(len(hist_pairs)),
        }
        atomic_json(output_dir / "reversal_thresholds.json", thresholds)
        reversal_metrics, reversal_bins = reversal_metric_rows(pairs, thresholds)
        reversal_metrics.to_csv(output_dir / "reversal_month_metrics.csv", index=False)
        reversal_bins.to_csv(output_dir / "reversal_delta_bins.csv", index=False)
        reversal_support = {
            "canonical_rows_with_valid_d": int(train["valid_d"].sum()),
            "exact_next_source_rows": int(train["next_source_exists"].sum()),
            "reversal_pairs_both_valid": int(len(pairs)),
            "calendar_gap_excluded_rows": int(train["valid_d"].sum() - len(pairs)),
            "no_adjacent_row_join": True,
        }
        atomic_json(output_dir / "reversal_support.json", reversal_support)

        # Saved B3/98 OOF attribution.  These files are Train replay artifacts,
        # and are joined by exact sample_id before using location/month keys.
        oof_frames: list[pd.DataFrame] = []
        details_by_origin: dict[str, dict[str, Any]] = {}
        for origin, details_name in (("2014-04", "origin_2014-04_details.json"), ("2014-12", "origin_2014-12_details.json")):
            oof_path = capacity_dir / f"oof_{origin}_iter98.csv.gz"
            details_path = capacity_dir / details_name
            oof = pd.read_csv(oof_path, usecols=OOF_COLUMNS)
            if oof["sample_id"].duplicated().any():
                raise AssertionError(f"duplicate OOF sample IDs for {origin}")
            details = json.loads(details_path.read_text(encoding="utf-8")) if details_path.is_file() else {}
            details["path"] = str(details_path)
            details_by_origin[origin] = details
            manifest["input_oof"][origin] = {"path": str(oof_path), "sha256": sha256_file(oof_path), "rows": int(len(oof))}
            mapping = train.loc[:, ["sample_id", "source_date", "source_ord", "location_id", "lat", "lon", "target", "d", "TWS_t"]]
            joined = oof.merge(mapping, on="sample_id", how="left", validate="one_to_one", suffixes=("_oof", "_train"), sort=False)
            if joined["source_ord"].isna().any():
                raise AssertionError(f"OOF IDs missing from Train for {origin}")
            if not (pd.to_datetime(joined["source_date_oof"]).dt.to_period("M").astype(str) == joined["source_ord"].map(ordinal_to_month)).all():
                raise AssertionError(f"OOF source dates do not resolve to exact Train calendar keys for {origin}")
            if not np.allclose(joined["target_oof"].to_numpy(dtype=np.float64), joined["target_train"].to_numpy(dtype=np.float64), atol=1e-6, rtol=0.0):
                raise AssertionError(f"OOF targets differ from Train targets for {origin}")
            joined["origin"] = origin
            joined["source_month"] = joined["source_ord"].map(ordinal_to_month)
            joined["last_observed_date"] = pd.to_datetime(joined["last_observed_date"])
            joined["target"] = joined["target_oof"].to_numpy(dtype=np.float64)
            joined["prediction"] = joined["prediction"].to_numpy(dtype=np.float64)
            joined["d"] = joined["d"].to_numpy(dtype=np.float64)
            joined = joined.rename(columns={"h": "h"})
            oof_frames.append(joined.loc[:, ["sample_id", "origin", "source_date_oof", "target_date", "last_observed_date", "last_observed_TWS", "h", "lat_oof", "lon_oof", "target", "prediction", "source_ord", "location_id", "d"]].rename(columns={"source_date_oof": "source_date", "lat_oof": "lat", "lon_oof": "lon"}))
        oof_all = pd.concat(oof_frames, ignore_index=True)
        q90 = thresholds["q90"]
        oof_all["source_month"] = month_label(oof_all["source_date"])
        oof_all["error"] = oof_all["prediction"].to_numpy(dtype=np.float64) - oof_all["target"].to_numpy(dtype=np.float64)
        oof_all["anchor_age_months"] = oof_all["source_ord"].to_numpy(dtype=np.int32) - month_ordinal(oof_all["last_observed_date"])
        metric_table, oof_spatial, oof_extreme, oof_age = oof_metric_tables(oof_all, historical_delta_q90=q90)
        metric_table.to_csv(output_dir / "oof_error_metrics.csv", index=False)
        oof_spatial.to_csv(output_dir / "oof_spatial_attribution.csv", index=False)
        oof_extreme.to_csv(output_dir / "oof_extreme_delta.csv", index=False)
        oof_age.to_csv(output_dir / "oof_anchor_age.csv", index=False)
        oof_all.loc[:, ["sample_id", "origin", "source_month", "h", "error", "d", "anchor_age_months"]].to_csv(output_dir / "oof_attribution_rows.csv.gz", index=False, compression="gzip")

        # Frozen 446-column schema audit.  The schema contains hydro regional
        # aggregates but no neighboring TWS state/date; focal anchor is explicit.
        schema = json.loads(feature_schema_path.read_text(encoding="utf-8"))
        names = list(schema["feature_names"])
        if len(names) != 446:
            raise AssertionError(f"expected 446 B3 names, found {len(names)}")
        tws_names = [n for n in names if any(token in n.lower() for token in ("tws", "last_observed", "anchor"))]
        neighbor_names = [n for n in names if any(token in n.lower() for token in ("neighbor", "neighbour"))]
        regional_names = [n for n in names if n.startswith(("reg5_", "dev5_", "count5_", "coverage5_", "reg15_", "dev15_", "count15_", "coverage15_", "regional5_", "regional15_", "local_minus_reg"))]
        focal_names = [n for n in names if n in {"last_observed_TWS", "h"} or "last_observed" in n]
        feature_audit = {
            "feature_count": len(names), "feature_names_sha256": sha256_values(names),
            "focal_anchor_feature_names": focal_names, "focal_anchor_feature_count": len(focal_names),
            "tws_or_anchor_feature_names": tws_names, "neighbor_tws_feature_names": neighbor_names,
            "regional_feature_count": len(regional_names), "regional_features_are_hydro_only": True,
            "focal_anchor_date_is_metadata_not_neighbor_date": True,
            "legal_neighbor_tws_not_currently_represented": len(neighbor_names) == 0,
            "oracle_target_time_neighbor_used": False,
        }
        atomic_json(output_dir / "b3_feature_audit.json", feature_audit)

        support_rows: list[dict[str, Any]] = []
        for origin in ("2014-04", "2014-12"):
            support_rows.append(replay_neighbor_support(oof_all, train, origin=origin, origin_ord=int(pd.Period(origin, freq="M").year * 12 + pd.Period(origin, freq="M").month), neighbor_index=neighbor_index, neighbor_distances=neighbor_distances, details=details_by_origin[origin]))
        pd.DataFrame(support_rows).to_csv(output_dir / "legal_neighbor_replay_support.csv", index=False)

        # Invariance checks: future TWS/target perturbations cannot alter the
        # legal source-date anchor state, and exact-calendar reversal never
        # bridges a missing month.
        base_state, base_ord = build_legal_anchor_state(train, origin_ord=int(pd.Period("2014-12", freq="M").year * 12 + 12))
        perturbed = train.loc[:, ["location_id", "source_ord", "TWS_t"]].copy()
        future = perturbed["source_ord"].to_numpy(dtype=np.int32) > (2014 * 12 + 12)
        perturbed.loc[future, "TWS_t"] = perturbed.loc[future, "TWS_t"] + 777.0
        perturbed_state, perturbed_ord = build_legal_anchor_state(perturbed, origin_ord=2014 * 12 + 12)
        invariance = {
            "future_tws_perturbation_rows": int(future.sum()),
            "legal_anchor_tws_max_abs_difference": float(np.nanmax(np.abs(base_state - perturbed_state))),
            "legal_anchor_date_max_difference_months": int(np.max(np.abs(base_ord - perturbed_ord))),
            "target_not_an_input_to_legal_state_builder": True,
            "reversal_missing_month_bridge_count": 0,
            "neighbor_self_exclusion": bool(not np.any(neighbor_index == np.arange(len(coords), dtype=np.int32)[:, None])),
            "longitude_wrap_uses_unit_vectors": True,
            "cell_sse_conservation_max_abs_error": float(max(float(x.get("sse_conservation_abs_error", 0.0)) for x in spatial_rows)),
        }
        atomic_json(output_dir / "invariance_checks.json", invariance)

        # Shared-scale maps: three suspicious months and three earlier same-month
        # controls.  SVG keeps the artifact readable without installing a plotting stack.
        selected = list(SUSPICIOUS_MONTHS)
        for candidate in ("2012-01", "2012-02", "2012-06", "2011-01", "2011-02", "2011-06"):
            if candidate not in selected and candidate in set(train["source_month"].unique()):
                selected.append(candidate)
        selected = selected[:6]
        selected_values = train.loc[train["source_month"].isin(selected), "d"].to_numpy(dtype=np.float64)
        finite_selected = selected_values[np.isfinite(selected_values)]
        scale = float(np.quantile(np.abs(finite_selected), 0.99)) if len(finite_selected) else 1.0
        scale = max(scale, 1e-6)
        map_manifest = {"shared_vmin": -scale, "shared_vmax": scale, "months": []}
        for month in selected:
            frame = train.loc[train["source_month"] == month, ["lat", "lon", "d"]]
            map_manifest["months"].append(write_svg_map(frame, output_dir / "maps" / f"d_{month}.svg", title=f"Train d_t = target - TWS_t | {month} | ORACLE_DIAGNOSTIC", vmin=-scale, vmax=scale))
        atomic_json(output_dir / "maps" / "map_manifest.json", map_manifest)

        manifest.update({
            "status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "resource_end": {"memory": read_memory(), "max_rss_gib": max_rss_gib()},
            "outputs": {p.name: {"bytes": int(p.stat().st_size), "sha256": sha256_file(p)} for p in output_dir.iterdir() if p.is_file()},
            "exclusions": {
                "test_rows_read": 0, "test_labels_read": False, "model_fits": 0,
                "submission_files_written": 0, "calendar_row_shift_used": False,
                "future_or_masked_values_used_as_predictors": False,
            },
            "identity_summary": identity_summary, "neighbor_geometry": neighbor_geometry,
            "feature_audit": feature_audit, "replay_neighbor_support": support_rows,
        })
        atomic_json(output_dir / "manifest.json", manifest)
        write_event(events_path, "completed", elapsed_seconds=manifest["elapsed_seconds"], memory=read_memory())
        print(json.dumps(_json_safe(manifest), sort_keys=True), flush=True)
    except Exception as exc:
        manifest.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.perf_counter() - started, "resource_end": {"memory": read_memory(), "max_rss_gib": max_rss_gib()}})
        atomic_json(output_dir / "manifest.json", manifest)
        write_event(events_path, "failed", error=manifest["error"])
        raise


if __name__ == "__main__":
    main()
