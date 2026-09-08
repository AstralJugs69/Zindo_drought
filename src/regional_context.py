"""Target-blind contemporaneous regional hydrology features."""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

HYDRO_COLUMNS = (
    "SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"
)
DEFAULT_WIDTHS = (5.0, 15.0)


def regional_feature_names(widths: Iterable[float] = DEFAULT_WIDTHS) -> list[str]:
    names: list[str] = []
    for width in widths:
        tag = f"{float(width):g}"
        for col in HYDRO_COLUMNS:
            names.extend([
                f"reg{tag}_{col}", f"dev{tag}_{col}",
                f"count{tag}_{col}", f"coverage{tag}_{col}",
            ])
    return names


def build_regional_context(source: pd.DataFrame, *, widths: tuple[float, ...] = DEFAULT_WIDTHS) -> pd.DataFrame:
    """Aggregate only rows actually present in each supplied source-date field.

    The builder never accepts TWS or target columns. Missing regional means stay
    missing; counts and coverage preserve the source-field availability contract.
    """
    required = {"sample_id", "time", "lat", "lon", *HYDRO_COLUMNS}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"source missing required columns: {sorted(missing)}")
    forbidden = {"target", "TWS_t"}.intersection(source.columns)
    if forbidden:
        raise AssertionError(f"regional builder must not receive target/TWS columns: {sorted(forbidden)}")
    if source["sample_id"].duplicated().any():
        raise AssertionError("sample_id must be unique")

    x = source.loc[:, ["sample_id", "time", "lat", "lon", *HYDRO_COLUMNS]].copy()
    x["_order"] = np.arange(len(x), dtype=np.int64)
    x["source_period"] = pd.to_datetime(x["time"]).dt.to_period("M")
    blocks: list[pd.DataFrame] = []
    for width in widths:
        if width <= 0:
            raise ValueError("regional widths must be positive")
        tag = f"{float(width):g}"
        x["_rb_lat"] = np.floor(x["lat"].astype(float) / width) * width
        x["_rb_lon"] = np.floor(x["lon"].astype(float) / width) * width
        keys = ["source_period", "_rb_lat", "_rb_lon"]
        group_size = x.groupby(keys, sort=False)["sample_id"].transform("size").to_numpy(np.float64)
        out = pd.DataFrame(index=x.index)
        for col in HYDRO_COLUMNS:
            g = x.groupby(keys, sort=False)[col]
            mean = g.transform("mean").astype(np.float64)
            count = g.transform("count").astype(np.float64)
            out[f"reg{tag}_{col}"] = mean
            out[f"dev{tag}_{col}"] = x[col].astype(np.float64) - mean
            out[f"count{tag}_{col}"] = count
            out[f"coverage{tag}_{col}"] = count.to_numpy(np.float64) / np.maximum(group_size, 1.0)
        blocks.append(out)

    result = pd.concat([x.loc[:, ["sample_id", "_order"]], *blocks], axis=1)
    result = result.sort_values("_order", kind="mergesort").drop(columns="_order").reset_index(drop=True)
    expected = ["sample_id", *regional_feature_names(widths)]
    if result.columns.tolist() != expected:
        raise AssertionError("regional feature schema/order drift")
    return result


def attach_regional_features(sample_ids: pd.Series, regional: pd.DataFrame) -> pd.DataFrame:
    """Stable keyed join used identically for training and inference ledgers."""
    if regional["sample_id"].duplicated().any():
        raise AssertionError("regional map sample_id must be unique")
    order = pd.DataFrame({"sample_id": sample_ids.astype(str).to_numpy(), "_order": np.arange(len(sample_ids))})
    r = regional.copy()
    r["sample_id"] = r["sample_id"].astype(str)
    joined = order.merge(r, on="sample_id", how="left", validate="many_to_one", sort=False)
    joined = joined.sort_values("_order", kind="mergesort")
    return joined.drop(columns=["sample_id", "_order"]).reset_index(drop=True)


def regional_bytes_hash(frame: pd.DataFrame) -> str:
    """Hash schema, NaN mask and numeric bytes for paired-support provenance."""
    arr = frame.to_numpy(np.float64)
    h = hashlib.sha256("\n".join(frame.columns).encode("utf-8"))
    h.update(np.ascontiguousarray(np.isnan(arr)).tobytes())
    h.update(np.ascontiguousarray(np.nan_to_num(arr, nan=0.0)).tobytes())
    return h.hexdigest()
