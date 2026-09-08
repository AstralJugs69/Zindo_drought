"""Target-blind, calendar-aware hydrological trajectory features.

All summaries are causal: a row at month ``t`` only reads its variable's finite
observations in the calendar interval ending at ``t``.  Missing source months are
not compressed into adjacent observations and insufficient windows remain NaN.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.regional_context import HYDRO_COLUMNS, regional_feature_names

WINDOWS = (3, 6)


def _assert_target_blind(frame: pd.DataFrame, name: str) -> None:
    forbidden = {c for c in frame.columns if c.lower() in {"target", "y", "label"}}
    if forbidden:
        raise AssertionError(f"{name} must be target-blind; found {sorted(forbidden)}")


def trajectory_feature_names(prefix: str) -> list[str]:
    """Return the stable trajectory schema for a source-panel prefix."""
    names: list[str] = []
    for variable in HYDRO_COLUMNS:
        for window in WINDOWS:
            stem = f"{prefix}{variable}_trail{window}_"
            names.extend([
                f"{stem}mean", f"{stem}std", f"{stem}min", f"{stem}max",
                f"{stem}slope", f"{stem}current_minus_mean", f"{stem}count",
                f"{stem}coverage",
            ])
        names.extend([
            f"{prefix}{variable}_recent_change",
            f"{prefix}{variable}_recent_elapsed_months",
            f"{prefix}{variable}_slope_change_3_6",
            f"{prefix}{variable}_drying_recovery_sign",
            f"{prefix}{variable}_observed_run_length",
            f"{prefix}{variable}_age_months",
        ])
    return names


def _calendar_ordinal(times: pd.Series) -> np.ndarray:
    periods = pd.to_datetime(times).dt.to_period("M")
    return periods.astype("int64").to_numpy(dtype=np.int64)


def _range_sum(prefix: np.ndarray, starts: np.ndarray) -> np.ndarray:
    previous = np.zeros(len(starts), dtype=np.float64)
    has_previous = starts > 0
    previous[has_previous] = prefix[starts[has_previous] - 1]
    return prefix - previous


def _variable_features(periods: np.ndarray, values: np.ndarray, prefix: str, variable: str) -> dict[str, np.ndarray]:
    """Calculate one variable's causal calendar summaries for a sorted group."""
    n_rows = len(values)
    finite = np.isfinite(values)
    values64 = values.astype(np.float64, copy=False)
    observed = np.where(finite, values64, 0.0)
    obs_count = finite.astype(np.float64)
    x = periods.astype(np.float64)
    prefix_count = np.cumsum(obs_count)
    prefix_y = np.cumsum(observed)
    prefix_y2 = np.cumsum(np.square(observed))
    prefix_x = np.cumsum(x * obs_count)
    prefix_x2 = np.cumsum(np.square(x) * obs_count)
    prefix_xy = np.cumsum(x * observed)
    output: dict[str, np.ndarray] = {}
    slopes: dict[int, np.ndarray] = {}

    for window in WINDOWS:
        starts = np.searchsorted(periods, periods - (window - 1), side="left")
        count = _range_sum(prefix_count, starts)
        sum_y = _range_sum(prefix_y, starts)
        sum_y2 = _range_sum(prefix_y2, starts)
        sum_x = _range_sum(prefix_x, starts)
        sum_x2 = _range_sum(prefix_x2, starts)
        sum_xy = _range_sum(prefix_xy, starts)
        mean = np.divide(sum_y, count, out=np.full(n_rows, np.nan), where=count > 0)
        variance = np.divide(sum_y2, count, out=np.full(n_rows, np.nan), where=count > 0) - np.square(mean)
        std = np.sqrt(np.maximum(variance, 0.0))
        denom = count * sum_x2 - np.square(sum_x)
        slope = np.divide(count * sum_xy - sum_x * sum_y, denom, out=np.full(n_rows, np.nan), where=denom > 0)
        minimum = np.full(n_rows, np.nan)
        maximum = np.full(n_rows, np.nan)
        for index, start in enumerate(starts):
            segment = values64[start:index + 1]
            segment = segment[np.isfinite(segment)]
            if len(segment):
                minimum[index] = float(segment.min())
                maximum[index] = float(segment.max())
        current_minus_mean = np.where(finite, values64 - mean, np.nan)
        stem = f"{prefix}{variable}_trail{window}_"
        output.update({
            f"{stem}mean": mean,
            f"{stem}std": std,
            f"{stem}min": minimum,
            f"{stem}max": maximum,
            f"{stem}slope": slope,
            f"{stem}current_minus_mean": current_minus_mean,
            f"{stem}count": count,
            f"{stem}coverage": count / float(window),
        })
        slopes[window] = slope

    previous_value = np.full(n_rows, np.nan)
    previous_period = np.full(n_rows, np.nan)
    latest_value = np.nan
    latest_period = np.nan
    run_length = np.zeros(n_rows, dtype=np.float64)
    previous_was_observed = False
    previous_row_period = np.nan
    previous_run_length = 0.0
    for index, (period, value, is_finite) in enumerate(zip(periods, values64, finite, strict=True)):
        previous_value[index] = latest_value
        previous_period[index] = latest_period
        if is_finite:
            contiguous = previous_was_observed and period == previous_row_period + 1
            run_length[index] = previous_run_length + 1.0 if contiguous else 1.0
            latest_value, latest_period = value, float(period)
            previous_was_observed, previous_row_period, previous_run_length = True, float(period), run_length[index]
        else:
            previous_was_observed, previous_row_period, previous_run_length = False, float(period), 0.0
    latest_period_at_or_before = np.where(finite, periods.astype(np.float64), previous_period)
    age = periods.astype(np.float64) - latest_period_at_or_before
    age[~np.isfinite(latest_period_at_or_before)] = np.nan
    recent_change = np.where(finite & np.isfinite(previous_value), values64 - previous_value, np.nan)
    recent_elapsed = np.where(finite & np.isfinite(previous_period), periods - previous_period, np.nan).astype(np.float64)
    slope_change = slopes[3] - slopes[6]
    output.update({
        f"{prefix}{variable}_recent_change": recent_change,
        f"{prefix}{variable}_recent_elapsed_months": recent_elapsed,
        f"{prefix}{variable}_slope_change_3_6": slope_change,
        f"{prefix}{variable}_drying_recovery_sign": np.sign(recent_change),
        f"{prefix}{variable}_observed_run_length": run_length,
        f"{prefix}{variable}_age_months": age,
    })
    return output


def build_hydro_trajectory_map(
    source: pd.DataFrame,
    *,
    group_columns: Iterable[str] = ("lat", "lon"),
    prefix: str = "local_",
) -> pd.DataFrame:
    """Build a keyed, causal trajectory map from source covariates alone.

    ``group_columns`` can be location coordinates or regional bin coordinates.
    The output is invariant to input-row order and to changes in later calendar
    observations when restricted to earlier rows.
    """
    _assert_target_blind(source, "source")
    group_columns = tuple(group_columns)
    required = {"sample_id", "time", *group_columns, *HYDRO_COLUMNS}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"source missing required columns: {sorted(missing)}")
    if source.sample_id.duplicated().any():
        raise AssertionError("source sample_id must be unique")
    x = source.loc[:, ["sample_id", "time", *group_columns, *HYDRO_COLUMNS]].copy()
    x["_period"] = _calendar_ordinal(x.time)
    if x.duplicated([*group_columns, "_period"]).any():
        raise AssertionError("source has duplicate group/calendar observations")
    x["_order"] = np.arange(len(x), dtype=np.int64)
    x = x.sort_values([*group_columns, "_period", "_order"], kind="mergesort").reset_index(drop=True)
    features = {name: np.full(len(x), np.nan, dtype=np.float64) for name in trajectory_feature_names(prefix)}
    for _, index in x.groupby(list(group_columns), sort=False).groups.items():
        positions = np.asarray(list(index), dtype=np.int64)
        periods = x.loc[positions, "_period"].to_numpy(dtype=np.int64)
        for variable in HYDRO_COLUMNS:
            result = _variable_features(periods, x.loc[positions, variable].to_numpy(dtype=np.float64), prefix, variable)
            for name, values in result.items():
                features[name][positions] = values
    out = pd.DataFrame({"sample_id": x.sample_id.to_numpy(), **features})
    out["_order"] = x._order.to_numpy()
    out = out.sort_values("_order", kind="mergesort").drop(columns="_order").reset_index(drop=True)
    if out.columns.tolist() != ["sample_id", *trajectory_feature_names(prefix)]:
        raise AssertionError("trajectory schema/order drift")
    if np.isinf(out.drop(columns="sample_id").to_numpy(dtype=np.float64)).any():
        raise AssertionError("trajectory feature map contains infinite values")
    return out.astype({name: "float32" for name in trajectory_feature_names(prefix)})


def attach_trajectory_features(sample_ids: pd.Series, trajectories: pd.DataFrame) -> pd.DataFrame:
    """Attach a precomputed trajectory map with stable sample-id alignment."""
    if trajectories.sample_id.duplicated().any():
        raise AssertionError("trajectory map sample_id must be unique")
    requested = pd.DataFrame({"sample_id": sample_ids.astype(str).to_numpy(), "_order": np.arange(len(sample_ids))})
    right = trajectories.copy(); right["sample_id"] = right.sample_id.astype(str)
    joined = requested.merge(right, how="left", on="sample_id", validate="many_to_one", sort=False)
    if joined.drop(columns=["sample_id", "_order"]).isna().all(axis=None):
        raise AssertionError("trajectory attachment found no matching source rows")
    return joined.sort_values("_order", kind="mergesort").drop(columns=["sample_id", "_order"]).reset_index(drop=True)


def _regional_panel(source: pd.DataFrame, regional: pd.DataFrame, width: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    tag = f"{float(width):g}"
    x = source.loc[:, ["sample_id", "time", "lat", "lon"]].copy()
    x["_region_lat"] = np.floor(x.lat.astype(float) / width) * width
    x["_region_lon"] = np.floor(x.lon.astype(float) / width) * width
    values = regional.loc[:, ["sample_id", *[f"reg{tag}_{v}" for v in HYDRO_COLUMNS]]].copy()
    x = x.merge(values, on="sample_id", how="left", validate="one_to_one", sort=False)
    rename = {f"reg{tag}_{v}": v for v in HYDRO_COLUMNS}
    x = x.rename(columns=rename)
    x["_period"] = pd.to_datetime(x.time).dt.to_period("M")
    unique = x.drop_duplicates(["_period", "_region_lat", "_region_lon"], keep="first").copy()
    unique["sample_id"] = (
        "region_" + tag + "_" + unique._period.astype(str) + "_"
        + unique._region_lat.astype(str) + "_" + unique._region_lon.astype(str)
    )
    lookup = x.loc[:, ["sample_id", "_period", "_region_lat", "_region_lon"]].copy()
    return unique.drop(columns="_period"), lookup


def build_regional_trajectory_map(
    source: pd.DataFrame,
    regional: pd.DataFrame,
    local_trajectories: pd.DataFrame,
    *,
    widths: Iterable[float] = (5.0, 15.0),
) -> pd.DataFrame:
    """Build regional trajectory and local-minus-regional trend features.

    Regional time series are reduced to one legal source-covariate aggregate per
    calendar month and bin before their trailing summaries are calculated.  No TWS
    or target field enters this construction.
    """
    _assert_target_blind(source, "source")
    _assert_target_blind(regional, "regional")
    _assert_target_blind(local_trajectories, "local_trajectories")
    required = {"sample_id", "time", "lat", "lon", *HYDRO_COLUMNS}
    if missing := required.difference(source.columns):
        raise ValueError(f"source missing required columns: {sorted(missing)}")
    expected_regional = {"sample_id", *regional_feature_names(widths)}
    if missing := expected_regional.difference(regional.columns):
        raise ValueError(f"regional context missing required columns: {sorted(missing)}")
    out = pd.DataFrame({"sample_id": source.sample_id.to_numpy()})
    local = local_trajectories.set_index("sample_id")
    for width in widths:
        tag = f"{float(width):g}"
        panel, lookup = _regional_panel(source, regional, width)
        prefix = f"regional{tag}_"
        trajectory = build_hydro_trajectory_map(
            panel, group_columns=("_region_lat", "_region_lon"), prefix=prefix,
        )
        panel["_period"] = pd.to_datetime(panel.time).dt.to_period("M")
        keyed = panel.loc[:, ["sample_id", "_period", "_region_lat", "_region_lon"]].merge(
            trajectory, on="sample_id", how="left", validate="one_to_one", sort=False,
        )
        attached = lookup.merge(
            keyed.drop(columns="sample_id"), on=["_period", "_region_lat", "_region_lon"],
            how="left", validate="many_to_one", sort=False,
        ).drop(columns=["_period", "_region_lat", "_region_lon"])
        attached = attached.set_index("sample_id").loc[source.sample_id].reset_index(drop=True)
        out = pd.concat([out, attached], axis=1)
        for variable in HYDRO_COLUMNS:
            for window in WINDOWS:
                local_mean = local.loc[source.sample_id, f"local_{variable}_trail{window}_mean"].to_numpy(dtype=np.float64)
                local_slope = local.loc[source.sample_id, f"local_{variable}_trail{window}_slope"].to_numpy(dtype=np.float64)
                reg_mean = attached[f"regional{tag}_{variable}_trail{window}_mean"].to_numpy(dtype=np.float64)
                reg_slope = attached[f"regional{tag}_{variable}_trail{window}_slope"].to_numpy(dtype=np.float64)
                out[f"local_minus_reg{tag}_{variable}_trail{window}_mean"] = local_mean - reg_mean
                out[f"local_minus_reg{tag}_{variable}_trail{window}_slope"] = local_slope - reg_slope
                signs_known = np.isfinite(local_slope) & np.isfinite(reg_slope)
                disagreement = np.full(len(out), np.nan, dtype=np.float64)
                disagreement[signs_known] = (np.sign(local_slope[signs_known]) != np.sign(reg_slope[signs_known])).astype(float)
                out[f"local_reg{tag}_{variable}_trail{window}_direction_disagree"] = disagreement
    if out.sample_id.duplicated().any():
        raise AssertionError("regional trajectory output sample_id must be unique")
    if np.isinf(out.drop(columns="sample_id").to_numpy(dtype=np.float64)).any():
        raise AssertionError("regional trajectory map contains infinite values")
    return out.astype({c: "float32" for c in out.columns if c != "sample_id"})
