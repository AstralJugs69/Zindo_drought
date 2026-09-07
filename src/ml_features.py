from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics import TEST_H_WEIGHTS


CORE_FEATURE_COLUMNS = [
    "last_observed_TWS",
    "h",
    "lat",
    "lon",
    "month_sin",
    "month_cos",
]

SOURCE_CORE_COLUMNS = ["sample_id", "month_sin", "month_cos"]


@dataclass(frozen=True)
class SampledTrainingRows:
    rows: pd.DataFrame
    horizon_counts: dict[int, int]
    dropped_missing_anchor: int


def _assert_target_blind(frame: pd.DataFrame, name: str) -> None:
    forbidden = {c for c in frame.columns if c.lower() in {"target", "y", "label"}}
    if forbidden:
        raise AssertionError(f"{name} must be target-blind; found {sorted(forbidden)}")


def deterministic_horizon_sample(
    sample_ids: pd.Series,
    *,
    seed: int = 20260907,
) -> np.ndarray:
    """Assign exactly one deterministic synthetic h to every source row.

    The hash is stable for a fixed sample_id/seed and horizons are sampled from the
    exact row-level test distribution. No target or feature value participates.
    """
    key = sample_ids.astype(str) + f"__seed{int(seed)}"
    hashes = pd.util.hash_pandas_object(key, index=False).to_numpy(dtype=np.uint64)
    # Convert uint64 to [0, 1) without losing deterministic ordering.
    u = hashes.astype(np.float64) / float(2**64)
    horizons = np.array(sorted(TEST_H_WEIGHTS), dtype=np.int8)
    cumulative = np.cumsum([TEST_H_WEIGHTS[int(h)] for h in horizons])
    cumulative[-1] = 1.0
    return horizons[np.searchsorted(cumulative, u, side="right")]


def build_sampled_training_rows(
    structural: pd.DataFrame,
    source_features: pd.DataFrame,
    *,
    max_target_month: str | pd.Period,
    seed: int = 20260907,
) -> SampledTrainingRows:
    """Build one target-blind sampled-horizon training row per eligible source row.

    Only rows whose supervised target month is <= max_target_month are eligible.
    For each source row, one h is deterministically sampled from the exact test
    horizon mixture. The TWS anchor is calendar month target-h.
    """
    _assert_target_blind(structural, "structural")
    _assert_target_blind(source_features, "source_features")

    required_state = {"sample_id", "time", "lat", "lon", "TWS_t"}
    missing = required_state.difference(structural.columns)
    if missing:
        raise ValueError(f"Missing structural columns: {sorted(missing)}")
    missing_source = set(SOURCE_CORE_COLUMNS).difference(source_features.columns)
    if missing_source:
        raise ValueError(f"Missing source feature columns: {sorted(missing_source)}")
    if source_features["sample_id"].duplicated().any():
        raise AssertionError("source_features sample_id must be unique")

    cutoff = pd.Period(max_target_month, freq="M")
    state = structural.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
    state["source_date"] = pd.to_datetime(state["time"])
    state["source_period"] = state["source_date"].dt.to_period("M")
    state["target_period"] = state["source_period"] + 1

    eligible = state.loc[state["target_period"] <= cutoff].copy()
    if eligible.empty:
        raise AssertionError(f"No training rows have target month <= {cutoff}")
    if not (eligible["source_period"] < cutoff).all():
        raise AssertionError("Training source month reached/passed target cutoff")

    eligible["h"] = deterministic_horizon_sample(eligible["sample_id"], seed=seed)
    eligible["anchor_period"] = [
        p + (1 - int(h)) for p, h in zip(eligible["source_period"], eligible["h"])
    ]

    anchor = state.loc[:, ["lat", "lon", "source_period", "TWS_t"]].rename(
        columns={"source_period": "anchor_period", "TWS_t": "last_observed_TWS"}
    )
    x = eligible.merge(
        anchor,
        how="left",
        on=["lat", "lon", "anchor_period"],
        validate="many_to_one",
        sort=False,
    )
    dropped = int(x["last_observed_TWS"].isna().sum())
    x = x.loc[x["last_observed_TWS"].notna()].copy()
    if x.empty:
        raise AssertionError("All sampled training rows lack a legal TWS anchor")

    # Fresh source-month calendar features are joined by sample_id. Target is not
    # present in either input table and therefore cannot leak into this function.
    x = x.merge(
        source_features.loc[:, SOURCE_CORE_COLUMNS],
        how="left",
        on="sample_id",
        validate="one_to_one",
        sort=False,
    )
    if x[["month_sin", "month_cos"]].isna().any().any():
        raise AssertionError("Missing source-month calendar features after join")

    x["last_observed_date"] = x["anchor_period"].dt.to_timestamp(how="start")
    if ((x["h"] > 1) & (x["last_observed_date"] == x["source_date"])).any():
        raise AssertionError("h>1 training row uses source-month TWS")
    if not (x["anchor_period"] <= x["source_period"]).all():
        raise AssertionError("Training anchor lies in the future")
    if "target" in x.columns:
        raise AssertionError("Target leaked into sampled training rows")

    keep = [
        "sample_id",
        "source_date",
        "source_period",
        "target_period",
        "lat",
        "lon",
        "h",
        "last_observed_date",
        "last_observed_TWS",
        "month_sin",
        "month_cos",
    ]
    x = x.loc[:, keep].reset_index(drop=True)
    counts = {int(k): int(v) for k, v in x["h"].value_counts().sort_index().items()}
    return SampledTrainingRows(rows=x, horizon_counts=counts, dropped_missing_anchor=dropped)


def build_core_feature_matrix(
    ledger: pd.DataFrame,
    source_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build EXP001 features without ever receiving a target/label column."""
    _assert_target_blind(ledger, "ledger")
    _assert_target_blind(source_features, "source_features")
    required = {"sample_id", "last_observed_TWS", "h", "lat", "lon"}
    missing = required.difference(ledger.columns)
    if missing:
        raise ValueError(f"Missing ledger columns: {sorted(missing)}")
    missing_source = set(SOURCE_CORE_COLUMNS).difference(source_features.columns)
    if missing_source:
        raise ValueError(f"Missing source feature columns: {sorted(missing_source)}")

    left = ledger.loc[:, ["sample_id", "last_observed_TWS", "h", "lat", "lon"]].copy()
    left["_order"] = np.arange(len(left), dtype=np.int64)
    x = left.merge(
        source_features.loc[:, SOURCE_CORE_COLUMNS],
        how="left",
        on="sample_id",
        validate="many_to_one",
        sort=False,
    ).sort_values("_order")
    if x[["month_sin", "month_cos"]].isna().any().any():
        raise AssertionError("Missing calendar features in feature matrix")

    out = x.loc[:, CORE_FEATURE_COLUMNS].astype(
        {
            "last_observed_TWS": "float32",
            "h": "int8",
            "lat": "float32",
            "lon": "float32",
            "month_sin": "float32",
            "month_cos": "float32",
        }
    )
    if not np.isfinite(out.to_numpy(dtype=np.float32)).all():
        raise AssertionError("Core feature matrix contains non-finite values")
    return out.reset_index(drop=True)


def horizon_rebalance_weights(horizons: pd.Series | np.ndarray) -> np.ndarray:
    """Reweight observed sampled-h rows back to the exact test h mixture."""
    h = np.asarray(horizons, dtype=np.int8)
    if len(h) == 0:
        raise ValueError("Cannot weight an empty horizon vector")
    counts = pd.Series(h).value_counts().to_dict()
    n = float(len(h))
    weights = np.empty(len(h), dtype=np.float32)
    for horizon in range(1, 8):
        mask = h == horizon
        if not mask.any():
            raise ValueError(f"No rows available for h={horizon}")
        observed_share = float(counts[horizon]) / n
        weights[mask] = TEST_H_WEIGHTS[horizon] / observed_share
    weights /= float(weights.mean())
    return weights


def validation_horizon_weights(horizons: pd.Series | np.ndarray) -> np.ndarray:
    """Per-row weights whose aggregate L2 equals the exact horizon-weighted MSE."""
    h = np.asarray(horizons, dtype=np.int8)
    weights = np.zeros(len(h), dtype=np.float32)
    for horizon in range(1, 8):
        mask = h == horizon
        count = int(mask.sum())
        if count == 0:
            raise ValueError(f"No validation rows for h={horizon}")
        weights[mask] = TEST_H_WEIGHTS[horizon] / count
    weights /= float(weights.mean())
    return weights
