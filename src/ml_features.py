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

HYDRO_FEATURE_COLUMNS = [
    "last_observed_TWS",
    "h",
    "lat",
    "lon",
    "month_sin",
    "month_cos",
    "SPEI_01_t",
    "SPEI_03_t",
    "SPEI_06_t",
    "SPEI_12_t",
    "SOIL_MOISTURE_t",
]

SOURCE_HYDRO_COLUMNS = [
    "sample_id",
    "month_sin",
    "month_cos",
    "SPEI_01_t",
    "SPEI_03_t",
    "SPEI_06_t",
    "SPEI_12_t",
    "SOIL_MOISTURE_t",
]

HYDRO_GAP_VARIABLES = [
    "SPEI_01_t",
    "SPEI_03_t",
    "SPEI_06_t",
    "SPEI_12_t",
    "SOIL_MOISTURE_t",
]

SOURCE_HYDRO_HISTORY_COLUMNS = [
    "sample_id",
    "time",
    "lat",
    "lon",
    "month_sin",
    "month_cos",
    *HYDRO_GAP_VARIABLES,
]

TWS_HISTORY_LAGS = (1, 2, 3, 6, 12)

TWS_HISTORY_FEATURE_COLUMNS = HYDRO_FEATURE_COLUMNS + [
    "TWS_anchor_lag1",
    "TWS_anchor_lag2",
    "TWS_anchor_lag3",
    "TWS_anchor_lag6",
    "TWS_anchor_lag12",
    "TWS_anchor_delta1",
    "TWS_anchor_delta3",
    "TWS_anchor_delta6",
    "TWS_anchor_delta12",
]

HYDRO_GAP_FEATURE_COLUMNS = TWS_HISTORY_FEATURE_COLUMNS + [
    "SPEI_01_gap_delta",
    "SPEI_03_gap_delta",
    "SPEI_06_gap_delta",
    "SPEI_12_gap_delta",
    "SOIL_MOISTURE_gap_delta",
]

HYDRO_GAP_SAFE_FEATURE_COLUMNS = HYDRO_FEATURE_COLUMNS + [
    "SPEI_01_gap_delta",
    "SPEI_03_gap_delta",
    "SPEI_06_gap_delta",
    "SPEI_12_gap_delta",
    "SOIL_MOISTURE_gap_delta",
]

# History reconstructed by ``simulate_observations`` rather than by looking up
# exact calendar lags in a dense truth panel.  These values describe only the
# two visible observations immediately behind the legal anchor and their real
# calendar spacing; missing observations remain missing.
VISIBLE_HISTORY_FEATURE_COLUMNS = HYDRO_GAP_SAFE_FEATURE_COLUMNS + [
    "previous_visible_TWS",
    "previous_visible_age_months",
    "older_visible_TWS",
    "older_visible_age_months",
    "previous_visible_spacing_months",
    "has_previous_visible",
    "has_older_visible",
]

LOCATION_CAT_FEATURE_COLUMNS = HYDRO_GAP_FEATURE_COLUMNS + ["location_id"]

EOF_RANK = 8
EOF_FEATURE_COLUMNS = HYDRO_GAP_FEATURE_COLUMNS + [
    f"eof_loading_{i}" for i in range(1, EOF_RANK + 1)
]


@dataclass(frozen=True)
class SampledTrainingRows:
    rows: pd.DataFrame
    horizon_counts: dict[int, int]
    dropped_missing_anchor: int


@dataclass(frozen=True)
class EOFLocationFeatures:
    loadings: pd.DataFrame
    explained_variance_ratio: tuple[float, ...]
    n_months: int


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


def build_hydro_feature_matrix(
    ledger: pd.DataFrame,
    source_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build EXP002 features using fresh source-month hydrometeorology.

    TWS state remains restricted to the exact legal historical anchor in the
    ledger. SPEI, soil moisture, and calendar variables are joined from the
    current source row by sample_id, so they remain fresh even when TWS is stale.
    The function is deliberately target-blind.
    """
    _assert_target_blind(ledger, "ledger")
    _assert_target_blind(source_features, "source_features")
    required = {"sample_id", "last_observed_TWS", "h", "lat", "lon"}
    missing = required.difference(ledger.columns)
    if missing:
        raise ValueError(f"Missing ledger columns: {sorted(missing)}")
    missing_source = set(SOURCE_HYDRO_COLUMNS).difference(source_features.columns)
    if missing_source:
        raise ValueError(f"Missing hydro source feature columns: {sorted(missing_source)}")
    if source_features["sample_id"].duplicated().any():
        raise AssertionError("source_features sample_id must be unique")

    left = ledger.loc[:, ["sample_id", "last_observed_TWS", "h", "lat", "lon"]].copy()
    left["_order"] = np.arange(len(left), dtype=np.int64)
    x = left.merge(
        source_features.loc[:, SOURCE_HYDRO_COLUMNS],
        how="left",
        on="sample_id",
        validate="many_to_one",
        sort=False,
    ).sort_values("_order")

    out = x.loc[:, HYDRO_FEATURE_COLUMNS].astype(
        {
            "last_observed_TWS": "float32",
            "h": "int8",
            "lat": "float32",
            "lon": "float32",
            "month_sin": "float32",
            "month_cos": "float32",
            "SPEI_01_t": "float32",
            "SPEI_03_t": "float32",
            "SPEI_06_t": "float32",
            "SPEI_12_t": "float32",
            "SOIL_MOISTURE_t": "float32",
        }
    )
    if not np.isfinite(out.to_numpy(dtype=np.float32)).all():
        raise AssertionError("Hydro feature matrix contains non-finite values")
    return out.reset_index(drop=True)


def build_tws_history_feature_matrix(
    ledger: pd.DataFrame,
    source_features: pd.DataFrame,
    structural: pd.DataFrame,
) -> pd.DataFrame:
    """Build EXP003 = EXP002 plus exact-calendar TWS history behind the legal anchor.

    Every TWS history value is joined relative to `last_observed_date`, never the
    current source month. For h>1 this means the whole history stack remains behind
    the simulated hidden interval. Missing GRACE calendar months remain NaN and are
    handled natively by LightGBM; they are never backfilled or interpolated.
    """
    _assert_target_blind(ledger, "ledger")
    _assert_target_blind(source_features, "source_features")
    _assert_target_blind(structural, "structural")

    required_ledger = {
        "sample_id", "last_observed_TWS", "last_observed_date", "h", "lat", "lon"
    }
    missing = required_ledger.difference(ledger.columns)
    if missing:
        raise ValueError(f"Missing ledger columns: {sorted(missing)}")
    required_structural = {"time", "lat", "lon", "TWS_t"}
    missing = required_structural.difference(structural.columns)
    if missing:
        raise ValueError(f"Missing structural columns: {sorted(missing)}")

    base = ledger.loc[:, [
        "sample_id", "last_observed_TWS", "last_observed_date", "h", "lat", "lon"
    ]].copy()
    base["_order"] = np.arange(len(base), dtype=np.int64)

    x = base.merge(
        source_features.loc[:, SOURCE_HYDRO_COLUMNS],
        how="left",
        on="sample_id",
        validate="many_to_one",
        sort=False,
    )

    state = structural.loc[:, ["time", "lat", "lon", "TWS_t"]].copy()
    state["state_period"] = pd.to_datetime(state["time"]).dt.to_period("M")
    state = state.loc[:, ["lat", "lon", "state_period", "TWS_t"]]
    if state.duplicated(["lat", "lon", "state_period"]).any():
        raise AssertionError("Structural TWS lookup is not unique by location/calendar month")

    anchor_period = pd.to_datetime(x["last_observed_date"]).dt.to_period("M")
    for lag in TWS_HISTORY_LAGS:
        lookup_period_col = f"_lag{lag}_period"
        value_col = f"TWS_anchor_lag{lag}"
        x[lookup_period_col] = anchor_period - lag
        lookup = state.rename(
            columns={"state_period": lookup_period_col, "TWS_t": value_col}
        )
        x = x.merge(
            lookup,
            how="left",
            on=["lat", "lon", lookup_period_col],
            validate="many_to_one",
            sort=False,
        )
        x = x.drop(columns=[lookup_period_col])

    for lag in (1, 3, 6, 12):
        x[f"TWS_anchor_delta{lag}"] = (
            x["last_observed_TWS"] - x[f"TWS_anchor_lag{lag}"]
        )

    x = x.sort_values("_order")
    out = x.loc[:, TWS_HISTORY_FEATURE_COLUMNS].copy()
    out = out.astype({
        c: ("int8" if c == "h" else "float32")
        for c in TWS_HISTORY_FEATURE_COLUMNS
    })

    values = out.to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise AssertionError("TWS-history feature matrix contains infinite values")
    return out.reset_index(drop=True)


def build_hydro_gap_feature_matrix(
    ledger: pd.DataFrame,
    source_features: pd.DataFrame,
    structural: pd.DataFrame,
) -> pd.DataFrame:
    """Build EXP005 = EXP003 plus source-minus-anchor hydrology deltas.

    The current hydrometeorological values are already legal EXP003 features. This
    ablation joins the *same variables* at the legal TWS anchor month and adds only
    their change from anchor -> current source month. No future month is consulted,
    and h=1 rows reduce to an exact zero gap by construction.

    This is intentionally the smallest hydrologic-gap test before trying richer
    path summaries. Exact calendar joins are used throughout; missing months are
    never interpolated or row-shifted.
    """
    _assert_target_blind(ledger, "ledger")
    _assert_target_blind(source_features, "source_features")
    _assert_target_blind(structural, "structural")

    required_ledger = {
        "sample_id",
        "last_observed_date",
        "last_observed_TWS",
        "h",
        "lat",
        "lon",
    }
    missing = required_ledger.difference(ledger.columns)
    if missing:
        raise ValueError(f"Missing ledger columns: {sorted(missing)}")

    missing_source = set(SOURCE_HYDRO_HISTORY_COLUMNS).difference(source_features.columns)
    if missing_source:
        raise ValueError(
            f"Missing hydrology-history source columns: {sorted(missing_source)}"
        )
    if source_features["sample_id"].duplicated().any():
        raise AssertionError("source_features sample_id must be unique")

    # Start from the fully frozen EXP003 matrix so this experiment changes only
    # the five hydrologic gap-delta columns below.
    out = build_tws_history_feature_matrix(ledger, source_features, structural)

    lookup = source_features.loc[:, ["time", "lat", "lon", *HYDRO_GAP_VARIABLES]].copy()
    lookup["anchor_period"] = pd.to_datetime(lookup["time"]).dt.to_period("M")
    lookup = lookup.drop(columns=["time"])
    if lookup.duplicated(["lat", "lon", "anchor_period"]).any():
        raise AssertionError(
            "Hydrology lookup is not unique by location/calendar month"
        )
    lookup = lookup.rename(
        columns={v: f"_anchor_{v}" for v in HYDRO_GAP_VARIABLES}
    )

    left = ledger.loc[:, ["last_observed_date", "lat", "lon", "h"]].copy()
    left["_order"] = np.arange(len(left), dtype=np.int64)
    left["anchor_period"] = pd.to_datetime(left["last_observed_date"]).dt.to_period("M")
    anchors = left.merge(
        lookup,
        how="left",
        on=["lat", "lon", "anchor_period"],
        validate="many_to_one",
        sort=False,
    ).sort_values("_order")

    anchor_columns = [f"_anchor_{v}" for v in HYDRO_GAP_VARIABLES]
    if anchors[anchor_columns].isna().any().any():
        missing_rows = int(anchors[anchor_columns].isna().any(axis=1).sum())
        raise AssertionError(
            f"Missing legal anchor hydrology for {missing_rows} examples"
        )

    gap_names = {
        "SPEI_01_t": "SPEI_01_gap_delta",
        "SPEI_03_t": "SPEI_03_gap_delta",
        "SPEI_06_t": "SPEI_06_gap_delta",
        "SPEI_12_t": "SPEI_12_gap_delta",
        "SOIL_MOISTURE_t": "SOIL_MOISTURE_gap_delta",
    }
    for variable, gap_name in gap_names.items():
        out[gap_name] = (
            out[variable].to_numpy(dtype=np.float32)
            - anchors[f"_anchor_{variable}"].to_numpy(dtype=np.float32)
        )

    h1 = ledger["h"].to_numpy(dtype=np.int8) == 1
    if h1.any():
        h1_gaps = out.loc[h1, list(gap_names.values())].to_numpy(dtype=np.float32)
        if not np.allclose(h1_gaps, 0.0, atol=1e-6, rtol=0.0):
            raise AssertionError("h=1 hydrology gap deltas must be exactly zero")

    out = out.loc[:, HYDRO_GAP_FEATURE_COLUMNS].astype(
        {c: ("int8" if c == "h" else "float32") for c in HYDRO_GAP_FEATURE_COLUMNS}
    )
    values = out.to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise AssertionError("Hydrology-gap feature matrix contains infinite values")
    return out.reset_index(drop=True)


def build_hydro_gap_safe_feature_matrix(
    ledger: pd.DataFrame,
    source_features: pd.DataFrame,
    structural: pd.DataFrame,
) -> pd.DataFrame:
    """Build EXP009: EXP002 plus source-minus-anchor hydrology deltas only.

    This deliberately excludes all exact-calendar TWS lag/delta features because
    their availability collapses on the sparse real Test calendar. The retained
    features are either current source-month covariates or the legal TWS anchor and
    hydrology at that same legal anchor month, all of which are reproducible at test.
    """
    _assert_target_blind(ledger, "ledger")
    _assert_target_blind(source_features, "source_features")
    _assert_target_blind(structural, "structural")

    required_ledger = {
        "sample_id",
        "last_observed_date",
        "last_observed_TWS",
        "h",
        "lat",
        "lon",
    }
    missing = required_ledger.difference(ledger.columns)
    if missing:
        raise ValueError(f"Missing ledger columns: {sorted(missing)}")

    missing_source = set(SOURCE_HYDRO_HISTORY_COLUMNS).difference(source_features.columns)
    if missing_source:
        raise ValueError(
            f"Missing hydrology-history source columns: {sorted(missing_source)}"
        )
    if source_features["sample_id"].duplicated().any():
        raise AssertionError("source_features sample_id must be unique")

    # Start from the always-reproducible EXP002 matrix, not EXP003 history.
    out = build_hydro_feature_matrix(ledger, source_features)

    lookup = source_features.loc[:, ["time", "lat", "lon", *HYDRO_GAP_VARIABLES]].copy()
    lookup["anchor_period"] = pd.to_datetime(lookup["time"]).dt.to_period("M")
    lookup = lookup.drop(columns=["time"])
    if lookup.duplicated(["lat", "lon", "anchor_period"]).any():
        raise AssertionError("Hydrology lookup is not unique by location/calendar month")
    lookup = lookup.rename(columns={v: f"_anchor_{v}" for v in HYDRO_GAP_VARIABLES})

    left = ledger.loc[:, ["last_observed_date", "lat", "lon", "h"]].copy()
    left["_order"] = np.arange(len(left), dtype=np.int64)
    left["anchor_period"] = pd.to_datetime(left["last_observed_date"]).dt.to_period("M")
    anchors = left.merge(
        lookup,
        how="left",
        on=["lat", "lon", "anchor_period"],
        validate="many_to_one",
        sort=False,
    ).sort_values("_order")

    anchor_columns = [f"_anchor_{v}" for v in HYDRO_GAP_VARIABLES]
    if anchors[anchor_columns].isna().any().any():
        missing_rows = int(anchors[anchor_columns].isna().any(axis=1).sum())
        raise AssertionError(
            f"Missing legal anchor hydrology for {missing_rows} examples"
        )

    gap_names = {
        "SPEI_01_t": "SPEI_01_gap_delta",
        "SPEI_03_t": "SPEI_03_gap_delta",
        "SPEI_06_t": "SPEI_06_gap_delta",
        "SPEI_12_t": "SPEI_12_gap_delta",
        "SOIL_MOISTURE_t": "SOIL_MOISTURE_gap_delta",
    }
    for variable, gap_name in gap_names.items():
        out[gap_name] = (
            out[variable].to_numpy(dtype=np.float32)
            - anchors[f"_anchor_{variable}"].to_numpy(dtype=np.float32)
        )

    h1 = ledger["h"].to_numpy(dtype=np.int8) == 1
    if h1.any():
        h1_gaps = out.loc[h1, list(gap_names.values())].to_numpy(dtype=np.float32)
        if not np.allclose(h1_gaps, 0.0, atol=1e-6, rtol=0.0):
            raise AssertionError("h=1 hydrology gap deltas must be exactly zero")

    out = out.loc[:, HYDRO_GAP_SAFE_FEATURE_COLUMNS].astype(
        {c: ("int8" if c == "h" else "float32") for c in HYDRO_GAP_SAFE_FEATURE_COLUMNS}
    )
    values = out.to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise AssertionError("Hydrology-gap-safe feature matrix contains infinite values")
    return out.reset_index(drop=True)


def build_visible_history_feature_matrix(
    ledger: pd.DataFrame,
    source_features: pd.DataFrame,
    structural: pd.DataFrame,
) -> pd.DataFrame:
    """Build R03 features from the simulator's legal visible-history stream.

    ``structural`` is retained for the common feature-builder API but is not
    consulted: every state value used here was emitted by the observation
    simulator from its supplied visibility schedule.  This deliberately avoids
    falling back to dense exact-month TWS lags when the true Test schedule is
    sparse.
    """
    _assert_target_blind(ledger, "ledger")
    required = {
        "last_observed_date", "previous_visible_date", "previous_visible_TWS",
        "older_visible_date", "older_visible_TWS",
    }
    missing = required.difference(ledger.columns)
    if missing:
        raise ValueError(f"Ledger lacks simulator visible-history columns: {sorted(missing)}")
    out = build_hydro_gap_safe_feature_matrix(ledger, source_features, structural)
    current = pd.to_datetime(ledger["last_observed_date"]).dt.to_period("M")
    previous = pd.to_datetime(ledger["previous_visible_date"], errors="coerce").dt.to_period("M")
    older = pd.to_datetime(ledger["older_visible_date"], errors="coerce").dt.to_period("M")
    out["previous_visible_TWS"] = ledger["previous_visible_TWS"].to_numpy(dtype=np.float32)
    out["previous_visible_age_months"] = np.asarray(current - previous, dtype="float32")
    out["older_visible_TWS"] = ledger["older_visible_TWS"].to_numpy(dtype=np.float32)
    out["older_visible_age_months"] = np.asarray(current - older, dtype="float32")
    out["previous_visible_spacing_months"] = np.asarray(previous - older, dtype="float32")
    out["has_previous_visible"] = previous.notna().to_numpy(dtype=np.float32)
    out["has_older_visible"] = older.notna().to_numpy(dtype=np.float32)
    values = out.to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise AssertionError("Visible-history feature matrix contains infinite values")
    return out.loc[:, VISIBLE_HISTORY_FEATURE_COLUMNS].astype(
        {c: ("int8" if c == "h" else "float32") for c in VISIBLE_HISTORY_FEATURE_COLUMNS}
    ).reset_index(drop=True)


def build_location_cat_feature_matrix(
    ledger: pd.DataFrame,
    source_features: pd.DataFrame,
    structural: pd.DataFrame,
) -> pd.DataFrame:
    """Build EXP006 = EXP005 plus a target-blind categorical location id.

    The identifier is a deterministic ordinal of the unique `(lat, lon)` pairs in
    the structural panel, sorted by coordinates. It uses no target information and
    is stable between training and validation because both use the same global grid.
    Latitude/longitude remain in the matrix so this is a strict add-one-feature
    ablation rather than a replacement of the existing spatial representation.
    """
    _assert_target_blind(ledger, "ledger")
    _assert_target_blind(source_features, "source_features")
    _assert_target_blind(structural, "structural")

    out = build_hydro_gap_feature_matrix(ledger, source_features, structural)

    required = {"lat", "lon"}
    missing = required.difference(structural.columns)
    if missing:
        raise ValueError(f"Missing structural location columns: {sorted(missing)}")

    locations = (
        structural.loc[:, ["lat", "lon"]]
        .drop_duplicates()
        .sort_values(["lat", "lon"], kind="mergesort")
        .reset_index(drop=True)
    )
    locations["location_id"] = np.arange(len(locations), dtype=np.int32)

    left = ledger.loc[:, ["lat", "lon"]].copy()
    left["_order"] = np.arange(len(left), dtype=np.int64)
    ids = (
        left.merge(
            locations,
            how="left",
            on=["lat", "lon"],
            validate="many_to_one",
            sort=False,
        )
        .sort_values("_order")
        ["location_id"]
    )
    if ids.isna().any():
        raise AssertionError("Missing location_id for one or more examples")

    out["location_id"] = ids.to_numpy(dtype=np.int32)
    out = out.loc[:, LOCATION_CAT_FEATURE_COLUMNS]
    return out.reset_index(drop=True)


def fit_eof_location_features(
    structural: pd.DataFrame,
    *,
    max_source_month: str | pd.Period,
    rank: int = EOF_RANK,
) -> EOFLocationFeatures:
    """Fit fold-causal EOF loadings from historical TWS fields only.

    The fit uses source-month TWS rows at or before ``max_source_month`` and never
    receives a target column. Missing location/month values are replaced by that
    location's historical mean *inside the fit prefix*; after centering, this is
    equivalent to a zero anomaly and does not interpolate temporal state.

    The returned right-singular vectors are static location embeddings. Their signs
    are canonicalized for deterministic reruns. They are intended as low-rank spatial
    descriptors, not as reconstructed future TWS values.
    """
    _assert_target_blind(structural, "structural")
    required = {"time", "lat", "lon", "TWS_t"}
    missing = required.difference(structural.columns)
    if missing:
        raise ValueError(f"Missing structural columns for EOF fit: {sorted(missing)}")
    if rank < 1:
        raise ValueError("EOF rank must be >= 1")

    cutoff = pd.Period(max_source_month, freq="M")
    state = structural.loc[:, ["time", "lat", "lon", "TWS_t"]].copy()
    state["source_period"] = pd.to_datetime(state["time"]).dt.to_period("M")
    state = state.loc[state["source_period"] <= cutoff].copy()
    if state.empty:
        raise AssertionError(f"No TWS rows available for EOF fit through {cutoff}")
    if state.duplicated(["source_period", "lat", "lon"]).any():
        raise AssertionError("EOF fit state is not unique by month/location")

    locations = (
        structural.loc[:, ["lat", "lon"]]
        .drop_duplicates()
        .sort_values(["lat", "lon"], kind="mergesort")
        .reset_index(drop=True)
    )
    location_index = pd.MultiIndex.from_frame(locations[["lat", "lon"]])

    field = state.pivot(
        index="source_period",
        columns=["lat", "lon"],
        values="TWS_t",
    ).sort_index()
    field = field.reindex(columns=location_index)

    matrix = field.to_numpy(dtype=np.float32)
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    sums = np.where(finite, matrix, 0.0).sum(axis=0, dtype=np.float64)
    means = np.divide(
        sums,
        counts,
        out=np.zeros(matrix.shape[1], dtype=np.float64),
        where=counts > 0,
    ).astype(np.float32)
    filled = np.where(finite, matrix, means[None, :]).astype(np.float32, copy=False)
    centered = filled - means[None, :]

    max_rank = min(centered.shape)
    if rank > max_rank:
        raise ValueError(f"EOF rank {rank} exceeds available matrix rank bound {max_rank}")

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    loadings = vt[:rank].T.astype(np.float32, copy=True)

    # SVD signs are mathematically arbitrary. Canonicalize each component so the
    # largest-magnitude loading is positive, which improves reproducibility.
    for j in range(rank):
        pivot_idx = int(np.argmax(np.abs(loadings[:, j])))
        if loadings[pivot_idx, j] < 0:
            loadings[:, j] *= -1.0

    energy = np.square(singular_values.astype(np.float64))
    total_energy = float(energy.sum())
    if total_energy > 0:
        explained = tuple(float(v / total_energy) for v in energy[:rank])
    else:
        explained = tuple(0.0 for _ in range(rank))

    loading_frame = locations.copy()
    for j in range(rank):
        loading_frame[f"eof_loading_{j + 1}"] = loadings[:, j]

    if loading_frame[[f"eof_loading_{i}" for i in range(1, rank + 1)]].isna().any().any():
        raise AssertionError("EOF location loadings contain missing values")

    return EOFLocationFeatures(
        loadings=loading_frame,
        explained_variance_ratio=explained,
        n_months=int(centered.shape[0]),
    )


def build_eof_feature_matrix(
    ledger: pd.DataFrame,
    source_features: pd.DataFrame,
    structural: pd.DataFrame,
    eof_locations: pd.DataFrame,
) -> pd.DataFrame:
    """Build EXP007 = EXP005 plus fold-causal static EOF location loadings."""
    _assert_target_blind(ledger, "ledger")
    _assert_target_blind(source_features, "source_features")
    _assert_target_blind(structural, "structural")
    _assert_target_blind(eof_locations, "eof_locations")

    out = build_hydro_gap_feature_matrix(ledger, source_features, structural)
    loading_cols = [f"eof_loading_{i}" for i in range(1, EOF_RANK + 1)]
    required = {"lat", "lon", *loading_cols}
    missing = required.difference(eof_locations.columns)
    if missing:
        raise ValueError(f"Missing EOF location columns: {sorted(missing)}")
    if eof_locations.duplicated(["lat", "lon"]).any():
        raise AssertionError("EOF location table is not unique by location")

    left = ledger.loc[:, ["lat", "lon"]].copy()
    left["_order"] = np.arange(len(left), dtype=np.int64)
    joined = left.merge(
        eof_locations.loc[:, ["lat", "lon", *loading_cols]],
        how="left",
        on=["lat", "lon"],
        validate="many_to_one",
        sort=False,
    ).sort_values("_order")
    if joined[loading_cols].isna().any().any():
        raise AssertionError("Missing EOF loading for one or more examples")

    for col in loading_cols:
        out[col] = joined[col].to_numpy(dtype=np.float32)
    out = out.loc[:, EOF_FEATURE_COLUMNS].astype(
        {c: ("int8" if c == "h" else "float32") for c in EOF_FEATURE_COLUMNS}
    )
    values = out.to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise AssertionError("EOF feature matrix contains infinite values")
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
