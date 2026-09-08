from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def month_number(values: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Convert monthly timestamps to a monotone integer month index."""
    dt = pd.to_datetime(values)
    if isinstance(dt, pd.Series):
        return dt.dt.year.to_numpy() * 12 + dt.dt.month.to_numpy()
    return dt.year.to_numpy() * 12 + dt.month.to_numpy()


def build_test_availability_ledger(test: pd.DataFrame) -> pd.DataFrame:
    """
    Build the row-level causal TWS state ledger used at test time.

    For every test row, the ledger records:
      - whether current-month TWS is legally visible,
      - the latest legally visible TWS month for that location,
      - the latest legally visible TWS value,
      - the next-calendar-month target date,
      - effective TWS horizon h = months(target_date) - months(last_observed_date).

    The operation is causal because each row only forward-fills TWS observations
    within a location after sorting chronologically. No future value is used.
    """
    required = {"ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"}
    missing = required.difference(test.columns)
    if missing:
        raise ValueError(f"Missing required test columns: {sorted(missing)}")

    x = test.loc[:, ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]].copy()
    x["source_date"] = pd.to_datetime(x["time"])
    x["target_date"] = x["source_date"] + pd.offsets.MonthBegin(1)
    x["tws_visible"] = (~x["TWS_t_masked"]) & x["TWS_t"].notna()

    # Stable location key. Coordinates are the competition's persistent location identity.
    x["location_id"] = x.groupby(["lat", "lon"], sort=True).ngroup()
    x["_row_order"] = np.arange(len(x), dtype=np.int64)

    x = x.sort_values(["location_id", "source_date", "_row_order"], kind="mergesort")

    x["observed_date_candidate"] = x["source_date"].where(x["tws_visible"])
    x["observed_tws_candidate"] = x["TWS_t"].where(x["tws_visible"])

    grouped = x.groupby("location_id", sort=False)
    x["last_observed_date"] = grouped["observed_date_candidate"].ffill()
    x["last_observed_TWS"] = grouped["observed_tws_candidate"].ffill()

    if x["last_observed_date"].isna().any():
        bad = int(x["last_observed_date"].isna().sum())
        raise AssertionError(
            f"{bad} test rows have no legal TWS anchor. "
            "A train-history anchor would be required for these rows."
        )

    target_month = month_number(x["target_date"])
    anchor_month = month_number(x["last_observed_date"])
    x["h"] = (target_month - anchor_month).astype(np.int8)

    if (x["h"] < 1).any():
        raise AssertionError("Effective horizon must always be >= 1.")

    # A visible current-month TWS row must have h=1 for next-month prediction.
    visible_bad = x.loc[x["tws_visible"] & (x["h"] != 1)]
    if not visible_bad.empty:
        raise AssertionError(
            f"Found {len(visible_bad)} visible-TWS rows whose effective horizon is not 1."
        )

    # A masked row must never accidentally use its own hidden TWS value.
    hidden_self_use = x.loc[
        x["TWS_t_masked"] & (x["last_observed_date"] == x["source_date"])
    ]
    if not hidden_self_use.empty:
        raise AssertionError(
            f"Found {len(hidden_self_use)} masked rows using same-month hidden TWS."
        )

    cols = [
        "ID",
        "source_date",
        "target_date",
        "lat",
        "lon",
        "location_id",
        "TWS_t_masked",
        "tws_visible",
        "last_observed_date",
        "last_observed_TWS",
        "h",
        "_row_order",
    ]
    ledger = x.loc[:, cols].sort_values("_row_order").drop(columns="_row_order")
    ledger = ledger.reset_index(drop=True)
    return ledger


def horizon_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return row counts and shares for each effective horizon."""
    counts = ledger["h"].value_counts().sort_index()
    out = counts.rename("rows").to_frame()
    out["share"] = out["rows"] / len(ledger)
    return out


@dataclass(frozen=True)
class ReplayObservationView:
    """Target-blind source panels legal for one historical replay.

    ``source`` retains every covariate row before a replay starts, plus exactly
    the source rows represented by its transported Test schedule.  ``structural``
    has the same row schedule but blanks current-window TWS wherever the replay
    ledger says it is hidden.  This makes the supplied covariate availability
    explicit instead of accidentally reading intervening dense Train rows.
    """

    source: pd.DataFrame
    structural: pd.DataFrame
    first_source_month: str
    last_source_month: str
    prefix_rows: int
    scheduled_rows: int
    withheld_window_rows: int


def build_replay_observation_view(
    source: pd.DataFrame,
    structural: pd.DataFrame,
    *,
    ledger: pd.DataFrame,
    first_source_month: str | pd.Period,
    last_source_month: str | pd.Period,
) -> ReplayObservationView:
    """Return the legal covariate/TWS panels for a transported replay.

    Historical replay folds deliberately keep a dense *prefix* because full
    Train history exists before the replay.  Inside the replay window, however,
    only rows whose IDs occur in the fold ledger correspond to supplied Test
    rows.  All other dense historical covariate rows in that window are withheld.
    The function has no target input and is shared by every candidate so paired
    comparisons use the same information set.
    """
    forbidden_source = {c for c in source.columns if c.lower() in {"target", "y", "label"}}
    forbidden_structural = {c for c in structural.columns if c.lower() in {"target", "y", "label"}}
    if forbidden_source or forbidden_structural:
        raise AssertionError("replay availability panels must be target-blind")
    required_source = {"sample_id", "time", "lat", "lon"}
    required_structural = {"sample_id", "time", "lat", "lon", "TWS_t"}
    required_ledger = {"sample_id", "source_date"}
    if missing := required_source.difference(source.columns):
        raise ValueError(f"source missing columns: {sorted(missing)}")
    if missing := required_structural.difference(structural.columns):
        raise ValueError(f"structural missing columns: {sorted(missing)}")
    if missing := required_ledger.difference(ledger.columns):
        raise ValueError(f"ledger missing columns: {sorted(missing)}")
    if source["sample_id"].duplicated().any() or structural["sample_id"].duplicated().any():
        raise AssertionError("source and structural sample IDs must each be unique")
    if ledger["sample_id"].duplicated().any():
        raise AssertionError("replay ledger sample IDs must be unique")

    first = pd.Period(first_source_month, freq="M")
    last = pd.Period(last_source_month, freq="M")
    if last < first:
        raise ValueError("last_source_month precedes first_source_month")
    source_period = pd.to_datetime(source["time"]).dt.to_period("M")
    structural_period = pd.to_datetime(structural["time"]).dt.to_period("M")
    scheduled_ids = set(ledger["sample_id"].astype(str))
    source_ids = source["sample_id"].astype(str)
    structural_ids = structural["sample_id"].astype(str)
    source_prefix = source_period < first
    structural_prefix = structural_period < first
    source_scheduled = source_ids.isin(scheduled_ids)
    structural_scheduled = structural_ids.isin(scheduled_ids)

    # A scheduled source row must be in the declared window, and every ledger ID
    # must resolve on both sides.  These checks make a shifted/mixed ID namespace
    # fail closed instead of silently creating a partially dense view.
    if (source_period[source_scheduled] < first).any() or (source_period[source_scheduled] > last).any():
        raise AssertionError("scheduled source row lies outside replay window")
    if (structural_period[structural_scheduled] < first).any() or (structural_period[structural_scheduled] > last).any():
        raise AssertionError("scheduled structural row lies outside replay window")
    source_found = set(source_ids[source_scheduled])
    structural_found = set(structural_ids[structural_scheduled])
    if source_found != scheduled_ids or structural_found != scheduled_ids:
        raise AssertionError("one or more replay ledger IDs are absent from availability panels")

    source_view = source.loc[source_prefix | source_scheduled].copy()
    structural_view = structural.loc[structural_prefix | structural_scheduled].copy()
    structural_view["_is_prefix"] = structural_prefix.loc[structural_view.index].to_numpy(dtype=bool)
    visibility = ledger.loc[:, ["sample_id"]].copy()
    visibility["sample_id"] = visibility["sample_id"].astype(str)
    if "tws_visible" in ledger.columns:
        visibility["_visible"] = ledger["tws_visible"].astype(bool).to_numpy()
    elif "sim_tws_visible" in ledger.columns:
        visibility["_visible"] = ledger["sim_tws_visible"].astype(bool).to_numpy()
    else:
        raise ValueError("ledger must expose tws_visible or sim_tws_visible")
    structural_view["_sample_key"] = structural_view["sample_id"].astype(str)
    structural_view = structural_view.merge(
        visibility.rename(columns={"sample_id": "_sample_key"}),
        how="left", on="_sample_key", validate="one_to_one", sort=False,
    )
    in_window = ~structural_view["_is_prefix"].to_numpy(dtype=bool)
    hidden = in_window & ~structural_view["_visible"].eq(True).to_numpy(dtype=bool)
    structural_view.loc[hidden, "TWS_t"] = np.nan
    structural_view = structural_view.drop(columns=["_sample_key", "_visible", "_is_prefix"])

    expected_schedule = ledger["sample_id"].astype(str).tolist()
    observed_schedule = source_view.loc[
        source_view["sample_id"].astype(str).isin(scheduled_ids), "sample_id"
    ].astype(str).tolist()
    if set(observed_schedule) != set(expected_schedule):
        raise AssertionError("source availability schedule differs from ledger")
    withheld = int(((source_period >= first) & (source_period <= last) & ~source_scheduled).sum())
    return ReplayObservationView(
        source=source_view.reset_index(drop=True),
        structural=structural_view.reset_index(drop=True),
        first_source_month=str(first),
        last_source_month=str(last),
        prefix_rows=int(source_prefix.sum()),
        scheduled_rows=int(source_scheduled.sum()),
        withheld_window_rows=withheld,
    )
