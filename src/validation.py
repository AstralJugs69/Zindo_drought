from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.availability import build_test_availability_ledger


@dataclass(frozen=True)
class HistoricalMaskFold:
    ledger: pd.DataFrame
    labels: pd.DataFrame
    start_month: pd.Period
    end_month: pd.Period
    dropped_missing_anchor: int


def build_test_mask_template(test: pd.DataFrame) -> pd.DataFrame:
    required = {"ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"}
    missing = required.difference(test.columns)
    if missing:
        raise ValueError(f"Missing required test columns: {sorted(missing)}")

    base = test.loc[:, ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]].copy()
    exact = build_test_availability_ledger(base)
    x = base.loc[:, ["ID", "time", "lat", "lon", "TWS_t_masked"]].merge(
        exact.loc[:, ["ID", "h", "tws_visible"]],
        how="left",
        on="ID",
        validate="one_to_one",
    )
    x["test_source_date"] = pd.to_datetime(x["time"])
    periods = x["test_source_date"].dt.to_period("M")
    first_period = periods.min()
    x["offset_months"] = periods.map(lambda p: p.ordinal - first_period.ordinal).astype(np.int16)
    x["template_visible"] = x["tws_visible"].astype(bool)
    x["template_h"] = x["h"].astype(np.int8)
    return x.loc[:, [
        "ID",
        "lat",
        "lon",
        "test_source_date",
        "offset_months",
        "template_visible",
        "template_h",
    ]]


def find_exact_template_starts(train: pd.DataFrame, template: pd.DataFrame) -> list[pd.Period]:
    if "time" not in train.columns:
        raise ValueError("train must contain time")

    train_periods = pd.PeriodIndex(pd.to_datetime(train["time"]).dt.to_period("M").unique()).sort_values()
    observed = set(train_periods)
    offsets = sorted(template["offset_months"].unique().tolist())
    max_offset = max(offsets)

    starts: list[pd.Period] = []
    for start in train_periods:
        if start + max_offset > train_periods.max():
            continue
        if all((start + int(offset)) in observed for offset in offsets):
            starts.append(start)
    return starts


def build_exact_historical_mask_fold(
    train: pd.DataFrame,
    template: pd.DataFrame,
    start_month: str | pd.Period,
) -> HistoricalMaskFold:
    """Replay the exact test row-presence/mask template on a shifted historical calendar.

    The returned ledger is target-blind: it contains no `target` column. Labels are
    returned separately. Historical TWS inside simulated masked validation rows is
    never used to update state.
    """
    required_train = {"sample_id", "time", "lat", "lon", "TWS_t", "target"}
    missing = required_train.difference(train.columns)
    if missing:
        raise ValueError(f"Missing required train columns: {sorted(missing)}")

    start = pd.Period(start_month, freq="M")
    max_offset = int(template["offset_months"].max())
    end = start + max_offset

    train_state = train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t", "target"]].copy()
    train_state["source_date"] = pd.to_datetime(train_state["time"])
    train_state["source_period"] = train_state["source_date"].dt.to_period("M")

    map_rows = template.copy()
    map_rows["source_period"] = map_rows["offset_months"].map(lambda o: start + int(o))

    selected = train_state.merge(
        map_rows.loc[:, [
            "ID",
            "lat",
            "lon",
            "source_period",
            "offset_months",
            "template_visible",
            "template_h",
        ]],
        how="inner",
        on=["lat", "lon", "source_period"],
        validate="one_to_one",
    )
    selected = selected.sort_values(["source_date", "lat", "lon"], kind="mergesort").reset_index(drop=True)

    if selected.empty:
        raise AssertionError("Historical template replay produced no rows")

    # Directly transplant the row-level legal horizon from the real test geometry.
    # For source t and effective horizon h, the legal TWS anchor is calendar
    # month t + 1 - h.  This avoids historical row-presence quirks inventing
    # artificial horizons that do not exist in the real test set.
    selected["anchor_period"] = [
        p + (1 - int(h))
        for p, h in zip(selected["source_period"], selected["template_h"])
    ]

    anchor_state = train_state.loc[:, ["lat", "lon", "source_period", "TWS_t"]].rename(
        columns={"source_period": "anchor_period", "TWS_t": "anchor_tws"}
    )
    x = selected.merge(
        anchor_state,
        how="left",
        on=["lat", "lon", "anchor_period"],
        validate="many_to_one",
    )

    dropped_missing_anchor = int(x["anchor_tws"].isna().sum())
    x = x.loc[x["anchor_tws"].notna()].copy()
    if x.empty:
        raise AssertionError("All historical validation rows lack their transplanted legal TWS anchor")

    labels = x.loc[:, ["sample_id", "target"]].copy()
    x = x.drop(columns=["target"])

    x["sim_tws_visible"] = x["template_visible"].astype(bool)
    x["last_observed_date"] = x["anchor_period"].dt.to_timestamp(how="start")
    x["last_observed_TWS"] = x["anchor_tws"]
    x["target_date"] = x["source_date"] + pd.offsets.MonthBegin(1)
    x["h"] = x["template_h"].astype(np.int8)

    if int(x["h"].min()) != 1 or int(x["h"].max()) != 7:
        raise AssertionError("Transplanted historical fold must preserve exact test horizon range 1..7")

    visible_h_mismatch = x["sim_tws_visible"] != (x["h"] == 1)
    if visible_h_mismatch.any():
        raise AssertionError(
            f"Found {int(visible_h_mismatch.sum())} rows where visibility disagrees with transplanted h"
        )

    hidden_self_use = (~x["sim_tws_visible"]) & (x["last_observed_date"] == x["source_date"])
    if hidden_self_use.any():
        raise AssertionError(f"Found {int(hidden_self_use.sum())} hidden rows using same-month TWS")

    x["_row_order"] = np.arange(len(x), dtype=np.int64)
    x["location_id"] = x.groupby(["lat", "lon"], sort=True).ngroup()

    ledger_cols = [
        "sample_id",
        "ID",
        "source_date",
        "target_date",
        "lat",
        "lon",
        "location_id",
        "offset_months",
        "template_visible",
        "sim_tws_visible",
        "last_observed_date",
        "last_observed_TWS",
        "h",
        "_row_order",
    ]
    ledger = x.loc[:, ledger_cols].sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)

    if "target" in ledger.columns:
        raise AssertionError("Target leaked into validation ledger")

    labels = labels.set_index("sample_id").loc[ledger["sample_id"]].reset_index()
    return HistoricalMaskFold(
        ledger=ledger,
        labels=labels,
        start_month=start,
        end_month=end,
        dropped_missing_anchor=dropped_missing_anchor,
    )
