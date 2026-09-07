from __future__ import annotations

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
