"""Calendar-safe structural audit for the Zindi drought challenge.

This script intentionally performs no model training.  Its job is to verify the
dataset assumptions that every later validation/modeling step depends on:

* exact schemas and key uniqueness;
* calendar-month coverage (without relying on row order);
* the identity target(cell, t) == TWS(cell, t+1) when t+1 exists;
* test TWS-mask counts and month-level geometry.

Run from the repository root, for example:

    python scripts/audit_dataset.py --data-dir /kaggle/input/datasets/.../drought
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_REQUIRED = [
    "sample_id",
    "time",
    "lat",
    "lon",
    "TWS_t",
    "SPEI_01_t",
    "SPEI_03_t",
    "SPEI_06_t",
    "SPEI_12_t",
    "SOIL_MOISTURE_t",
    "month_sin",
    "month_cos",
    "target",
]

TEST_REQUIRED = [
    "ID",
    "time",
    "lat",
    "lon",
    "TWS_t",
    "SPEI_01_t",
    "SPEI_03_t",
    "SPEI_06_t",
    "SPEI_12_t",
    "SOIL_MOISTURE_t",
    "month_sin",
    "month_cos",
    "TWS_t_masked",
]


def _month_strings(values: pd.Series) -> list[str]:
    periods = values.dt.to_period("M").drop_duplicates().sort_values()
    return [str(p) for p in periods]


def _missing_months(values: pd.Series) -> list[str]:
    observed = pd.PeriodIndex(values.dt.to_period("M").drop_duplicates())
    full = pd.period_range(observed.min(), observed.max(), freq="M")
    missing = full.difference(observed)
    return [str(p) for p in missing]


def audit(data_dir: Path) -> dict:
    train_path = data_dir / "Train.csv"
    test_path = data_dir / "Test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected Train.csv and Test.csv under {data_dir.resolve()}"
        )

    print("[1/5] Reading schemas...")
    train_header = pd.read_csv(train_path, nrows=0)
    test_header = pd.read_csv(test_path, nrows=0)

    if train_header.columns.tolist() != TRAIN_REQUIRED:
        raise AssertionError(
            "Train schema changed.\n"
            f"Expected: {TRAIN_REQUIRED}\n"
            f"Actual:   {train_header.columns.tolist()}"
        )
    if test_header.columns.tolist() != TEST_REQUIRED:
        raise AssertionError(
            "Test schema changed.\n"
            f"Expected: {TEST_REQUIRED}\n"
            f"Actual:   {test_header.columns.tolist()}"
        )

    # Only read columns required for structural/causality checks.  This keeps the
    # audit much lighter than loading every feature column.
    print("[2/5] Reading structural columns...")
    train = pd.read_csv(
        train_path,
        usecols=["sample_id", "time", "lat", "lon", "TWS_t", "target"],
        parse_dates=["time"],
    )
    test = pd.read_csv(
        test_path,
        usecols=["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"],
        parse_dates=["time"],
    )

    print("[3/5] Checking keys and calendar structure...")
    if train["sample_id"].duplicated().any():
        raise AssertionError("Train sample_id is not unique")
    if test["ID"].duplicated().any():
        raise AssertionError("Test ID is not unique")
    if train.duplicated(["time", "lat", "lon"]).any():
        raise AssertionError("Duplicate (time, lat, lon) rows exist in Train")
    if test.duplicated(["time", "lat", "lon"]).any():
        raise AssertionError("Duplicate (time, lat, lon) rows exist in Test")

    train_months = _month_strings(train["time"])
    test_months = _month_strings(test["time"])
    train_missing_months = _missing_months(train["time"])
    test_missing_months = _missing_months(test["time"])

    train_locations = train[["lat", "lon"]].drop_duplicates()
    test_locations = test[["lat", "lon"]].drop_duplicates()

    print("[4/5] Verifying target == next-calendar-month TWS...")

    # IMPORTANT: this is an exact calendar join.  It deliberately does NOT use
    # groupby.shift(-1), because GRACE has missing calendar months.
    target_rows = train[["time", "lat", "lon", "target"]].copy()
    target_rows["target_time"] = target_rows["time"] + pd.DateOffset(months=1)

    next_state = train[["time", "lat", "lon", "TWS_t"]].rename(
        columns={"time": "target_time", "TWS_t": "next_calendar_TWS"}
    )

    matched = target_rows.merge(
        next_state,
        how="inner",
        on=["target_time", "lat", "lon"],
        validate="many_to_one",
    )

    comparable = matched["target"].notna() & matched["next_calendar_TWS"].notna()
    target_diff = (
        matched.loc[comparable, "target"]
        - matched.loc[comparable, "next_calendar_TWS"]
    ).to_numpy(dtype=np.float64)

    if target_diff.size == 0:
        raise AssertionError("No comparable next-calendar-month target/TWS pairs found")

    max_abs_target_error = float(np.max(np.abs(target_diff)))
    target_identity_rmse = float(np.sqrt(np.mean(np.square(target_diff))))
    exact_target_share = float(np.mean(target_diff == 0.0))

    # The research pass found exact equality.  Fail loudly if the mounted data
    # contradicts that assumption.
    if max_abs_target_error > 1e-12:
        raise AssertionError(
            "Target identity check failed: target(t) is not equal to "
            f"TWS(t+1) for all matched calendar pairs. max_abs_error={max_abs_target_error}"
        )

    print("[5/5] Auditing test TWS mask...")
    masked_count = int(test["TWS_t_masked"].sum())
    test_rows = int(len(test))
    mask_share = float(masked_count / test_rows)
    tws_na_count = int(test["TWS_t"].isna().sum())
    mask_na_mismatch = int(
        (test["TWS_t_masked"].astype(bool) != test["TWS_t"].isna()).sum()
    )

    per_month_mask = (
        test.assign(month=test["time"].dt.to_period("M").astype(str))
        .groupby("month", sort=True)
        .agg(
            rows=("ID", "size"),
            masked=("TWS_t_masked", "sum"),
            visible_TWS=("TWS_t", lambda s: int(s.notna().sum())),
        )
    )
    per_month_mask["masked_share"] = (
        per_month_mask["masked"] / per_month_mask["rows"]
    )

    report = {
        "train_rows": int(len(train)),
        "test_rows": test_rows,
        "train_unique_locations": int(len(train_locations)),
        "test_unique_locations": int(len(test_locations)),
        "train_unique_source_months": int(len(train_months)),
        "test_unique_source_months": int(len(test_months)),
        "train_first_month": train_months[0],
        "train_last_month": train_months[-1],
        "test_first_month": test_months[0],
        "test_last_month": test_months[-1],
        "train_missing_calendar_months": train_missing_months,
        "test_missing_calendar_months": test_missing_months,
        "target_identity_matched_pairs": int(target_diff.size),
        "target_identity_max_abs_error": max_abs_target_error,
        "target_identity_rmse": target_identity_rmse,
        "target_identity_exact_share": exact_target_share,
        "test_masked_rows": masked_count,
        "test_masked_share": mask_share,
        "test_TWS_nan_rows": tws_na_count,
        "test_mask_vs_nan_mismatch_rows": mask_na_mismatch,
    }

    print("\n=== DATASET AUDIT SUMMARY ===")
    print(json.dumps(report, indent=2))
    print("\n=== TEST MASK BY SOURCE MONTH ===")
    print(per_month_mask.to_string())
    print("\nAUDIT PASSED: calendar-safe target identity and structural checks are valid.")

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing Train.csv and Test.csv",
    )
    args = parser.parse_args()
    audit(args.data_dir)


if __name__ == "__main__":
    main()
