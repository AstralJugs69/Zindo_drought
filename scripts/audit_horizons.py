from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.availability import build_test_availability_ledger, horizon_summary


EXPECTED_COUNTS = {
    1: 94048,
    2: 62576,
    3: 46777,
    4: 31076,
    5: 15560,
    6: 15479,
    7: 15445,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()

    test_path = args.data_dir / "Test.csv"
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    print("[1/3] Reading test state columns...")
    test = pd.read_csv(
        test_path,
        usecols=["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"],
    )

    print("[2/3] Building causal TWS availability ledger...")
    ledger = build_test_availability_ledger(test)

    print("[3/3] Verifying exact h=1..7 distribution...")
    summary = horizon_summary(ledger)
    observed_counts = summary["rows"].astype(int).to_dict()

    if observed_counts != EXPECTED_COUNTS:
        raise AssertionError(
            "Unexpected horizon distribution.\n"
            f"Observed: {observed_counts}\n"
            f"Expected: {EXPECTED_COUNTS}"
        )

    if int(summary.index.min()) != 1 or int(summary.index.max()) != 7:
        raise AssertionError("Expected exact horizon range h=1..7.")

    weighted_mean_h = float((summary.index.to_numpy() * summary["share"].to_numpy()).sum())

    print("\n=== EFFECTIVE TWS HORIZON DISTRIBUTION ===")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n=== AVAILABILITY LEDGER CHECKS ===")
    report = {
        "rows": int(len(ledger)),
        "locations": int(ledger["location_id"].nunique()),
        "min_h": int(ledger["h"].min()),
        "max_h": int(ledger["h"].max()),
        "weighted_mean_h": weighted_mean_h,
        "visible_rows": int(ledger["tws_visible"].sum()),
        "masked_rows": int(ledger["TWS_t_masked"].sum()),
        "masked_rows_using_same_month_hidden_tws": int(
            (
                ledger["TWS_t_masked"]
                & (ledger["last_observed_date"] == ledger["source_date"])
            ).sum()
        ),
    }
    print(json.dumps(report, indent=2))

    print("\nSample masked rows:")
    print(
        ledger.loc[
            ledger["TWS_t_masked"],
            [
                "ID",
                "source_date",
                "last_observed_date",
                "last_observed_TWS",
                "h",
            ],
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nHORIZON AUDIT PASSED: causal TWS availability ledger matches test geometry exactly.")


if __name__ == "__main__":
    main()

