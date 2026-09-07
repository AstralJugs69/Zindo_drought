from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation import build_direct_horizon_fold, recent_observed_month_blocks

TEST_H_WEIGHTS = {
    1: 0.334737,
    2: 0.222721,
    3: 0.166489,
    4: 0.110606,
    5: 0.055381,
    6: 0.055093,
    7: 0.054972,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()

    train_path = args.data_dir / "Train.csv"
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    print("[1/4] Reading structural train columns...")
    train = pd.read_csv(
        train_path,
        usecols=["sample_id", "time", "lat", "lon", "TWS_t", "target"],
    )

    print("[2/4] Defining the four most recent 18-observed-month blocks...")
    blocks = recent_observed_month_blocks(train, block_size=18, n_blocks=4)
    names = ["dev1", "dev2", "dev3", "lockbox"]

    print("[3/4] Building direct h=1..7 causal ledgers...")
    fold_reports = []
    all_ok = True

    # Number of source rows per block is the theoretical max examples per horizon.
    train_period = pd.to_datetime(train["time"]).dt.to_period("M")

    for name, months in zip(names, blocks):
        fold = build_direct_horizon_fold(train, months)
        ledger = fold.ledger

        source_rows = int(train_period.isin(months).sum())
        by_h = ledger.groupby("h", sort=True).size().rename("rows").to_frame()
        by_h["coverage"] = by_h["rows"] / source_rows
        by_h["test_weight"] = [TEST_H_WEIGHTS[int(h)] for h in by_h.index]

        if set(by_h.index.tolist()) != set(range(1, 8)):
            all_ok = False

        # Critical structural checks.
        target_columns = [c for c in ledger.columns if c.lower() == "target"]
        h_gt1_same_month = int(
            ((ledger["h"] > 1) & (ledger["last_observed_date"] == ledger["source_date"])).sum()
        )
        duplicated_examples = int(ledger["example_id"].duplicated().sum())
        min_coverage = float(by_h["coverage"].min())

        if target_columns or h_gt1_same_month or duplicated_examples:
            all_ok = False

        report = {
            "fold": name,
            "source_start": str(fold.source_start),
            "source_end": str(fold.source_end),
            "first_target_month": str(fold.first_target_month),
            "max_training_target_month": str(fold.max_training_target_month),
            "observed_source_months": int(len(months)),
            "source_rows": source_rows,
            "validation_examples": int(len(ledger)),
            "locations": int(ledger["location_id"].nunique()),
            "min_h": int(ledger["h"].min()),
            "max_h": int(ledger["h"].max()),
            "target_columns_in_ledger": len(target_columns),
            "h_gt1_rows_using_source_month_tws": h_gt1_same_month,
            "duplicated_example_ids": duplicated_examples,
            "min_anchor_coverage_across_h": min_coverage,
        }
        fold_reports.append(report)

        print(f"\n=== {name.upper()} ===")
        print(json.dumps(report, indent=2))
        print("\nCoverage by horizon:")
        print(by_h.to_string(float_format=lambda x: f"{x:.6f}"))

    print("[4/4] Verifying fold chronology and lockbox separation...")
    for prev, nxt in zip(fold_reports, fold_reports[1:]):
        prev_end = pd.Period(prev["source_end"], freq="M")
        next_start = pd.Period(nxt["source_start"], freq="M")
        if not (prev_end < next_start):
            all_ok = False
            raise AssertionError(f"Fold overlap: {prev['fold']} -> {nxt['fold']}")

    summary = {
        "folds": [r["fold"] for r in fold_reports],
        "dev_fold_count": 3,
        "lockbox": "lockbox",
        "lockbox_source_start": fold_reports[-1]["source_start"],
        "lockbox_source_end": fold_reports[-1]["source_end"],
        "all_ledgers_target_blind": all(r["target_columns_in_ledger"] == 0 for r in fold_reports),
        "all_h_gt1_source_tws_hidden": all(r["h_gt1_rows_using_source_month_tws"] == 0 for r in fold_reports),
        "all_horizons_present": all(r["min_h"] == 1 and r["max_h"] == 7 for r in fold_reports),
    }

    print("\n=== RECENT DIRECT-HORIZON CV SUMMARY ===")
    print(json.dumps(summary, indent=2))

    if not all_ok or not all(summary[k] for k in [
        "all_ledgers_target_blind",
        "all_h_gt1_source_tws_hidden",
        "all_horizons_present",
    ]):
        raise AssertionError("Recent direct-horizon fold audit failed")

    print("\nRECENT FOLD AUDIT PASSED: direct h=1..7 folds are causal, target-blind, and chronologically separated.")


if __name__ == "__main__":
    main()
