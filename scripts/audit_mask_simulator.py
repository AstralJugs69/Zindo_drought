from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation import (
    build_exact_historical_mask_fold,
    build_test_mask_template,
    find_exact_template_starts,
)


TEST_HORIZON_SHARES = {
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

    print("[1/5] Reading structural train/test columns...")
    train = pd.read_csv(
        args.data_dir / "Train.csv",
        usecols=["sample_id", "time", "lat", "lon", "TWS_t", "target"],
    )
    test = pd.read_csv(
        args.data_dir / "Test.csv",
        usecols=["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"],
    )

    print("[2/5] Building exact test mask template...")
    template = build_test_mask_template(test)
    offsets = sorted(template["offset_months"].unique().tolist())

    print("[3/5] Finding historical calendar shifts that preserve all 18 test source-month slots...")
    starts = find_exact_template_starts(train, template)
    if not starts:
        raise AssertionError("No exact historical calendar shift can replay the test template")
    latest = starts[-1]

    print("[4/5] Replaying latest exact historical template...")
    fold = build_exact_historical_mask_fold(train, template, latest)
    ledger = fold.ledger

    if "target" in ledger.columns:
        raise AssertionError("Target leaked into ledger")
    if len(fold.labels) != len(ledger):
        raise AssertionError("Ledger/label row counts differ")
    if ledger["sample_id"].duplicated().any():
        raise AssertionError("Duplicate sample_id in historical fold")

    print("[5/5] Running hidden-truth invariance check...")
    hidden_ids = set(ledger.loc[~ledger["sim_tws_visible"], "sample_id"])
    mutated = train.copy()
    mask = mutated["sample_id"].isin(hidden_ids)
    rng = np.random.default_rng(20260907)
    mutated.loc[mask, "TWS_t"] = rng.normal(999.0, 50.0, size=int(mask.sum()))
    fold_mut = build_exact_historical_mask_fold(mutated, template, latest)

    cols = ["sample_id", "last_observed_date", "last_observed_TWS", "h"]
    left = ledger[cols].sort_values("sample_id").reset_index(drop=True)
    right = fold_mut.ledger[cols].sort_values("sample_id").reset_index(drop=True)

    same_dates = left["last_observed_date"].equals(right["last_observed_date"])
    same_h = left["h"].equals(right["h"])
    same_tws = np.allclose(left["last_observed_TWS"], right["last_observed_TWS"], equal_nan=True)
    if not (same_dates and same_h and same_tws):
        raise AssertionError("Hidden validation TWS changed the causal ledger")

    h_summary = ledger["h"].value_counts().sort_index().rename("rows").to_frame()
    h_summary["share"] = h_summary["rows"] / len(ledger)
    max_h_share_delta = max(
        abs(float(h_summary.loc[h, "share"]) - expected)
        for h, expected in TEST_HORIZON_SHARES.items()
    )
    if max_h_share_delta > 0.002:
        raise AssertionError(
            f"Historical replay horizon mix drifted too far from test: max share delta={max_h_share_delta}"
        )

    report = {
        "exact_template_candidate_starts": len(starts),
        "earliest_exact_start": str(starts[0]),
        "latest_exact_start": str(latest),
        "latest_exact_end": str(fold.end_month),
        "fold_rows": int(len(ledger)),
        "fold_locations": int(ledger["location_id"].nunique()),
        "visible_rows": int(ledger["sim_tws_visible"].sum()),
        "hidden_rows": int((~ledger["sim_tws_visible"]).sum()),
        "dropped_missing_anchor_rows": int(fold.dropped_missing_anchor),
        "min_h": int(ledger["h"].min()),
        "max_h": int(ledger["h"].max()),
        "target_columns_in_ledger": int("target" in ledger.columns),
        "hidden_truth_invariance": bool(same_dates and same_h and same_tws),
        "max_abs_horizon_share_delta_vs_test": float(max_h_share_delta),
        "template_offsets": offsets,
    }

    print("\n=== EXACT HISTORICAL MASK SIMULATOR ===")
    print(json.dumps(report, indent=2))
    print("\n=== HISTORICAL FOLD HORIZON DISTRIBUTION ===")
    print(h_summary.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\nMASK SIMULATOR AUDIT PASSED: exact calendar transplant is causal and hidden-truth invariant.")


if __name__ == "__main__":
    main()
