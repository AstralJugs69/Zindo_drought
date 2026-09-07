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


from src.availability import build_test_availability_ledger
from src.ml_features import (
    SOURCE_HYDRO_HISTORY_COLUMNS,
    build_hydro_gap_feature_matrix,
)
from src.validation import build_direct_horizon_fold, recent_observed_month_blocks


TWS_HISTORY_ONLY = [
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


def _summary(name: str, frame: pd.DataFrame) -> dict:
    missing = frame[TWS_HISTORY_ONLY].isna()
    out = {
        "name": name,
        "rows": int(len(frame)),
        "fully_available_share": float((~missing.any(axis=1)).mean()),
        "any_missing_share": float(missing.any(axis=1).mean()),
        "all_missing_share": float(missing.all(axis=1).mean()),
        "feature_available_share": {
            c: float(frame[c].notna().mean()) for c in TWS_HISTORY_ONLY
        },
    }
    return out


def _group_availability(meta: pd.DataFrame, features: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    x = meta.loc[:, keys].reset_index(drop=True).copy()
    available = features[TWS_HISTORY_ONLY].notna().reset_index(drop=True)
    for c in TWS_HISTORY_ONLY:
        x[f"{c}_avail"] = available[c].astype(np.float32)
    x["all_tws_history_available"] = available.all(axis=1).astype(np.float32)
    agg_cols = [c for c in x.columns if c not in keys]
    out = x.groupby(keys, sort=True)[agg_cols].mean()
    counts = x.groupby(keys, sort=True).size().rename("rows")
    return counts.to_frame().join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "No-training audit of EXP005 TWS-history feature availability in "
            "historical validation versus the sparse real Test calendar."
        )
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()

    train_path = args.data_dir / "Train.csv"
    test_path = args.data_dir / "Test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected Train.csv and Test.csv")

    print("[1/5] Reading train/test target-blind state and source features...")
    train = pd.read_csv(
        train_path,
        usecols=[
            "sample_id", "time", "lat", "lon", "TWS_t",
            "month_sin", "month_cos",
            "SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t",
            "SOIL_MOISTURE_t", "target",
        ],
    )
    test = pd.read_csv(
        test_path,
        usecols=[
            "ID", "time", "lat", "lon", "TWS_t",
            "month_sin", "month_cos",
            "SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t",
            "SOIL_MOISTURE_t", "TWS_t_masked",
        ],
    )

    train_structural = train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
    train_source = train.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()

    print("[2/5] Building dev3/lockbox EXP005 feature matrices exactly as historical CV did...")
    blocks = recent_observed_month_blocks(train_structural, block_size=18, n_blocks=4)
    historical_reports = []
    for name, idx in (("dev3", 2), ("lockbox", 3)):
        fold = build_direct_horizon_fold(train, blocks[idx])
        X = build_hydro_gap_feature_matrix(fold.ledger, train_source, train_structural)
        historical_reports.append(_summary(name, X))
        print(f"\n=== {name.upper()} HISTORY AVAILABILITY ===")
        print(json.dumps(historical_reports[-1], indent=2))

    print("[3/5] Building real Test EXP005 feature matrix exactly as submission inference did...")
    test_ledger = build_test_availability_ledger(
        test.loc[:, ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]]
    )

    train_ns = train.copy()
    test_ns = test.copy()
    train_ns["_internal_id"] = "tr__" + train_ns["sample_id"].astype(str)
    test_ns["_internal_id"] = "te__" + test_ns["ID"].astype(str)

    train_structural_ns = train_ns.loc[
        :, ["_internal_id", "time", "lat", "lon", "TWS_t"]
    ].rename(columns={"_internal_id": "sample_id"})
    test_structural_ns = test_ns.loc[
        :, ["_internal_id", "time", "lat", "lon", "TWS_t"]
    ].rename(columns={"_internal_id": "sample_id"})
    structural_all = pd.concat(
        [train_structural_ns, test_structural_ns], ignore_index=True, sort=False
    )

    hydro_cols = SOURCE_HYDRO_HISTORY_COLUMNS[4:]
    train_source_ns = train_ns.loc[
        :, ["_internal_id", "time", "lat", "lon", *hydro_cols]
    ].rename(columns={"_internal_id": "sample_id"})
    test_source_ns = test_ns.loc[
        :, ["_internal_id", "time", "lat", "lon", *hydro_cols]
    ].rename(columns={"_internal_id": "sample_id"})
    source_all = pd.concat([train_source_ns, test_source_ns], ignore_index=True, sort=False)
    source_all = source_all.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS]

    test_feature_ledger = test_ledger.loc[
        :, [
            "ID", "source_date", "lat", "lon", "last_observed_date",
            "last_observed_TWS", "h",
        ]
    ].copy()
    test_feature_ledger["sample_id"] = "te__" + test_feature_ledger["ID"].astype(str)
    X_test = build_hydro_gap_feature_matrix(
        test_feature_ledger,
        source_all,
        structural_all,
    )

    test_report = _summary("test", X_test)
    print("\n=== REAL TEST HISTORY AVAILABILITY ===")
    print(json.dumps(test_report, indent=2))

    print("[4/5] Breaking Test availability down by source month and horizon...")
    meta = test_ledger.loc[:, ["source_date", "h"]].copy()
    meta["source_month"] = pd.to_datetime(meta["source_date"]).dt.to_period("M").astype(str)

    by_h = _group_availability(meta, X_test, ["h"])
    by_month = _group_availability(meta, X_test, ["source_month"])

    keep_cols = [
        "rows",
        "TWS_anchor_lag1_avail",
        "TWS_anchor_lag2_avail",
        "TWS_anchor_lag3_avail",
        "TWS_anchor_lag6_avail",
        "TWS_anchor_lag12_avail",
        "all_tws_history_available",
    ]

    print("\n=== TEST AVAILABILITY BY HORIZON ===")
    print(by_h.loc[:, keep_cols].to_string(float_format=lambda x: f"{x:.6f}"))
    print("\n=== TEST AVAILABILITY BY SOURCE MONTH ===")
    print(by_month.loc[:, keep_cols].to_string(float_format=lambda x: f"{x:.6f}"))

    print("[5/5] Cross-domain availability gap...")
    comparison = pd.DataFrame(
        [
            {
                "domain": r["name"],
                "rows": r["rows"],
                "fully_available_share": r["fully_available_share"],
                **{
                    c: r["feature_available_share"][c]
                    for c in [
                        "TWS_anchor_lag1",
                        "TWS_anchor_lag2",
                        "TWS_anchor_lag3",
                        "TWS_anchor_lag6",
                        "TWS_anchor_lag12",
                    ]
                },
            }
            for r in [*historical_reports, test_report]
        ]
    )
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(
        "\nFEATURE AVAILABILITY AUDIT COMPLETE: no model training or target scoring performed."
    )


if __name__ == "__main__":
    main()
