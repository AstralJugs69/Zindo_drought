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


def _summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, sort=True, dropna=False)
    out = grouped.agg(
        rows=("ID", "size"),
        visible_share=("tws_visible", "mean"),
        anchor_mean=("last_observed_TWS", "mean"),
        anchor_std=("last_observed_TWS", "std"),
        pred_mean=("prediction", "mean"),
        pred_std=("prediction", "std"),
        persistence_mean=("persistence", "mean"),
        correction_mean=("correction", "mean"),
        correction_abs_mean=("correction_abs", "mean"),
        correction_std=("correction", "std"),
        SPEI_06_mean=("SPEI_06_t", "mean"),
        SPEI_12_mean=("SPEI_12_t", "mean"),
        soil_mean=("SOIL_MOISTURE_t", "mean"),
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-training audit of a submission versus legal TWS persistence, "
            "broken down by test month and effective horizon."
        )
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    args = parser.parse_args()

    test_path = args.data_dir / "Test.csv"
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    if not args.submission.exists():
        raise FileNotFoundError(args.submission)

    print("[1/4] Reading Test.csv and submission...")
    test = pd.read_csv(
        test_path,
        usecols=[
            "ID",
            "time",
            "lat",
            "lon",
            "TWS_t",
            "TWS_t_masked",
            "SPEI_01_t",
            "SPEI_03_t",
            "SPEI_06_t",
            "SPEI_12_t",
            "SOIL_MOISTURE_t",
        ],
    )
    sub = pd.read_csv(args.submission)
    if sub.columns.tolist() != ["ID", "Target"]:
        raise AssertionError(f"Unexpected submission columns: {sub.columns.tolist()}")
    if test["ID"].duplicated().any() or sub["ID"].duplicated().any():
        raise AssertionError("Duplicate IDs in Test/submission")

    print("[2/4] Rebuilding exact causal test ledger...")
    ledger = build_test_availability_ledger(
        test.loc[:, ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]]
    )

    base = test.loc[:, [
        "ID", "time", "SPEI_01_t", "SPEI_03_t", "SPEI_06_t",
        "SPEI_12_t", "SOIL_MOISTURE_t",
    ]].copy()
    base["_id_str"] = base["ID"].astype(str)
    ledger = ledger.copy()
    ledger["_id_str"] = ledger["ID"].astype(str)
    sub = sub.copy()
    sub["_id_str"] = sub["ID"].astype(str)

    x = ledger.merge(
        base.drop(columns=["ID"]),
        on="_id_str",
        how="left",
        validate="one_to_one",
        sort=False,
    ).merge(
        sub.loc[:, ["_id_str", "Target"]].rename(columns={"Target": "prediction"}),
        on="_id_str",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if x["prediction"].isna().any():
        raise AssertionError("Missing prediction after ID join")
    if not np.isfinite(x["prediction"].to_numpy(dtype=np.float64)).all():
        raise AssertionError("Submission prediction contains NaN/inf")

    x["source_month"] = pd.to_datetime(x["source_date"]).dt.to_period("M").astype(str)
    x["source_year"] = pd.to_datetime(x["source_date"]).dt.year.astype(np.int16)
    x["persistence"] = x["last_observed_TWS"].astype(np.float64)
    x["correction"] = x["prediction"] - x["persistence"]
    x["correction_abs"] = x["correction"].abs()

    print("[3/4] Computing month/horizon diagnostics...")
    by_month = _summary(x, ["source_month"])
    by_h = _summary(x, ["h"])
    by_year = _summary(x, ["source_year"])
    by_month_h = _summary(x, ["source_month", "h"])

    overall = {
        "rows": int(len(x)),
        "prediction_mean": float(x["prediction"].mean()),
        "prediction_std": float(x["prediction"].std(ddof=0)),
        "persistence_mean": float(x["persistence"].mean()),
        "persistence_std": float(x["persistence"].std(ddof=0)),
        "mean_model_minus_persistence": float(x["correction"].mean()),
        "mean_abs_model_minus_persistence": float(x["correction_abs"].mean()),
        "max_abs_model_minus_persistence": float(x["correction_abs"].max()),
        "corr_model_persistence": float(x[["prediction", "persistence"]].corr().iloc[0, 1]),
    }

    print("[4/4] Audit report...")
    print("\n=== OVERALL ===")
    print(json.dumps(overall, indent=2))
    print("\n=== BY SOURCE MONTH ===")
    print(by_month.to_string(float_format=lambda v: f"{v:.6f}"))
    print("\n=== BY HORIZON ===")
    print(by_h.to_string(float_format=lambda v: f"{v:.6f}"))
    print("\n=== BY SOURCE YEAR ===")
    print(by_year.to_string(float_format=lambda v: f"{v:.6f}"))
    print("\n=== BY SOURCE MONTH x H (non-empty cells) ===")
    print(by_month_h.to_string(float_format=lambda v: f"{v:.6f}"))

    print("\nSUBMISSION SHIFT AUDIT COMPLETE: no model training or target scoring performed.")


if __name__ == "__main__":
    main()
