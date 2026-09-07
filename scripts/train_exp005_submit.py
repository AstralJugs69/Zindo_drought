from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_lgbm_core import DEFAULT_PARAMS
from src.availability import build_test_availability_ledger, horizon_summary
from src.ml_features import (
    HYDRO_GAP_FEATURE_COLUMNS,
    SOURCE_CORE_COLUMNS,
    SOURCE_HYDRO_HISTORY_COLUMNS,
    build_hydro_gap_feature_matrix,
    build_sampled_training_rows,
    horizon_rebalance_weights,
)


FROZEN_NUM_BOOST_ROUND = 118


def _namespace_train_ids(values: pd.Series) -> pd.Series:
    return "tr__" + values.astype(str)


def _namespace_test_ids(values: pd.Series) -> pd.Series:
    return "te__" + values.astype(str)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train frozen EXP005 on all legal training labels and generate "
            "Submission #1. Intended to run on Kaggle."
        )
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/submission_exp005_r118.csv"),
    )
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--num-boost-round", type=int, default=FROZEN_NUM_BOOST_ROUND)
    args = parser.parse_args()

    if args.num_boost_round != FROZEN_NUM_BOOST_ROUND:
        raise ValueError(
            f"Submission #1 is frozen at {FROZEN_NUM_BOOST_ROUND} rounds; "
            f"received {args.num_boost_round}."
        )

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM is required; run this script on Kaggle.") from exc

    train_path = args.data_dir / "Train.csv"
    test_path = args.data_dir / "Test.csv"
    sample_path = args.data_dir / "SampleSubmission.csv"
    for path in (train_path, test_path, sample_path):
        if not path.exists():
            raise FileNotFoundError(path)

    print("[1/7] Reading train/test/submission template...")
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
    sample = pd.read_csv(sample_path)

    if sample.columns.tolist() != ["ID", "Target"]:
        raise AssertionError(
            f"Unexpected SampleSubmission schema: {sample.columns.tolist()}"
        )
    if test["ID"].duplicated().any() or sample["ID"].duplicated().any():
        raise AssertionError("Test/SampleSubmission IDs must be unique")
    if set(test["ID"].astype(str)) != set(sample["ID"].astype(str)):
        raise AssertionError("Test and SampleSubmission ID sets do not match")

    print("[2/7] Building exact causal test TWS ledger...")
    test_ledger = build_test_availability_ledger(
        test.loc[:, ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]]
    )
    print(horizon_summary(test_ledger).to_string(float_format=lambda x: f"{x:.6f}"))

    # Private namespaced IDs prevent any accidental collision between the Train
    # sample_id namespace and Test ID namespace while reusing target-blind builders.
    train = train.copy()
    test = test.copy()
    train["_internal_id"] = _namespace_train_ids(train["sample_id"])
    test["_internal_id"] = _namespace_test_ids(test["ID"])

    train_structural = train.loc[
        :, ["_internal_id", "time", "lat", "lon", "TWS_t"]
    ].rename(columns={"_internal_id": "sample_id"})
    test_structural = test.loc[
        :, ["_internal_id", "time", "lat", "lon", "TWS_t"]
    ].rename(columns={"_internal_id": "sample_id"})
    structural_all = pd.concat(
        [train_structural, test_structural], ignore_index=True, sort=False
    )
    if structural_all.duplicated(["time", "lat", "lon"]).any():
        raise AssertionError("Combined train/test panel has duplicate location-month rows")

    hydro_cols = SOURCE_HYDRO_HISTORY_COLUMNS[4:]
    train_source = train.loc[
        :, ["_internal_id", "time", "lat", "lon", *hydro_cols]
    ].rename(columns={"_internal_id": "sample_id"})
    test_source = test.loc[
        :, ["_internal_id", "time", "lat", "lon", *hydro_cols]
    ].rename(columns={"_internal_id": "sample_id"})
    source_all = pd.concat([train_source, test_source], ignore_index=True, sort=False)
    source_all = source_all.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS]
    if source_all["sample_id"].duplicated().any():
        raise AssertionError("Internal source-feature IDs are not unique")

    labels = train.loc[:, ["_internal_id", "target"]].rename(
        columns={"_internal_id": "sample_id"}
    )

    print("[3/7] Building all legal sampled-h EXP005 training rows...")
    train_source_period = pd.to_datetime(train["time"]).dt.to_period("M")
    supplied_label = train["target"].notna()
    if not supplied_label.any():
        raise AssertionError("Train contains no supplied labels")
    max_target_month = (train_source_period.loc[supplied_label] + 1).max()
    print("Latest supplied labelled target month:", max_target_month)

    sampled = build_sampled_training_rows(
        train_structural,
        train_source.loc[:, SOURCE_CORE_COLUMNS],
        max_target_month=max_target_month,
        seed=args.seed,
    )

    # Real GRACE calendar gaps can leave source rows without a supplied label.
    # Preserve them in structural history, but never use them as supervised rows.
    label_presence = labels.set_index("sample_id")["target"].notna()
    keep_label = sampled.rows["sample_id"].map(label_presence).fillna(False).to_numpy(bool)
    train_rows = sampled.rows.loc[keep_label].reset_index(drop=True)
    dropped_missing_label = int((~keep_label).sum())
    if train_rows.empty:
        raise AssertionError("No legal training rows remain")

    X_train = build_hydro_gap_feature_matrix(
        train_rows,
        source_all,
        structural_all,
    )
    supervised = train_rows.loc[:, ["sample_id", "last_observed_TWS"]].merge(
        labels,
        how="left",
        on="sample_id",
        validate="one_to_one",
        sort=False,
    )
    if supervised["target"].isna().any():
        raise AssertionError("Missing label after supervised join")
    y_train_delta = (
        supervised["target"].to_numpy(dtype=np.float32)
        - supervised["last_observed_TWS"].to_numpy(dtype=np.float32)
    )
    train_weight = horizon_rebalance_weights(train_rows["h"])

    print("Training rows:", len(train_rows))
    print("Dropped sampled rows lacking supplied label:", dropped_missing_label)
    print(
        "Training h counts:",
        {int(k): int(v) for k, v in train_rows["h"].value_counts().sort_index().items()},
    )

    print("[4/7] Building exact competition-test EXP005 matrix...")
    test_feature_ledger = test_ledger.loc[
        :, [
            "ID", "source_date", "lat", "lon", "last_observed_date",
            "last_observed_TWS", "h",
        ]
    ].copy()
    test_feature_ledger["sample_id"] = _namespace_test_ids(test_feature_ledger["ID"])
    X_test = build_hydro_gap_feature_matrix(
        test_feature_ledger,
        source_all,
        structural_all,
    )
    if len(X_test) != len(test):
        raise AssertionError("Test feature row count changed")

    print("[5/7] Training frozen full-data EXP005 at 118 rounds...")
    params = dict(DEFAULT_PARAMS)
    params["seed"] = args.seed
    params["feature_fraction_seed"] = args.seed
    params["bagging_seed"] = args.seed
    params["data_random_seed"] = args.seed

    started = time.perf_counter()
    train_set = lgb.Dataset(
        X_train,
        label=y_train_delta,
        weight=train_weight,
        feature_name=HYDRO_GAP_FEATURE_COLUMNS,
        free_raw_data=True,
    )
    model = lgb.train(
        params,
        train_set,
        num_boost_round=FROZEN_NUM_BOOST_ROUND,
        callbacks=[lgb.log_evaluation(50)],
    )

    print("[6/7] Predicting Test.csv...")
    delta_pred = model.predict(X_test, num_iteration=FROZEN_NUM_BOOST_ROUND)
    prediction = (
        test_feature_ledger["last_observed_TWS"].to_numpy(dtype=np.float64)
        + delta_pred
    )
    if not np.isfinite(prediction).all():
        raise AssertionError("Predictions contain NaN/inf")

    pred_by_id = pd.DataFrame(
        {
            "_id_str": test_feature_ledger["ID"].astype(str),
            "Target": prediction,
        }
    )
    submission = sample.loc[:, ["ID"]].copy()
    submission["_id_str"] = submission["ID"].astype(str)
    submission = submission.merge(
        pred_by_id,
        how="left",
        on="_id_str",
        validate="one_to_one",
        sort=False,
    )
    if submission["Target"].isna().any():
        raise AssertionError("Submission has missing predictions")
    submission = submission.loc[:, ["ID", "Target"]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    print("[7/7] Submission ready.")
    report = {
        "model": "exp005_hydro_gap_delta_lgbm_full_train_r118",
        "seed": int(args.seed),
        "num_boost_round": FROZEN_NUM_BOOST_ROUND,
        "train_rows": int(len(train_rows)),
        "test_rows": int(len(submission)),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction)),
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
        "train_seconds": float(time.perf_counter() - started),
        "output": str(args.output),
    }
    print(json.dumps(report, indent=2))
    print("\nFirst submission rows:")
    print(submission.head(10).to_string(index=False))
    print(f"\nSUBMISSION_READY={args.output}")


if __name__ == "__main__":
    main()
