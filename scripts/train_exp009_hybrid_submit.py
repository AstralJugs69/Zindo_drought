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
    HYDRO_GAP_SAFE_FEATURE_COLUMNS,
    SOURCE_CORE_COLUMNS,
    SOURCE_HYDRO_HISTORY_COLUMNS,
    build_hydro_gap_feature_matrix,
    build_hydro_gap_safe_feature_matrix,
    build_sampled_training_rows,
    horizon_rebalance_weights,
)


EXP005_ROUNDS = 183
EXP009_ROUNDS = 173


def _submission_from_prediction(
    sample: pd.DataFrame,
    test_ids: pd.Series,
    prediction: np.ndarray,
    output: Path,
) -> pd.DataFrame:
    pred_by_id = pd.DataFrame(
        {"_id_str": test_ids.astype(str), "Target": prediction.astype(np.float64)}
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
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train availability-safe EXP009 and the replay-selected EXP010 hybrid. "
            "Intended to run on Kaggle only."
        )
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--exp009-output",
        type=Path,
        default=Path("/kaggle/working/submission_exp009_r173.csv"),
    )
    parser.add_argument(
        "--hybrid-output",
        type=Path,
        default=Path("/kaggle/working/submission_exp010_hybrid.csv"),
    )
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

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

    print("[1/8] Reading Train/Test/SampleSubmission...")
    usecols_train = [
        "sample_id", "time", "lat", "lon", "TWS_t",
        "month_sin", "month_cos",
        "SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t",
        "SOIL_MOISTURE_t", "target",
    ]
    usecols_test = [
        "ID", "time", "lat", "lon", "TWS_t",
        "month_sin", "month_cos",
        "SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t",
        "SOIL_MOISTURE_t", "TWS_t_masked",
    ]
    train = pd.read_csv(train_path, usecols=usecols_train)
    test = pd.read_csv(test_path, usecols=usecols_test)
    sample = pd.read_csv(sample_path)

    if sample.columns.tolist() != ["ID", "Target"]:
        raise AssertionError(f"Unexpected SampleSubmission schema: {sample.columns.tolist()}")
    if set(test["ID"].astype(str)) != set(sample["ID"].astype(str)):
        raise AssertionError("Test and SampleSubmission ID sets do not match")

    print("[2/8] Building exact causal Test availability ledger...")
    test_ledger = build_test_availability_ledger(
        test.loc[:, ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]]
    )
    print(horizon_summary(test_ledger).to_string(float_format=lambda x: f"{x:.6f}"))

    # IMPORTANT: preserve raw Train sample_id for the deterministic-horizon sampler.
    # Submission #1 namespaced Train ids before sampling, which changed the hash-based
    # h assignment relative to development. Here training uses the exact development
    # recipe; namespacing is restricted to the Test-side feature panel only.
    train_structural = train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
    train_source = train.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
    labels = train.loc[:, ["sample_id", "target"]].copy()

    print("[3/8] Building all legal sampled-h training rows with development-stable IDs...")
    train_source_period = pd.to_datetime(train["time"]).dt.to_period("M")
    supplied_label = train["target"].notna()
    max_target_month = (train_source_period.loc[supplied_label] + 1).max()
    sampled = build_sampled_training_rows(
        train_structural,
        train_source.loc[:, SOURCE_CORE_COLUMNS],
        max_target_month=max_target_month,
        seed=args.seed,
    )
    label_presence = labels.set_index("sample_id")["target"].notna()
    keep_label = sampled.rows["sample_id"].map(label_presence).fillna(False).to_numpy(bool)
    train_rows = sampled.rows.loc[keep_label].reset_index(drop=True)
    if train_rows.empty:
        raise AssertionError("No legal training rows remain")

    supervised = train_rows.loc[:, ["sample_id", "last_observed_TWS"]].merge(
        labels,
        how="left",
        on="sample_id",
        validate="one_to_one",
        sort=False,
    )
    if supervised["target"].isna().any():
        raise AssertionError("Missing training target after supervised join")
    y_train_delta = (
        supervised["target"].to_numpy(dtype=np.float32)
        - supervised["last_observed_TWS"].to_numpy(dtype=np.float32)
    )
    train_weight = horizon_rebalance_weights(train_rows["h"])
    print("Training rows:", len(train_rows))
    print("Training h counts:", {
        int(k): int(v) for k, v in train_rows["h"].value_counts().sort_index().items()
    })

    print("[4/8] Building EXP005 and EXP009 training matrices...")
    X_train_005 = build_hydro_gap_feature_matrix(
        train_rows, train_source, train_structural
    )
    X_train_009 = build_hydro_gap_safe_feature_matrix(
        train_rows, train_source, train_structural
    )

    print("[5/8] Building exact competition-Test feature matrices...")
    # Test rows need unique internal IDs for current-row feature joins. Historical
    # lookups themselves are keyed by calendar month + location.
    test_ns = test.copy()
    test_ns["_internal_id"] = "te__" + test_ns["ID"].astype(str)
    train_ns = train.copy()
    train_ns["_internal_id"] = "tr__" + train_ns["sample_id"].astype(str)

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
        :, ["ID", "source_date", "lat", "lon", "last_observed_date", "last_observed_TWS", "h"]
    ].copy()
    test_feature_ledger["sample_id"] = "te__" + test_feature_ledger["ID"].astype(str)
    X_test_005 = build_hydro_gap_feature_matrix(
        test_feature_ledger, source_all, structural_all
    )
    X_test_009 = build_hydro_gap_safe_feature_matrix(
        test_feature_ledger, source_all, structural_all
    )

    params = dict(DEFAULT_PARAMS)
    params["seed"] = args.seed
    params["feature_fraction_seed"] = args.seed
    params["bagging_seed"] = args.seed
    params["data_random_seed"] = args.seed

    print(f"[6/8] Training EXP005 at replay-selected {EXP005_ROUNDS} rounds...")
    started = time.perf_counter()
    model005 = lgb.train(
        params,
        lgb.Dataset(
            X_train_005,
            label=y_train_delta,
            weight=train_weight,
            feature_name=HYDRO_GAP_FEATURE_COLUMNS,
            free_raw_data=True,
        ),
        num_boost_round=EXP005_ROUNDS,
        callbacks=[lgb.log_evaluation(50)],
    )
    pred005 = (
        test_feature_ledger["last_observed_TWS"].to_numpy(dtype=np.float64)
        + model005.predict(X_test_005, num_iteration=EXP005_ROUNDS)
    )

    print(f"[7/8] Training EXP009 at replay-selected {EXP009_ROUNDS} rounds...")
    model009 = lgb.train(
        params,
        lgb.Dataset(
            X_train_009,
            label=y_train_delta,
            weight=train_weight,
            feature_name=HYDRO_GAP_SAFE_FEATURE_COLUMNS,
            free_raw_data=True,
        ),
        num_boost_round=EXP009_ROUNDS,
        callbacks=[lgb.log_evaluation(50)],
    )
    pred009 = (
        test_feature_ledger["last_observed_TWS"].to_numpy(dtype=np.float64)
        + model009.predict(X_test_009, num_iteration=EXP009_ROUNDS)
    )

    h = test_feature_ledger["h"].to_numpy(dtype=np.int8)
    pred_hybrid = np.where(h <= 2, pred005, pred009)
    for name, pred in (("EXP005", pred005), ("EXP009", pred009), ("EXP010", pred_hybrid)):
        if not np.isfinite(pred).all():
            raise AssertionError(f"{name} predictions contain NaN/inf")

    print("[8/8] Writing pure EXP009 and h1-2/h3-7 hybrid submissions...")
    sub009 = _submission_from_prediction(
        sample, test_feature_ledger["ID"], pred009, args.exp009_output
    )
    sub010 = _submission_from_prediction(
        sample, test_feature_ledger["ID"], pred_hybrid, args.hybrid_output
    )

    report = {
        "training_rows": int(len(train_rows)),
        "exp005_rounds": EXP005_ROUNDS,
        "exp009_rounds": EXP009_ROUNDS,
        "hybrid_rule": "EXP005 for h<=2; EXP009 for h>=3",
        "train_seconds_total": float(time.perf_counter() - started),
        "exp005_prediction": {
            "mean": float(np.mean(pred005)), "std": float(np.std(pred005)),
            "min": float(np.min(pred005)), "max": float(np.max(pred005)),
        },
        "exp009_prediction": {
            "mean": float(np.mean(pred009)), "std": float(np.std(pred009)),
            "min": float(np.min(pred009)), "max": float(np.max(pred009)),
        },
        "hybrid_prediction": {
            "mean": float(np.mean(pred_hybrid)), "std": float(np.std(pred_hybrid)),
            "min": float(np.min(pred_hybrid)), "max": float(np.max(pred_hybrid)),
        },
        "mean_abs_exp009_minus_exp005": float(np.mean(np.abs(pred009 - pred005))),
        "corr_exp009_exp005": float(np.corrcoef(pred009, pred005)[0, 1]),
        "exp009_output": str(args.exp009_output),
        "hybrid_output": str(args.hybrid_output),
    }
    print(json.dumps(report, indent=2))
    print("\nFirst EXP009 rows:")
    print(sub009.head(5).to_string(index=False))
    print("\nFirst hybrid rows:")
    print(sub010.head(5).to_string(index=False))
    print(f"\nEXP009_READY={args.exp009_output}")
    print(f"HYBRID_READY={args.hybrid_output}")


if __name__ == "__main__":
    main()
