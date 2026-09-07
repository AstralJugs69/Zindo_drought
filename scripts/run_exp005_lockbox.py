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
from src.baselines import predict_persistence
from src.metrics import score_by_horizon
from src.ml_features import (
    HYDRO_GAP_FEATURE_COLUMNS,
    SOURCE_CORE_COLUMNS,
    SOURCE_HYDRO_HISTORY_COLUMNS,
    build_hydro_gap_feature_matrix,
    build_sampled_training_rows,
    horizon_rebalance_weights,
    validation_horizon_weights,
)
from src.validation import build_direct_horizon_fold, recent_observed_month_blocks


def _attach_delta_labels(
    feature_rows: pd.DataFrame,
    labels: pd.DataFrame,
) -> np.ndarray:
    joined = feature_rows.loc[:, ["sample_id", "last_observed_TWS"]].merge(
        labels.loc[:, ["sample_id", "target"]],
        how="left",
        on="sample_id",
        validate="one_to_one",
        sort=False,
    )
    if joined["target"].isna().any():
        raise AssertionError("Missing training label after supervised join")
    return (
        joined["target"].to_numpy(dtype=np.float32)
        - joined["last_observed_TWS"].to_numpy(dtype=np.float32)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-time final-candidate check of frozen EXP005 on the protected "
            "2013-11..2015-08 lockbox. This script intentionally exposes no "
            "feature/model switches."
        )
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--num-boost-round", type=int, default=1600)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM is required; run this script on Kaggle.") from exc

    train_path = args.data_dir / "Train.csv"
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    print("[1/6] Reading target-blind state/features and isolated labels...")
    usecols = [
        "sample_id", "time", "lat", "lon", "TWS_t",
        "month_sin", "month_cos",
        "SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t",
        "SOIL_MOISTURE_t", "target",
    ]
    raw = pd.read_csv(train_path, usecols=usecols)
    structural = raw.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
    source_features = raw.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
    labels = raw.loc[:, ["sample_id", "target"]].copy()
    del raw

    print("[2/6] Opening the protected recent lockbox for frozen EXP005 only...")
    blocks = recent_observed_month_blocks(structural, block_size=18, n_blocks=4)
    lockbox_months = blocks[3]
    fold = build_direct_horizon_fold(
        structural.merge(labels, on="sample_id", validate="one_to_one"),
        lockbox_months,
    )

    if str(fold.source_start) != "2013-11" or str(fold.source_end) != "2015-08":
        raise AssertionError(
            f"Unexpected lockbox: {fold.source_start} -> {fold.source_end}"
        )

    print("Lockbox source:", fold.source_start, "->", fold.source_end)
    print("Training target cutoff:", fold.max_training_target_month)

    print("[3/6] Building frozen EXP005 training matrix...")
    sampled = build_sampled_training_rows(
        structural,
        source_features.loc[:, SOURCE_CORE_COLUMNS],
        max_target_month=fold.max_training_target_month,
        seed=args.seed,
    )
    X_train = build_hydro_gap_feature_matrix(
        sampled.rows,
        source_features,
        structural,
    )
    y_train_delta = _attach_delta_labels(sampled.rows, labels)
    train_weight = horizon_rebalance_weights(sampled.rows["h"])

    print("[4/6] Building lockbox validation matrix...")
    X_valid = build_hydro_gap_feature_matrix(
        fold.ledger,
        source_features,
        structural,
    )
    validation_labels = fold.labels.loc[:, ["example_id", "target"]]
    if validation_labels["example_id"].tolist() != fold.ledger["example_id"].tolist():
        raise AssertionError("Validation labels are not aligned to lockbox ledger")
    y_valid = validation_labels["target"].to_numpy(dtype=np.float32)
    y_valid_delta = (
        y_valid
        - fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float32)
    )
    valid_weight = validation_horizon_weights(fold.ledger["h"])

    persistence_pred = predict_persistence(fold.ledger)
    persistence_score, persistence_by_h = score_by_horizon(
        y_valid,
        persistence_pred,
        fold.ledger["h"],
    )

    print("[5/6] Training frozen EXP005...")
    params = dict(DEFAULT_PARAMS)
    params["seed"] = args.seed
    params["feature_fraction_seed"] = args.seed
    params["bagging_seed"] = args.seed
    params["data_random_seed"] = args.seed

    train_set = lgb.Dataset(
        X_train,
        label=y_train_delta,
        weight=train_weight,
        feature_name=HYDRO_GAP_FEATURE_COLUMNS,
        free_raw_data=True,
    )
    valid_set = lgb.Dataset(
        X_valid,
        label=y_valid_delta,
        weight=valid_weight,
        feature_name=HYDRO_GAP_FEATURE_COLUMNS,
        reference=train_set,
        free_raw_data=True,
    )

    started = time.perf_counter()
    model = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[valid_set],
        valid_names=["lockbox"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    print("[6/6] Scoring final-candidate lockbox...")
    delta_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    pred = (
        fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float64)
        + delta_pred
    )
    score, by_h = score_by_horizon(y_valid, pred, fold.ledger["h"])

    comparison = by_h.loc[:, ["rows", "rmse", "bias"]].copy()
    comparison["persistence_rmse"] = persistence_by_h["rmse"]
    comparison["gain"] = comparison["persistence_rmse"] - comparison["rmse"]

    result = {
        "model": "exp005_hydro_gap_delta_lgbm",
        "lockbox_source_start": str(fold.source_start),
        "lockbox_source_end": str(fold.source_end),
        "train_target_cutoff": str(fold.max_training_target_month),
        "train_rows": int(len(sampled.rows)),
        "validation_rows": int(len(fold.ledger)),
        "persistence_weighted_rmse": float(persistence_score),
        "weighted_rmse": float(score),
        "absolute_gain_vs_persistence": float(persistence_score - score),
        "pct_gain_vs_persistence": float(
            (persistence_score - score) / persistence_score * 100.0
        ),
        "best_iteration": int(model.best_iteration),
        "runtime_seconds": float(time.perf_counter() - started),
        "by_h": {
            int(h): {
                "rmse": float(by_h.loc[h, "rmse"]),
                "bias": float(by_h.loc[h, "bias"]),
                "persistence_rmse": float(persistence_by_h.loc[h, "rmse"]),
            }
            for h in range(1, 8)
        },
    }

    print("\n===================================")
    print("=== EXP005 FINAL LOCKBOX RESULT ===")
    print("===================================")
    print(f"persistence = {persistence_score:.6f}")
    print(f"EXP005      = {score:.6f}")
    print(f"gain        = {persistence_score - score:+.6f}")
    print(
        "relative    = "
        f"{(persistence_score - score) / persistence_score * 100.0:+.2f}%"
    )
    print(f"best_iter   = {model.best_iteration}")
    print("\nBy horizon:")
    print(comparison.to_string(float_format=lambda x: f"{x:.6f}"))
    print("\nJSON_RESULT")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
