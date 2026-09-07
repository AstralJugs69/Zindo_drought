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

from src.baselines import predict_persistence
from src.metrics import score_by_horizon
from src.ml_features import (
    CORE_FEATURE_COLUMNS,
    HYDRO_GAP_FEATURE_COLUMNS,
    HYDRO_FEATURE_COLUMNS,
    LOCATION_CAT_FEATURE_COLUMNS,
    TWS_HISTORY_FEATURE_COLUMNS,
    SOURCE_CORE_COLUMNS,
    SOURCE_HYDRO_COLUMNS,
    SOURCE_HYDRO_HISTORY_COLUMNS,
    build_core_feature_matrix,
    build_hydro_gap_feature_matrix,
    build_hydro_feature_matrix,
    build_location_cat_feature_matrix,
    build_tws_history_feature_matrix,
    build_sampled_training_rows,
    horizon_rebalance_weights,
    validation_horizon_weights,
)
from src.validation import build_direct_horizon_fold, recent_observed_month_blocks


DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 1000,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "max_bin": 255,
    "verbosity": -1,
    "force_col_wise": True,
    "seed": 20260907,
    "feature_fraction_seed": 20260907,
    "bagging_seed": 20260907,
    "data_random_seed": 20260907,
}


def _attach_labels(
    feature_rows: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if labels["sample_id"].duplicated().any():
        raise AssertionError("Training labels sample_id must be unique")
    joined = feature_rows.loc[:, ["sample_id", "last_observed_TWS"]].merge(
        labels.loc[:, ["sample_id", "target"]],
        how="left",
        on="sample_id",
        validate="one_to_one",
        sort=False,
    )
    if joined["target"].isna().any():
        raise AssertionError("Missing training label after final supervised join")
    y_absolute = joined["target"].to_numpy(dtype=np.float32)
    y_delta = y_absolute - joined["last_observed_TWS"].to_numpy(dtype=np.float32)
    return y_delta, y_absolute


def _validation_labels(fold) -> np.ndarray:
    labels = fold.labels.loc[:, ["example_id", "target"]].copy()
    if labels["example_id"].tolist() != fold.ledger["example_id"].tolist():
        raise AssertionError("Validation labels are not aligned to ledger")
    return labels["target"].to_numpy(dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--folds", nargs="+", default=["dev1", "dev2", "dev3"])
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--num-boost-round", type=int, default=1600)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument(
        "--feature-set",
        choices=["core", "hydro", "tws_history", "hydro_gap", "location_cat"],
        default="core",
        help=(
            "core=EXP001; hydro=EXP002 fresh SPEI+soil; "
            "tws_history=EXP003 adds exact-calendar TWS history behind legal anchor; "
            "hydro_gap=EXP005 adds current-minus-anchor hydrology deltas; "
            "location_cat=EXP006 adds categorical location identity"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    allowed_names = {"dev1": 0, "dev2": 1, "dev3": 2}
    unknown = [name for name in args.folds if name not in allowed_names]
    if unknown:
        raise ValueError(
            f"Development experiments intentionally support dev1-dev3 only; lockbox is protected. Unknown: {unknown}"
        )

    train_path = args.data_dir / "Train.csv"
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    print("[1/5] Reading target-blind state/features and isolated labels...")
    usecols = [
        "sample_id", "time", "lat", "lon", "TWS_t",
        "month_sin", "month_cos",
        "SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t",
        "SOIL_MOISTURE_t", "target",
    ]
    raw = pd.read_csv(train_path, usecols=usecols)
    structural = raw.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
    if args.feature_set == "core":
        feature_columns = CORE_FEATURE_COLUMNS
        source_columns = SOURCE_CORE_COLUMNS
        feature_builder = lambda ledger, sf: build_core_feature_matrix(ledger, sf)
        experiment_name = "EXP001"
        model_name = "exp001_core_delta_lgbm"
        categorical_features = []
    elif args.feature_set == "hydro":
        feature_columns = HYDRO_FEATURE_COLUMNS
        source_columns = SOURCE_HYDRO_COLUMNS
        feature_builder = lambda ledger, sf: build_hydro_feature_matrix(ledger, sf)
        experiment_name = "EXP002"
        model_name = "exp002_fresh_hydro_delta_lgbm"
        categorical_features = []
    elif args.feature_set == "tws_history":
        feature_columns = TWS_HISTORY_FEATURE_COLUMNS
        source_columns = SOURCE_HYDRO_COLUMNS
        feature_builder = lambda ledger, sf: build_tws_history_feature_matrix(
            ledger, sf, structural
        )
        experiment_name = "EXP003"
        model_name = "exp003_tws_history_delta_lgbm"
        categorical_features = []
    elif args.feature_set == "hydro_gap":
        feature_columns = HYDRO_GAP_FEATURE_COLUMNS
        source_columns = SOURCE_HYDRO_HISTORY_COLUMNS
        feature_builder = lambda ledger, sf: build_hydro_gap_feature_matrix(
            ledger, sf, structural
        )
        experiment_name = "EXP005"
        model_name = "exp005_hydro_gap_delta_lgbm"
        categorical_features = []
    else:
        feature_columns = LOCATION_CAT_FEATURE_COLUMNS
        source_columns = SOURCE_HYDRO_HISTORY_COLUMNS
        feature_builder = lambda ledger, sf: build_location_cat_feature_matrix(
            ledger, sf, structural
        )
        experiment_name = "EXP006"
        model_name = "exp006_location_cat_delta_lgbm"
        categorical_features = ["location_id"]

    source_features = raw.loc[:, source_columns].copy()
    labels = raw.loc[:, ["sample_id", "target"]].copy()
    del raw

    print("[2/5] Defining development folds (lockbox excluded)...")
    fold_blocks = recent_observed_month_blocks(structural, block_size=18, n_blocks=4)

    params = dict(DEFAULT_PARAMS)
    params["seed"] = args.seed
    params["feature_fraction_seed"] = args.seed
    params["bagging_seed"] = args.seed
    params["data_random_seed"] = args.seed

    all_results = []
    print(f"[3/5] Building {experiment_name} sampled-h training sets and direct-h validation sets...")

    for fold_name in args.folds:
        started = time.perf_counter()
        months = fold_blocks[allowed_names[fold_name]]
        fold = build_direct_horizon_fold(
            structural.merge(labels, on="sample_id", validate="one_to_one"),
            months,
        )

        sampled = build_sampled_training_rows(
            structural,
            source_features.loc[:, SOURCE_CORE_COLUMNS],
            max_target_month=fold.max_training_target_month,
            seed=args.seed,
        )
        train_rows = sampled.rows
        X_train = feature_builder(train_rows, source_features)
        y_train_delta, _ = _attach_labels(train_rows, labels)
        train_weight = horizon_rebalance_weights(train_rows["h"])

        X_valid = feature_builder(fold.ledger, source_features)
        y_valid = _validation_labels(fold)
        y_valid_delta = y_valid - fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float32)
        valid_weight = validation_horizon_weights(fold.ledger["h"])

        persistence_pred = predict_persistence(fold.ledger)
        persistence_score, persistence_by_h = score_by_horizon(
            y_valid, persistence_pred, fold.ledger["h"]
        )

        prep_report = {
            "fold": fold_name,
            "train_target_cutoff": str(fold.max_training_target_month),
            "train_rows": int(len(train_rows)),
            "validation_rows": int(len(fold.ledger)),
            "dropped_training_missing_anchor": int(sampled.dropped_missing_anchor),
            "training_h_counts": sampled.horizon_counts,
            "feature_columns": feature_columns,
            "persistence_weighted_rmse": float(persistence_score),
        }
        print(f"\n=== {fold_name.upper()} PREP ===")
        print(json.dumps(prep_report, indent=2))

        if args.dry_run:
            print("DRY RUN: feature/label/weight construction passed; LightGBM not imported.")
            all_results.append({**prep_report, "dry_run": True})
            continue

        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM is required for training. Run this experiment on Kaggle, "
                "where LightGBM is expected to be available."
            ) from exc

        print("[4/5] Training pooled direct-residual LightGBM...")
        train_set = lgb.Dataset(
            X_train,
            label=y_train_delta,
            weight=train_weight,
            feature_name=feature_columns,
            categorical_feature=categorical_features,
            free_raw_data=True,
        )
        valid_set = lgb.Dataset(
            X_valid,
            label=y_valid_delta,
            weight=valid_weight,
            feature_name=feature_columns,
            categorical_feature=categorical_features,
            reference=train_set,
            free_raw_data=True,
        )
        model = lgb.train(
            params,
            train_set,
            num_boost_round=args.num_boost_round,
            valid_sets=[valid_set],
            valid_names=["dev"],
            callbacks=[
                lgb.early_stopping(args.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(100),
            ],
        )

        delta_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        pred = fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float64) + delta_pred
        score, by_h = score_by_horizon(y_valid, pred, fold.ledger["h"])
        improvement = persistence_score - score
        improvement_pct = improvement / persistence_score * 100.0

        importance = pd.DataFrame(
            {
                "feature": feature_columns,
                "gain": model.feature_importance(importance_type="gain"),
                "split": model.feature_importance(importance_type="split"),
            }
        ).sort_values("gain", ascending=False)

        result = {
            **prep_report,
            "model": model_name,
            "weighted_rmse": float(score),
            "absolute_gain_vs_persistence": float(improvement),
            "pct_gain_vs_persistence": float(improvement_pct),
            "best_iteration": int(model.best_iteration),
            "runtime_seconds": float(time.perf_counter() - started),
            "by_h": {
                int(h): {
                    "rmse": float(by_h.loc[h, "rmse"]),
                    "persistence_rmse": float(persistence_by_h.loc[h, "rmse"]),
                    "gain": float(persistence_by_h.loc[h, "rmse"] - by_h.loc[h, "rmse"]),
                }
                for h in range(1, 8)
            },
            "feature_importance_gain": {
                row.feature: float(row.gain) for row in importance.itertuples(index=False)
            },
        }
        all_results.append(result)

        print(f"\n=== {fold_name.upper()} RESULT ===")
        print(
            f"persistence={persistence_score:.6f}  "
            f"lgbm={score:.6f}  gain={improvement:+.6f} ({improvement_pct:+.2f}%)  "
            f"best_iter={model.best_iteration}"
        )
        print("By horizon:")
        print(by_h[["rows", "rmse", "bias"]].to_string(float_format=lambda x: f"{x:.6f}"))
        print("Feature importance (gain):")
        print(importance.to_string(index=False))

    print(f"[5/5] {experiment_name} summary...")
    print("\nJSON_RESULTS")
    print(json.dumps(all_results, indent=2))

    if not args.dry_run:
        scores = [r["weighted_rmse"] for r in all_results]
        gains = [r["absolute_gain_vs_persistence"] for r in all_results]
        print(
            f"\n{experiment_name} MEAN weighted_RMSE={np.mean(scores):.6f}  "
            f"mean_abs_gain_vs_persistence={np.mean(gains):+.6f}"
        )
        print(f"{experiment_name} COMPLETE: pooled target-blind delta LightGBM scored on dev folds only.")
    else:
        print(f"\n{experiment_name} DRY RUN PASSED: target-blind feature construction is ready for Kaggle training.")


if __name__ == "__main__":
    main()
