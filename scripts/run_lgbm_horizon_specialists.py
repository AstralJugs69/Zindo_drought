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
)
from src.validation import build_direct_horizon_fold, recent_observed_month_blocks


def _attach_delta_labels(
    feature_rows: pd.DataFrame,
    labels: pd.DataFrame,
) -> np.ndarray:
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
    return (
        joined["target"].to_numpy(dtype=np.float32)
        - joined["last_observed_TWS"].to_numpy(dtype=np.float32)
    )


def _validation_labels(fold) -> np.ndarray:
    labels = fold.labels.loc[:, ["example_id", "target"]]
    if labels["example_id"].tolist() != fold.ledger["example_id"].tolist():
        raise AssertionError("Validation labels are not aligned to ledger")
    return labels["target"].to_numpy(dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--folds", nargs="+", default=["dev3"])
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--num-boost-round", type=int, default=1600)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    args = parser.parse_args()

    allowed_names = {"dev1": 0, "dev2": 1, "dev3": 2}
    unknown = [name for name in args.folds if name not in allowed_names]
    if unknown:
        raise ValueError(
            "EXP008 supports dev1-dev3 only; the lockbox must not be used for tuning. "
            f"Unknown folds: {unknown}"
        )

    train_path = args.data_dir / "Train.csv"
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is required. Run EXP008 on Kaggle rather than the local PC."
        ) from exc

    print("[1/5] Reading target-blind state/features and isolated labels...")
    usecols = [
        "sample_id",
        "time",
        "lat",
        "lon",
        "TWS_t",
        "month_sin",
        "month_cos",
        "SPEI_01_t",
        "SPEI_03_t",
        "SPEI_06_t",
        "SPEI_12_t",
        "SOIL_MOISTURE_t",
        "target",
    ]
    raw = pd.read_csv(train_path, usecols=usecols)
    structural = raw.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
    source_features = raw.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
    labels = raw.loc[:, ["sample_id", "target"]].copy()
    del raw

    print("[2/5] Defining development folds (lockbox excluded)...")
    fold_blocks = recent_observed_month_blocks(structural, block_size=18, n_blocks=4)

    params = dict(DEFAULT_PARAMS)
    params["seed"] = args.seed
    params["feature_fraction_seed"] = args.seed
    params["bagging_seed"] = args.seed
    params["data_random_seed"] = args.seed

    all_results: list[dict] = []

    for fold_name in args.folds:
        started = time.perf_counter()
        months = fold_blocks[allowed_names[fold_name]]
        fold = build_direct_horizon_fold(
            structural.merge(labels, on="sample_id", validate="one_to_one"),
            months,
        )

        print(f"\n[3/5] Building frozen EXP005 matrices for {fold_name}...")
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

        X_valid = build_hydro_gap_feature_matrix(
            fold.ledger,
            source_features,
            structural,
        )
        y_valid = _validation_labels(fold)
        y_valid_delta = (
            y_valid
            - fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float32)
        )

        persistence_pred = predict_persistence(fold.ledger)
        persistence_score, persistence_by_h = score_by_horizon(
            y_valid,
            persistence_pred,
            fold.ledger["h"],
        )

        print(
            json.dumps(
                {
                    "fold": fold_name,
                    "train_target_cutoff": str(fold.max_training_target_month),
                    "train_rows": int(len(sampled.rows)),
                    "validation_rows": int(len(fold.ledger)),
                    "training_h_counts": sampled.horizon_counts,
                    "feature_columns": HYDRO_GAP_FEATURE_COLUMNS,
                    "persistence_weighted_rmse": float(persistence_score),
                },
                indent=2,
            )
        )

        print("[4/5] Training seven horizon-specialist LightGBMs...")
        delta_pred = np.full(len(fold.ledger), np.nan, dtype=np.float64)
        horizon_results: dict[int, dict] = {}

        train_h = sampled.rows["h"].to_numpy(dtype=np.int8)
        valid_h = fold.ledger["h"].to_numpy(dtype=np.int8)

        for horizon in range(1, 8):
            train_mask = train_h == horizon
            valid_mask = valid_h == horizon
            n_train = int(train_mask.sum())
            n_valid = int(valid_mask.sum())
            if n_train == 0 or n_valid == 0:
                raise AssertionError(
                    f"EXP008 requires non-empty train/validation rows for h={horizon}"
                )

            train_set = lgb.Dataset(
                X_train.loc[train_mask, HYDRO_GAP_FEATURE_COLUMNS],
                label=y_train_delta[train_mask],
                feature_name=HYDRO_GAP_FEATURE_COLUMNS,
                free_raw_data=True,
            )
            valid_set = lgb.Dataset(
                X_valid.loc[valid_mask, HYDRO_GAP_FEATURE_COLUMNS],
                label=y_valid_delta[valid_mask],
                feature_name=HYDRO_GAP_FEATURE_COLUMNS,
                reference=train_set,
                free_raw_data=True,
            )

            model = lgb.train(
                params,
                train_set,
                num_boost_round=args.num_boost_round,
                valid_sets=[valid_set],
                valid_names=[f"h{horizon}"],
                callbacks=[
                    lgb.early_stopping(args.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(100),
                ],
            )

            delta_pred[valid_mask] = model.predict(
                X_valid.loc[valid_mask, HYDRO_GAP_FEATURE_COLUMNS],
                num_iteration=model.best_iteration,
            )

            importance = pd.DataFrame(
                {
                    "feature": HYDRO_GAP_FEATURE_COLUMNS,
                    "gain": model.feature_importance(importance_type="gain"),
                }
            ).sort_values("gain", ascending=False)

            horizon_results[horizon] = {
                "train_rows": n_train,
                "validation_rows": n_valid,
                "best_iteration": int(model.best_iteration),
                "top_features": [
                    {
                        "feature": row.feature,
                        "gain": float(row.gain),
                    }
                    for row in importance.head(8).itertuples(index=False)
                ],
            }
            print(
                f"h={horizon}: train={n_train:,} valid={n_valid:,} "
                f"best_iter={model.best_iteration}"
            )

        if np.isnan(delta_pred).any():
            raise AssertionError("EXP008 failed to predict one or more validation rows")

        print("[5/5] Scoring combined horizon-specialist predictions...")
        pred = (
            fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float64)
            + delta_pred
        )
        score, by_h = score_by_horizon(y_valid, pred, fold.ledger["h"])
        improvement = persistence_score - score
        improvement_pct = improvement / persistence_score * 100.0

        for horizon in range(1, 8):
            horizon_results[horizon].update(
                {
                    "rmse": float(by_h.loc[horizon, "rmse"]),
                    "bias": float(by_h.loc[horizon, "bias"]),
                    "persistence_rmse": float(
                        persistence_by_h.loc[horizon, "rmse"]
                    ),
                }
            )

        result = {
            "fold": fold_name,
            "model": "exp008_horizon_specialist_hydro_gap_lgbm",
            "weighted_rmse": float(score),
            "persistence_weighted_rmse": float(persistence_score),
            "absolute_gain_vs_persistence": float(improvement),
            "pct_gain_vs_persistence": float(improvement_pct),
            "runtime_seconds": float(time.perf_counter() - started),
            "by_h": horizon_results,
        }
        all_results.append(result)

        print(f"\n=== {fold_name.upper()} EXP008 RESULT ===")
        print(
            f"persistence={persistence_score:.6f}  "
            f"specialists={score:.6f}  "
            f"gain={improvement:+.6f} ({improvement_pct:+.2f}%)"
        )
        print("By horizon:")
        print(by_h[["rows", "rmse", "bias"]].to_string(float_format=lambda x: f"{x:.6f}"))

    print("\nJSON_RESULTS")
    print(json.dumps(all_results, indent=2))
    print(
        f"\nEXP008 MEAN weighted_RMSE="
        f"{np.mean([r['weighted_rmse'] for r in all_results]):.6f}"
    )


if __name__ == "__main__":
    main()
