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


FROZEN_ROUNDS = 118
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)


def _attach_delta_labels(rows: pd.DataFrame, labels: pd.DataFrame) -> np.ndarray:
    joined = rows.loc[:, ["sample_id", "last_observed_TWS"]].merge(
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


def _optimal_alpha(
    y: np.ndarray,
    persistence: np.ndarray,
    model: np.ndarray,
    weights: np.ndarray,
) -> float:
    direction = model - persistence
    target_offset = y - persistence
    denom = float(np.sum(weights * direction * direction))
    if denom <= 0:
        return 0.0
    return float(np.sum(weights * target_offset * direction) / denom)


def _score_alpha(
    alpha: float,
    y: np.ndarray,
    persistence: np.ndarray,
    model: np.ndarray,
    h: pd.Series,
) -> float:
    pred = persistence + alpha * (model - persistence)
    score, _ = score_by_horizon(y, pred, h)
    return float(score)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Kaggle-only diagnostic: test whether frozen EXP005 corrections should "
            "be shrunk toward legal TWS persistence on dev3 and the already-open lockbox."
        )
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM is required; run this audit on Kaggle.") from exc

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
    source_features = raw.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
    labels = raw.loc[:, ["sample_id", "target"]].copy()
    del raw

    blocks = recent_observed_month_blocks(structural, block_size=18, n_blocks=4)
    fold_specs = [("dev3", blocks[2]), ("lockbox", blocks[3])]

    params = dict(DEFAULT_PARAMS)
    params["seed"] = args.seed
    params["feature_fraction_seed"] = args.seed
    params["bagging_seed"] = args.seed
    params["data_random_seed"] = args.seed

    all_results = []
    print("[2/5] Building fixed-118-round EXP005 diagnostics...")

    for fold_name, months in fold_specs:
        started = time.perf_counter()
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

        X_train = build_hydro_gap_feature_matrix(
            sampled.rows,
            source_features,
            structural,
        )
        y_train_delta = _attach_delta_labels(sampled.rows, labels)
        train_weight = horizon_rebalance_weights(sampled.rows["h"])

        X_valid = build_hydro_gap_feature_matrix(
            fold.ledger,
            source_features,
            structural,
        )
        y_valid = fold.labels["target"].to_numpy(dtype=np.float64)
        valid_weight = validation_horizon_weights(fold.ledger["h"]).astype(np.float64)
        persistence = predict_persistence(fold.ledger).astype(np.float64)

        print(
            f"\n[3/5] Training frozen EXP005 r{FROZEN_ROUNDS} for {fold_name} "
            f"({fold.source_start} -> {fold.source_end})..."
        )
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
            num_boost_round=FROZEN_ROUNDS,
            callbacks=[lgb.log_evaluation(0)],
        )
        delta_pred = model.predict(X_valid, num_iteration=FROZEN_ROUNDS)
        model_pred = (
            fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float64)
            + delta_pred
        )

        baseline_score, _ = score_by_horizon(y_valid, persistence, fold.ledger["h"])
        model_score, _ = score_by_horizon(y_valid, model_pred, fold.ledger["h"])
        alpha_star = _optimal_alpha(y_valid, persistence, model_pred, valid_weight)

        grid = {
            str(alpha): _score_alpha(alpha, y_valid, persistence, model_pred, fold.ledger["h"])
            for alpha in ALPHA_GRID
        }

        by_h = {}
        for h in range(1, 8):
            mask = fold.ledger["h"].to_numpy(dtype=np.int8) == h
            if not mask.any():
                continue
            h_weights = np.ones(int(mask.sum()), dtype=np.float64)
            h_alpha = _optimal_alpha(
                y_valid[mask],
                persistence[mask],
                model_pred[mask],
                h_weights,
            )
            h_alpha_clip = float(np.clip(h_alpha, 0.0, 1.25))
            rmse_model = float(np.sqrt(np.mean((y_valid[mask] - model_pred[mask]) ** 2)))
            rmse_persist = float(np.sqrt(np.mean((y_valid[mask] - persistence[mask]) ** 2)))
            rmse_shrunk = float(
                np.sqrt(
                    np.mean(
                        (
                            y_valid[mask]
                            - (
                                persistence[mask]
                                + h_alpha_clip * (model_pred[mask] - persistence[mask])
                            )
                        )
                        ** 2
                    )
                )
            )
            by_h[h] = {
                "rows": int(mask.sum()),
                "alpha_unclipped": float(h_alpha),
                "alpha_clipped": h_alpha_clip,
                "rmse_persistence": rmse_persist,
                "rmse_model": rmse_model,
                "rmse_optimal_shrink": rmse_shrunk,
            }

        result = {
            "fold": fold_name,
            "source_start": str(fold.source_start),
            "source_end": str(fold.source_end),
            "fixed_rounds": FROZEN_ROUNDS,
            "persistence_rmse": float(baseline_score),
            "exp005_r118_rmse": float(model_score),
            "optimal_global_alpha_unclipped": float(alpha_star),
            "optimal_global_alpha_clipped_0_1_25": float(np.clip(alpha_star, 0.0, 1.25)),
            "alpha_grid_rmse": grid,
            "by_h": by_h,
            "runtime_seconds": float(time.perf_counter() - started),
        }
        all_results.append(result)

        print(f"\n=== {fold_name.upper()} SHRINKAGE RESULT ===")
        print(json.dumps(result, indent=2))

    print("\n[4/5] Cross-fold alpha comparison...")
    summary = pd.DataFrame(
        [
            {
                "fold": r["fold"],
                "persistence": r["persistence_rmse"],
                "exp005_r118": r["exp005_r118_rmse"],
                "alpha_star": r["optimal_global_alpha_unclipped"],
            }
            for r in all_results
        ]
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n[5/5] SHRINKAGE AUDIT COMPLETE")
    print(
        "Interpretation: alpha=1 reproduces EXP005; alpha=0 is legal persistence. "
        "A stable alpha<1 across dev3 and lockbox is evidence that the submitted "
        "correction should be shrunk before using another leaderboard slot."
    )


if __name__ == "__main__":
    main()
