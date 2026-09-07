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
    HYDRO_GAP_SAFE_FEATURE_COLUMNS,
    SOURCE_CORE_COLUMNS,
    SOURCE_HYDRO_HISTORY_COLUMNS,
    build_hydro_gap_feature_matrix,
    build_hydro_gap_safe_feature_matrix,
    build_sampled_training_rows,
    horizon_rebalance_weights,
    validation_horizon_weights,
)
from src.validation import (
    build_exact_historical_mask_fold,
    build_test_mask_template,
    find_exact_template_starts,
)


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


def _build_replay_available_panels(
    train: pd.DataFrame,
    template: pd.DataFrame,
    start: pd.Period,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror full-Train-prefix + sparse-Test-window availability historically."""
    x = train.copy()
    x["source_period"] = pd.to_datetime(x["time"]).dt.to_period("M")

    prefix = x.loc[x["source_period"] < start].copy()

    replay_map = template.loc[:, [
        "lat", "lon", "offset_months", "template_visible"
    ]].copy()
    replay_map["source_period"] = replay_map["offset_months"].map(
        lambda o: start + int(o)
    )

    window = x.merge(
        replay_map.loc[:, ["lat", "lon", "source_period", "template_visible"]],
        how="inner",
        on=["lat", "lon", "source_period"],
        validate="one_to_one",
    )
    if window.empty:
        raise AssertionError("Replay availability window is empty")

    # Test supplies hydrometeorology for every Test row but hides TWS_t on masked
    # rows. Preserve all replay source rows for source-feature lookups and blank only
    # the TWS state where the transplanted Test mask says it is hidden.
    replay_structural = window.loc[
        :, ["sample_id", "time", "lat", "lon", "TWS_t", "template_visible"]
    ].copy()
    replay_structural.loc[~replay_structural["template_visible"], "TWS_t"] = np.nan
    replay_structural = replay_structural.drop(columns=["template_visible"])

    prefix_structural = prefix.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]]
    structural_available = pd.concat(
        [prefix_structural, replay_structural], ignore_index=True, sort=False
    )
    if structural_available.duplicated(["time", "lat", "lon"]).any():
        raise AssertionError("Replay structural availability has duplicate location-month rows")

    source_cols = SOURCE_HYDRO_HISTORY_COLUMNS
    prefix_source = prefix.loc[:, source_cols]
    replay_source = window.loc[:, source_cols]
    source_available = pd.concat(
        [prefix_source, replay_source], ignore_index=True, sort=False
    )
    if source_available["sample_id"].duplicated().any():
        raise AssertionError("Replay source-feature IDs are not unique")

    return structural_available, source_available


def _history_availability_from_features(features: pd.DataFrame) -> dict[str, float]:
    lag_cols = [
        "TWS_anchor_lag1",
        "TWS_anchor_lag2",
        "TWS_anchor_lag3",
        "TWS_anchor_lag6",
        "TWS_anchor_lag12",
    ]
    available = features.loc[:, lag_cols].notna()
    result = {
        col.replace("TWS_anchor_", ""): float(available[col].mean())
        for col in lag_cols
    }
    result["all_lags"] = float(available.all(axis=1).mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Kaggle-only scoring diagnostic: compare EXP005 with EXP009 on the latest "
            "exact historical Test replay while reproducing sparse Test TWS-history availability."
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
        raise RuntimeError("LightGBM is required; run this diagnostic on Kaggle.") from exc

    print("[1/7] Reading Train/Test...")
    usecols = [
        "sample_id", "time", "lat", "lon", "TWS_t",
        "month_sin", "month_cos",
        "SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t",
        "SOIL_MOISTURE_t", "target",
    ]
    train = pd.read_csv(args.data_dir / "Train.csv", usecols=usecols)
    test = pd.read_csv(
        args.data_dir / "Test.csv",
        usecols=["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"],
    )

    print("[2/7] Building latest exact historical Test replay...")
    template = build_test_mask_template(test)
    starts = find_exact_template_starts(train, template)
    if not starts:
        raise AssertionError("No exact historical Test replay is available")
    start = starts[-1]
    fold = build_exact_historical_mask_fold(train, template, start)
    print("Replay:", fold.start_month, "->", fold.end_month)

    print("[3/7] Reconstructing inference-faithful sparse state availability...")
    structural_available, source_available = _build_replay_available_panels(
        train, template, start
    )
    # Training must stop before the replay's first target future. The source rows
    # therefore end before the replay start, matching the real full-Train/Test split.
    train_structural = train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
    train_source = train.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
    labels = train.loc[:, ["sample_id", "target"]].copy()
    sampled = build_sampled_training_rows(
        train_structural,
        train_source.loc[:, SOURCE_CORE_COLUMNS],
        max_target_month=start,
        seed=args.seed,
    )
    training_rows = sampled.rows
    y_train = _attach_delta_labels(training_rows, labels)
    train_weight = horizon_rebalance_weights(training_rows["h"])

    validation_labels = fold.labels.set_index("sample_id").loc[
        fold.ledger["sample_id"]
    ]["target"].to_numpy(dtype=np.float32)
    valid_weight = validation_horizon_weights(fold.ledger["h"])
    persistence = predict_persistence(fold.ledger)
    persistence_score, persistence_by_h = score_by_horizon(
        validation_labels, persistence, fold.ledger["h"]
    )

    print("[4/7] Building EXP005/EXP009 training matrices...")
    X_train_005 = build_hydro_gap_feature_matrix(
        training_rows, train_source, train_structural
    )
    X_train_009 = build_hydro_gap_safe_feature_matrix(
        training_rows, train_source, train_structural
    )

    print("[5/7] Building inference-faithful replay validation matrices...")
    X_valid_005 = build_hydro_gap_feature_matrix(
        fold.ledger, source_available, structural_available
    )
    X_valid_009 = build_hydro_gap_safe_feature_matrix(
        fold.ledger, source_available, structural_available
    )
    availability = _history_availability_from_features(X_valid_005)
    print("Replay TWS-history availability:")
    print(json.dumps(availability, indent=2))
    y_valid_delta = (
        validation_labels
        - fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float32)
    )

    params = dict(DEFAULT_PARAMS)
    params["seed"] = args.seed
    params["feature_fraction_seed"] = args.seed
    params["bagging_seed"] = args.seed
    params["data_random_seed"] = args.seed

    results = []
    print("[6/7] Training and scoring both formulations...")
    for name, feature_names, X_train, X_valid in [
        ("EXP005", HYDRO_GAP_FEATURE_COLUMNS, X_train_005, X_valid_005),
        ("EXP009", HYDRO_GAP_SAFE_FEATURE_COLUMNS, X_train_009, X_valid_009),
    ]:
        started = time.perf_counter()
        train_set = lgb.Dataset(
            X_train,
            label=y_train,
            weight=train_weight,
            feature_name=feature_names,
            free_raw_data=True,
        )
        valid_set = lgb.Dataset(
            X_valid,
            label=y_valid_delta,
            weight=valid_weight,
            feature_name=feature_names,
            reference=train_set,
            free_raw_data=True,
        )
        model = lgb.train(
            params,
            train_set,
            num_boost_round=args.num_boost_round,
            valid_sets=[valid_set],
            valid_names=[name.lower()],
            callbacks=[
                lgb.early_stopping(args.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(100),
            ],
        )
        pred = (
            fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float64)
            + model.predict(X_valid, num_iteration=model.best_iteration)
        )
        score, by_h = score_by_horizon(validation_labels, pred, fold.ledger["h"])
        result = {
            "model": name,
            "weighted_rmse": float(score),
            "best_iteration": int(model.best_iteration),
            "runtime_seconds": float(time.perf_counter() - started),
            "by_h": {
                int(h): {
                    "rmse": float(by_h.loc[h, "rmse"]),
                    "persistence_rmse": float(persistence_by_h.loc[h, "rmse"]),
                }
                for h in range(1, 8)
            },
        }
        results.append(result)
        print(
            f"{name}: weighted_RMSE={score:.6f} best_iter={model.best_iteration} "
            f"gain_vs_persistence={persistence_score - score:+.6f}"
        )

    print("[7/7] Availability-faithful replay summary...")
    print("Persistence:", f"{persistence_score:.6f}")
    print("JSON_RESULTS")
    print(json.dumps({
        "replay_start": str(fold.start_month),
        "replay_end": str(fold.end_month),
        "validation_rows": int(len(fold.ledger)),
        "history_availability": availability,
        "persistence_rmse": float(persistence_score),
        "models": results,
    }, indent=2))
    print("\nAVAILABILITY-FAITHFUL REPLAY COMPLETE")


if __name__ == "__main__":
    main()
