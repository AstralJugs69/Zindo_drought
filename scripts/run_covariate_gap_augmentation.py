"""One recoverable Kaggle fit for D0/D1/D2 covariate-gap augmentation.

No competition Test predictions are made.  Test is read only to derive its
unlabelled source-date schedule; training labels remain outside augmentation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiment import _attach_delta
from scripts.run_hydrological_trajectory import INNER_ORIGINS, OUTER_ORIGINS, _fold
from scripts.run_lgbm_core import DEFAULT_PARAMS
from src.availability import build_replay_observation_view
from src.covariate_gap_augmentation import build_augmented_b3_matrix, derive_test_schedule_patterns
from src.metrics import raw_rmse, score_by_horizon
from src.ml_features import SOURCE_CORE_COLUMNS, SOURCE_HYDRO_HISTORY_COLUMNS, build_sampled_training_rows, horizon_rebalance_weights
from src.neural_sequence import build_b3_feature_maps, build_b3_matrix
from src.regional_context import HYDRO_COLUMNS
from src.validation import build_test_mask_template

RECIPES = ("D0_dense", "D1_sparse", "D2_mixed")
SEEDS = (20260909, 20260910)
ALL_ORIGINS = (*INNER_ORIGINS, *OUTER_ORIGINS)
ROUNDS = 98


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_values(values: pd.Series | np.ndarray) -> str:
    if isinstance(values, pd.Series):
        data = values.astype(str).str.cat(sep="\n").encode("utf-8")
    else:
        data = np.ascontiguousarray(values).tobytes()
    return hashlib.sha256(data).hexdigest()


def _score(ledger: pd.DataFrame, labels: pd.DataFrame, prediction: np.ndarray) -> tuple[dict[str, object], pd.DataFrame]:
    target = labels.set_index("sample_id").loc[ledger.sample_id, "target"].to_numpy(np.float64)
    result = ledger.loc[:, ["sample_id", "source_date", "last_observed_date", "last_observed_TWS", "h", "lat", "lon"]].copy()
    result["target"] = target; result["prediction"] = prediction; result["residual"] = prediction - target
    supported = result.loc[result.h.between(1, 7)].copy()
    present = sorted(int(value) for value in supported.h.unique())
    weighted = None; by_h: list[dict[str, object]] = []
    if present == list(range(1, 8)):
        weighted, horizon = score_by_horizon(supported.target, supported.prediction, supported.h)
        by_h = horizon.reset_index().to_dict(orient="records")
    elif len(supported):
        by_h = [{"h": int(h), "rows": int(len(group)), "rmse": raw_rmse(group.target, group.prediction)} for h, group in supported.groupby("h", sort=True)]
    tail = result.loc[~result.h.between(1, 7)]
    source = pd.to_datetime(result.source_date).dt.to_period("M").astype("int64")
    anchor = pd.to_datetime(result.last_observed_date).dt.to_period("M").astype("int64")
    result["anchor_age_months"] = (source - anchor).astype(np.int16)
    result["geo5"] = (np.floor(result.lat / 5).astype(int).astype(str) + ":" + np.floor(result.lon / 5).astype(int).astype(str))
    result["calendar_block"] = pd.to_datetime(result.source_date).dt.to_period("M").astype(str)
    age_rows = []
    for bucket, group in result.groupby(pd.cut(result.anchor_age_months, [-1, 0, 1, 2, np.inf], labels=["0", "1", "2", "3+"]), observed=True):
        age_rows.append({"anchor_age": str(bucket), "rows": int(len(group)), "raw_rmse": raw_rmse(group.target, group.prediction)})
    return {
        "rows": int(len(result)), "raw_rmse": raw_rmse(target, prediction),
        "official_h1_7_rows": int(len(supported)), "official_horizons_present": present,
        "official_h1_7_weighted_rmse": weighted, "by_h": by_h,
        "stress_h_gt7_rows": int(len(tail)), "stress_h_gt7_raw_rmse": None if tail.empty else raw_rmse(tail.target, tail.prediction),
        "by_anchor_age": age_rows,
    }, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--origin", choices=ALL_ORIGINS, required=True)
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--augmentation-seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.recipe == "D0_dense" and args.augmentation_seed is not None:
        raise ValueError("D0_dense has no augmentation seed")
    if args.recipe != "D0_dense" and args.augmentation_seed is None:
        raise ValueError("D1/D2 require an augmentation seed")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit:
        raise RuntimeError({"expected_commit": args.expected_commit, "actual_commit": head})
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("refusing to run from dirty checkout")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest: dict[str, object] = {"status": "running", "commit": head, "origin": args.origin, "recipe": args.recipe,
        "augmentation_seed": args.augmentation_seed, "rounds": ROUNDS, "no_test_predictions": True,
        "started_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform()}
    _json(args.output_dir / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        import lightgbm as lgb
        train_columns = ["sample_id", "time", "lat", "lon", "TWS_t", "target", "month_sin", "month_cos", *HYDRO_COLUMNS]
        test_columns = ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]
        train = pd.read_csv(args.data_dir / "Train.csv", usecols=train_columns)
        test = pd.read_csv(args.data_dir / "Test.csv", usecols=test_columns)
        template = build_test_mask_template(test)
        patterns = derive_test_schedule_patterns(test)
        structural = train[["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
        source = train.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
        labels = train[["sample_id", "target"]].copy()
        fold, role = _fold(train, template, args.origin)
        view = build_replay_observation_view(source, structural, ledger=fold.ledger, first_source_month=fold.spec.first_source_month, last_source_month=fold.spec.last_source_month)
        rows = build_sampled_training_rows(view.structural, view.source.loc[:, SOURCE_CORE_COLUMNS], max_target_month=pd.Period(args.origin, freq="M") - 1, seed=20260908).rows
        y_train = _attach_delta(rows, labels).astype(np.float32)
        weights = np.asarray(horizon_rebalance_weights(rows.h), dtype=np.float32)
        maps = build_b3_feature_maps(view.source)
        valid_x = build_b3_matrix(fold.ledger, view.source, view.structural, maps)
        if args.recipe == "D0_dense":
            train_x = build_b3_matrix(rows, view.source, view.structural, maps)
            augmentation = {"recipe": args.recipe, "augmentation_seed": None, "pattern_fingerprint": patterns.fingerprint, "sparse_rows": 0, "dense_rows": int(len(rows))}
        else:
            train_x, augmentation = build_augmented_b3_matrix(rows, view.source, view.structural, maps, patterns, recipe=args.recipe, seed=args.augmentation_seed)
        if train_x.columns.tolist() != valid_x.columns.tolist():
            raise AssertionError("training/validation B3 feature schemas differ")
        params = dict(DEFAULT_PARAMS, num_leaves=63, min_data_in_leaf=1000, num_threads=min(4, os.cpu_count() or 1),
                      seed=20260908, feature_fraction_seed=20260908, bagging_seed=20260908, data_random_seed=20260908)
        model = lgb.train(params, lgb.Dataset(train_x, label=y_train, weight=weights, feature_name=list(train_x.columns)), num_boost_round=ROUNDS, callbacks=[lgb.log_evaluation(0)])
        prediction = fold.ledger.last_observed_TWS.to_numpy(np.float64) + model.predict(valid_x)
        metrics, oof = _score(fold.ledger, labels, prediction)
        model.save_model(str(args.output_dir / "model.txt")); oof.to_csv(args.output_dir / "oof.csv.gz", index=False, compression="gzip")
        manifest.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started,
            "role": role, "augmentation": augmentation, "training_rows": int(len(rows)), "validation_rows": int(len(fold.ledger)),
            "training_ids_hash": _hash_values(rows.sample_id), "training_labels_hash": _hash_values(y_train), "training_weights_hash": _hash_values(weights),
            "validation_ids_hash": _hash_values(fold.ledger.sample_id), "feature_names": train_x.columns.tolist(), "feature_count": int(train_x.shape[1]), "metrics": metrics,
            "checksums": {"model.txt": _sha256(args.output_dir / "model.txt"), "oof.csv.gz": _sha256(args.output_dir / "oof.csv.gz")}})
        _json(args.output_dir / "manifest.json", manifest)
        package = Path(shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir))
        manifest["package"] = {"path": str(package), "bytes": package.stat().st_size, "sha256": _sha256(package)}
        _json(args.output_dir / "manifest.json", manifest)
        print(json.dumps(manifest, sort_keys=True, default=str))
    except Exception as exc:
        manifest.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}"})
        _json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
