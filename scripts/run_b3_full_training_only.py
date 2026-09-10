"""Fit the frozen dense-history B3/98 model from Train only.

This entrypoint is deliberately separate from ``run_b3_full_submission.py``:
it never opens Test.csv or SampleSubmission.csv and therefore cannot create a
competition prediction or submission.  It is intended for remote diagnostic
artifacts on the persistent CPU VM.
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
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_lgbm_core import DEFAULT_PARAMS
from src.ml_features import (
    SOURCE_CORE_COLUMNS,
    SOURCE_HYDRO_HISTORY_COLUMNS,
    build_sampled_training_rows,
    horizon_rebalance_weights,
)
from src.neural_sequence import build_b3_feature_maps, build_b3_matrix
from src.regional_context import HYDRO_COLUMNS


SEED = 20260908
ROUNDS = 98
NUM_LEAVES = 63
MIN_DATA_IN_LEAF = 1000
DEFAULT_NUM_THREADS = 12
MAX_NUM_THREADS = 24
TRAIN_COLUMNS = [
    "sample_id", "time", "lat", "lon", "TWS_t", "month_sin", "month_cos",
    *HYDRO_COLUMNS, "target",
]


def _json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_text(values: pd.Series | list[str]) -> str:
    text = values.astype(str).str.cat(sep="\n") if isinstance(values, pd.Series) else "\n".join(map(str, values))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _memory() -> dict[str, float | None]:
    try:
        import psutil

        process = psutil.Process()
        vm = psutil.virtual_memory()
        return {"rss_gib": process.memory_info().rss / 1024**3, "available_gib": vm.available / 1024**3}
    except Exception:
        return {"rss_gib": None, "available_gib": None}


def _namespace(prefix: str, values: pd.Series) -> pd.Series:
    return prefix + values.astype(str)


def _attach_delta(rows: pd.DataFrame, labels: pd.DataFrame) -> np.ndarray:
    joined = rows.loc[:, ["sample_id", "last_observed_TWS"]].merge(
        labels.loc[:, ["sample_id", "target"]], how="left", on="sample_id", validate="one_to_one", sort=False,
    )
    if joined["target"].isna().any():
        raise AssertionError("A sampled training row has no supplied label")
    return joined["target"].to_numpy(dtype=np.float32) - joined["last_observed_TWS"].to_numpy(dtype=np.float32)


def _assert_float32_or_missing(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise AssertionError("B3 training feature matrix is empty")
    if any(dtype != np.dtype("float32") for dtype in frame.dtypes):
        raise AssertionError("B3 training feature matrix has non-float32 columns")
    if np.isinf(frame.to_numpy(dtype=np.float32)).any():
        raise AssertionError("B3 training feature matrix contains infinite values")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help=f"LightGBM threads (default {DEFAULT_NUM_THREADS}; bounded to {MAX_NUM_THREADS} for safe VM benchmarks)",
    )
    args = parser.parse_args()
    if args.seed != SEED or args.rounds != ROUNDS:
        raise ValueError(f"Frozen B3/98 recipe requires seed={SEED} and rounds={ROUNDS}")
    available_cpus = os.cpu_count() or 1
    if args.num_threads < 1 or args.num_threads > MAX_NUM_THREADS:
        raise ValueError(f"--num-threads must be in [1, {MAX_NUM_THREADS}]")
    num_threads = min(int(args.num_threads), available_cpus)

    import lightgbm as lgb

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit:
        raise RuntimeError({"expected_commit": args.expected_commit, "actual_commit": head})
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Refusing to run from a dirty repository checkout")

    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "status": "running",
        "stage": "FULL_B3_DENSE_TRAINING_ONLY",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit": head,
        "branch": branch,
        "seed": SEED,
        "rounds": ROUNDS,
        "num_leaves": NUM_LEAVES,
        "min_data_in_leaf": MIN_DATA_IN_LEAF,
        "requested_num_threads": int(args.num_threads),
        "num_threads": num_threads,
        "no_test_predictions": True,
        "test_rows_read": 0,
        "test_labels_read": False,
        "fit_count": 0,
    }
    _json(output_dir / "manifest.json", manifest)
    try:
        train_path = data_dir / "Train.csv"
        if not train_path.is_file():
            raise FileNotFoundError(train_path)
        train = pd.read_csv(train_path, usecols=TRAIN_COLUMNS)
        if train["sample_id"].duplicated().any() or train[["lat", "lon", "time"]].duplicated().any():
            raise AssertionError("Train IDs and location-month keys must be unique")
        if train["target"].isna().any():
            raise AssertionError("Every Train target is required")
        periods = pd.to_datetime(train["time"]).dt.to_period("M")
        max_training_target_month = (periods + 1).max()
        structural = train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
        source = train.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
        labels = train.loc[:, ["sample_id", "target"]].copy()
        sampled = build_sampled_training_rows(
            structural,
            source.loc[:, SOURCE_CORE_COLUMNS],
            max_target_month=max_training_target_month,
            seed=SEED,
        )
        rows_plain = sampled.rows
        y_train = _attach_delta(rows_plain, labels)
        weights = np.asarray(horizon_rebalance_weights(rows_plain["h"]), dtype=np.float32)
        rows = rows_plain.copy()
        rows["sample_id"] = _namespace("tr__", rows["sample_id"])
        structural["sample_id"] = _namespace("tr__", structural["sample_id"])
        source["sample_id"] = _namespace("tr__", source["sample_id"])
        labels["sample_id"] = _namespace("tr__", labels["sample_id"])
        training_info = {
            "rows": int(len(rows)),
            "source_rows": int(len(train)),
            "horizon_counts": {str(int(k)): int(v) for k, v in rows_plain["h"].value_counts().sort_index().items()},
            "dropped_missing_anchor": int(sampled.dropped_missing_anchor),
            "max_training_target_month": str(max_training_target_month),
            "sample_seed": SEED,
            "row_id_hash": _hash_text(rows_plain["sample_id"]),
            "label_hash": _hash_array(y_train),
            "weight_hash": _hash_array(weights),
        }
        manifest["training"] = training_info
        manifest["data"] = {"train_rows": int(len(train)), "train_sha256": _sha256(train_path), "python": sys.version,
                             "platform": platform.platform(), "cpu_count": os.cpu_count(), "memory": _memory()}
        _json(output_dir / "manifest.json", manifest)

        maps = build_b3_feature_maps(source)
        x_train = build_b3_matrix(rows, source, structural, maps)
        _assert_float32_or_missing(x_train)
        feature_names = x_train.columns.tolist()
        if len(feature_names) != 446:
            raise AssertionError(f"Frozen B3 schema must contain 446 features, found {len(feature_names)}")
        feature_schema = {"feature_count": len(feature_names), "feature_names": feature_names, "dtype": "float32",
                          "missing_values_allowed": True, "schema_sha256": _hash_text(feature_names)}
        _json(output_dir / "feature_schema.json", feature_schema)
        params = dict(DEFAULT_PARAMS, num_leaves=NUM_LEAVES, min_data_in_leaf=MIN_DATA_IN_LEAF,
                      num_threads=num_threads, seed=SEED, feature_fraction_seed=SEED,
                      bagging_seed=SEED, data_random_seed=SEED)
        _json(output_dir / "resolved_config.json", {"model": "B3_dense_history_delta_lightgbm_training_only",
              "params": params, "feature_schema": feature_schema, "training": training_info,
              "no_test_predictions": True, "test_rows_read": 0})
        train_set = lgb.Dataset(x_train, label=y_train, weight=weights, feature_name=feature_names, free_raw_data=True)
        fit_started = time.perf_counter()
        booster = lgb.train(params, train_set, num_boost_round=ROUNDS, callbacks=[lgb.log_evaluation(25)])
        model_path = output_dir / "model.txt"
        booster.save_model(str(model_path))
        manifest.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
                         "elapsed_seconds": time.perf_counter() - started, "fit_seconds": time.perf_counter() - fit_started,
                         "fit_count": 1, "feature_schema": feature_schema,
                         "model": {"path": str(model_path), "bytes": int(model_path.stat().st_size), "sha256": _sha256(model_path)},
                         "output_checksums": {"model.txt": _sha256(model_path),
                                              "feature_schema.json": _sha256(output_dir / "feature_schema.json"),
                                              "resolved_config.json": _sha256(output_dir / "resolved_config.json")}})
        _json(output_dir / "manifest.json", manifest)
        package = Path(shutil.make_archive(str(output_dir), "zip", root_dir=output_dir))
        manifest["package"] = {"path": str(package), "bytes": int(package.stat().st_size), "sha256": _sha256(package)}
        _json(output_dir / "manifest.json", manifest)
        print(json.dumps(manifest, sort_keys=True, default=str), flush=True)
    except Exception as exc:
        manifest.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(),
                         "elapsed_seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}",
                         "memory": _memory()})
        _json(output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
