"""Stage C controls: flattened hydrology history and prefix-normalized ridge.

Kaggle-only.  This runner deliberately stops before any neural sequence model.
It compares the frozen B3/98 tree control with two direct-history controls on
the already-selected 2003-04/2004-04 inner replays.  It never predicts Test
targets or writes a competition submission.
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

from scripts.run_experiment import _attach_delta, _score
from scripts.run_hydrological_trajectory import (
    INNER_ORIGINS,
    MemoryTracker,
    _feature_frame,
    _fold,
)
from scripts.run_lgbm_core import DEFAULT_NUM_THREADS, DEFAULT_PARAMS
from src.hydro_sequence import PrefixNormalizer, build_hydro_sequence_features
from src.ml_features import (
    SOURCE_CORE_COLUMNS,
    SOURCE_HYDRO_HISTORY_COLUMNS,
    build_sampled_training_rows,
    horizon_rebalance_weights,
)
from src.regional_context import HYDRO_COLUMNS, build_regional_context
from src.validation import build_test_mask_template


SEED = 20260908
ROUNDS = 98
SPANS = (6, 12)
RIDGE_ALPHA = 1000.0


def _json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_ids(sample_ids: pd.Series) -> str:
    return hashlib.sha256(sample_ids.astype(str).str.cat(sep="\n").encode("utf-8")).hexdigest()


def _emit(log, payload: object) -> None:
    text = json.dumps(payload, sort_keys=True)
    print(text, flush=True)
    log.write(text + "\n")
    log.flush()


def _fit_tree(lgb, params: dict[str, object], x_train: pd.DataFrame, y_train: np.ndarray,
              weights: np.ndarray, x_valid: pd.DataFrame):
    model = lgb.train(
        params,
        lgb.Dataset(x_train, label=y_train, weight=weights, feature_name=list(x_train.columns)),
        num_boost_round=ROUNDS,
        callbacks=[lgb.log_evaluation(0)],
    )
    return model, model.predict(x_valid)


def _ridge_design(train_raw: np.ndarray, valid_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Impute/normalize only from the training prefix and retain missing flags."""
    normalizer = PrefixNormalizer.fit(train_raw)
    train_norm = normalizer.transform(train_raw)
    valid_norm = normalizer.transform(valid_raw)
    train_design = np.concatenate([train_norm, np.isfinite(train_raw).astype(np.float32)], axis=1)
    valid_design = np.concatenate([valid_norm, np.isfinite(valid_raw).astype(np.float32)], axis=1)
    fingerprint = hashlib.sha256(
        np.ascontiguousarray(np.concatenate([normalizer.fill_values, normalizer.centers, normalizer.scales])).tobytes()
    ).hexdigest()
    return train_design, valid_design, fingerprint


def _save_oof(output_dir: Path, name: str, oof: pd.DataFrame, *, training_hash: str,
              extra: dict[str, object]) -> None:
    oof = oof.copy()
    oof["training_row_hash"] = training_hash
    for key, value in extra.items():
        oof[key] = value
    oof.to_csv(output_dir / f"oof_{name}.csv.gz", index=False, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage C hydrological sequence controls on Kaggle.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--origins", nargs="+", choices=INNER_ORIGINS, default=list(INNER_ORIGINS))
    parser.add_argument("--spans", nargs="+", type=int, choices=SPANS, default=list(SPANS))
    args = parser.parse_args()
    if len(set(args.origins)) != len(args.origins) or len(set(args.spans)) != len(args.spans):
        raise ValueError("origins and spans must not contain duplicates")

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit:
        raise RuntimeError({"expected_commit": args.expected_commit, "actual_commit": head})
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Refusing to run from a dirty checkout")
    required = [args.data_dir / name for name in ("Train.csv", "Test.csv", "SampleSubmission.csv")]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(f"required CSVs missing: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    manifest: dict[str, object] = {
        "status": "running",
        "stage": "HYDROLOGICAL_SEQUENCE_CONTROLS",
        "commit": head,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "no_test_predictions": True,
        "seed": SEED,
        "rounds": ROUNDS,
        "origins": args.origins,
        "spans": args.spans,
        "ridge_alpha": RIDGE_ALPHA,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    _json(args.output_dir / "manifest.json", manifest)
    memory_tracker = MemoryTracker()
    memory_tracker.start()
    results: list[dict[str, object]] = []
    try:
        with (args.output_dir / "console.log").open("w", encoding="utf-8") as log:
            _emit(log, {"phase": "start", "commit": head, "memory": memory_tracker.snapshot()})
            import joblib
            import lightgbm as lgb
            from sklearn.linear_model import Ridge

            train_cols = ["sample_id", "time", "lat", "lon", "TWS_t", "target", "month_sin", "month_cos", *HYDRO_COLUMNS]
            test_cols = ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked", *HYDRO_COLUMNS]
            train = pd.read_csv(args.data_dir / "Train.csv", usecols=train_cols)
            test = pd.read_csv(args.data_dir / "Test.csv", usecols=test_cols)
            template = build_test_mask_template(test[["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]])
            # Test is used solely to reconstruct the fixed replay geometry.
            del test
            structural = train[["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
            source = train.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
            labels = train[["sample_id", "target"]].copy()
            regional_source = train[["sample_id", "time", "lat", "lon", *HYDRO_COLUMNS]].copy()

            _emit(log, {"phase": "build_causal_trajectory_maps", "rows": int(len(source)), "memory": memory_tracker.snapshot()})
            regional = build_regional_context(regional_source)
            from src.hydro_trajectory import build_hydro_trajectory_map, build_regional_trajectory_map
            local_trajectory = build_hydro_trajectory_map(regional_source)
            regional_trajectory = build_regional_trajectory_map(regional_source, regional, local_trajectory)
            _emit(log, {"phase": "trajectory_maps_ready", "memory": memory_tracker.snapshot()})

            params = dict(
                DEFAULT_PARAMS,
                num_leaves=63,
                min_data_in_leaf=1000,
                num_threads=min(DEFAULT_NUM_THREADS, os.cpu_count() or 1),
                seed=SEED,
                feature_fraction_seed=SEED,
                bagging_seed=SEED,
                data_random_seed=SEED,
            )
            for origin in args.origins:
                fold, role = _fold(train, template, origin)
                start = pd.Period(origin, freq="M")
                sampled = build_sampled_training_rows(
                    structural, source[SOURCE_CORE_COLUMNS], max_target_month=start - 1, seed=SEED,
                )
                rows = sampled.rows
                y_train = _attach_delta(rows, labels)
                weights = horizon_rebalance_weights(rows.h)
                b3_train = _feature_frame(
                    "B3_both", rows, source, structural, regional, local_trajectory, regional_trajectory,
                )
                b3_valid = _feature_frame(
                    "B3_both", fold.ledger, source, structural, regional, local_trajectory, regional_trajectory,
                )
                if b3_train.columns.tolist() != b3_valid.columns.tolist():
                    raise AssertionError("B3 control schema differs between training and validation")
                row_hash = _hash_ids(rows.sample_id)
                coverage_hash = _hash_ids(fold.ledger.sample_id)
                _emit(log, {
                    "phase": "origin_features_ready", "origin": origin, "role": role,
                    "training_rows": int(len(rows)), "validation_rows": int(len(fold.ledger)),
                    "training_row_hash": row_hash, "coverage_hash": coverage_hash,
                    "memory": memory_tracker.snapshot(),
                })

                fit_started = time.perf_counter()
                model, delta = _fit_tree(lgb, params, b3_train, y_train, weights, b3_valid)
                prediction = fold.ledger.last_observed_TWS.to_numpy(dtype=np.float64) + delta
                baseline, oof = _score("B3_98_tree", fold, role, prediction)
                baseline.update({
                    "origin": origin,
                    "candidate": "B3_98_tree",
                    "span": None,
                    "feature_count": int(b3_train.shape[1]),
                    "training_rows": int(len(rows)),
                    "training_row_hash": row_hash,
                    "coverage_hash": coverage_hash,
                    "elapsed_seconds": time.perf_counter() - fit_started,
                })
                baseline_name = f"{origin}_B3_98_tree"
                model.save_model(str(args.output_dir / f"model_{baseline_name}.txt"))
                _save_oof(args.output_dir, baseline_name, oof, training_hash=row_hash, extra={"span": ""})
                results.append(baseline)
                _emit(log, {"phase": "fit_complete", "origin": origin, "candidate": "B3_98_tree",
                            "raw_rmse": baseline["raw_rmse"], "weighted_rmse": baseline["weighted_rmse"],
                            "memory": memory_tracker.snapshot()})
                model.free_dataset()

                requested_ids = pd.concat([rows.sample_id, fold.ledger.sample_id], ignore_index=True)
                if requested_ids.duplicated().any():
                    raise AssertionError("Stage C training/validation sample_ids overlap")
                for span in args.spans:
                    sequence = build_hydro_sequence_features(source, requested_ids, span=span, prefix=f"seq{span}_")
                    seq_train = sequence.iloc[:len(rows)].reset_index(drop=True)
                    seq_valid = sequence.iloc[len(rows):].reset_index(drop=True)
                    flat_train = pd.concat([b3_train.reset_index(drop=True), seq_train], axis=1)
                    flat_valid = pd.concat([b3_valid.reset_index(drop=True), seq_valid], axis=1)
                    if flat_train.columns.tolist() != flat_valid.columns.tolist():
                        raise AssertionError("flattened history schema differs between training and validation")

                    fit_started = time.perf_counter()
                    model, delta = _fit_tree(lgb, params, flat_train, y_train, weights, flat_valid)
                    prediction = fold.ledger.last_observed_TWS.to_numpy(dtype=np.float64) + delta
                    flat_score, oof = _score(f"flat_tree_span{span}", fold, role, prediction)
                    flat_score.update({
                        "origin": origin,
                        "candidate": f"flat_tree_span{span}",
                        "span": span,
                        "feature_count": int(flat_train.shape[1]),
                        "training_rows": int(len(rows)),
                        "training_row_hash": row_hash,
                        "coverage_hash": coverage_hash,
                        "elapsed_seconds": time.perf_counter() - fit_started,
                    })
                    flat_name = f"{origin}_flat_tree_span{span}"
                    model.save_model(str(args.output_dir / f"model_{flat_name}.txt"))
                    _save_oof(args.output_dir, flat_name, oof, training_hash=row_hash, extra={"span": span})
                    results.append(flat_score)
                    _emit(log, {"phase": "fit_complete", "origin": origin, "candidate": flat_score["candidate"],
                                "raw_rmse": flat_score["raw_rmse"], "weighted_rmse": flat_score["weighted_rmse"],
                                "memory": memory_tracker.snapshot()})
                    model.free_dataset()

                    raw_train = flat_train.to_numpy(dtype=np.float32)
                    raw_valid = flat_valid.to_numpy(dtype=np.float32)
                    ridge_train, ridge_valid, normalizer_hash = _ridge_design(raw_train, raw_valid)
                    fit_started = time.perf_counter()
                    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
                    ridge.fit(ridge_train, y_train, sample_weight=weights)
                    prediction = fold.ledger.last_observed_TWS.to_numpy(dtype=np.float64) + ridge.predict(ridge_valid)
                    ridge_score, oof = _score(f"ridge_span{span}", fold, role, prediction)
                    ridge_score.update({
                        "origin": origin,
                        "candidate": f"ridge_span{span}",
                        "span": span,
                        "feature_count": int(ridge_train.shape[1]),
                        "training_rows": int(len(rows)),
                        "training_row_hash": row_hash,
                        "coverage_hash": coverage_hash,
                        "normalizer_hash": normalizer_hash,
                        "ridge_alpha": RIDGE_ALPHA,
                        "elapsed_seconds": time.perf_counter() - fit_started,
                    })
                    ridge_name = f"{origin}_ridge_span{span}"
                    joblib.dump(ridge, args.output_dir / f"model_{ridge_name}.joblib")
                    _save_oof(args.output_dir, ridge_name, oof, training_hash=row_hash, extra={"span": span})
                    results.append(ridge_score)
                    _emit(log, {"phase": "fit_complete", "origin": origin, "candidate": ridge_score["candidate"],
                                "raw_rmse": ridge_score["raw_rmse"], "weighted_rmse": ridge_score["weighted_rmse"],
                                "memory": memory_tracker.snapshot()})
                    del sequence, seq_train, seq_valid, flat_train, flat_valid, raw_train, raw_valid, ridge_train, ridge_valid

                _json(args.output_dir / "metrics.partial.json", {"results": results})
                del b3_train, b3_valid

            baseline_by_origin = {r["origin"]: r for r in results if r["candidate"] == "B3_98_tree"}
            comparisons = []
            for row in results:
                if row["candidate"] == "B3_98_tree":
                    continue
                baseline = baseline_by_origin[str(row["origin"])]
                comparisons.append({
                    "origin": row["origin"],
                    "candidate": row["candidate"],
                    "span": row["span"],
                    "raw_rmse_gain_vs_B3_98_tree": float(baseline["raw_rmse"] - row["raw_rmse"]),
                    "weighted_rmse_gain_vs_B3_98_tree": float(baseline["weighted_rmse"] - row["weighted_rmse"]),
                })
            _json(args.output_dir / "metrics.json", {
                "results": results,
                "comparisons_vs_frozen_B3_98_tree": comparisons,
                "protocol": {
                    "test_predictions": False,
                    "test_usage": "template_geometry_only",
                    "ridge_alpha": RIDGE_ALPHA,
                    "prefix_normalization": True,
                },
            })
            manifest.update({
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "result_rows": len(results),
                "memory_peak": memory_tracker.snapshot(),
                "output_checksums": {
                    str(path.relative_to(args.output_dir)): _sha256(path)
                    for path in args.output_dir.rglob("*") if path.is_file() and path.name != "manifest.json"
                },
            })
            _json(args.output_dir / "manifest.json", manifest)
            package = shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir)
            manifest["package"] = {"path": package, "bytes": Path(package).stat().st_size, "sha256": _sha256(Path(package))}
            _json(args.output_dir / "manifest.json", manifest)
            _emit(log, {"status": "completed", "output_dir": str(args.output_dir), "package": package, "results": len(results)})
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "memory_peak": memory_tracker.snapshot(),
        })
        _json(args.output_dir / "manifest.json", manifest)
        raise
    finally:
        memory_tracker.stop()


if __name__ == "__main__":
    main()
