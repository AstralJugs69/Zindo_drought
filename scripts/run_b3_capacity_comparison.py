"""Bounded B3 capacity comparison at the two recent out-of-time origins.

This runner fits the unchanged D0-dense B3 recipe to 392 boosting rounds and
scores the same fitted booster at iterations 98 and 392.  It builds one legal
historical observation view and one pair of feature matrices per origin, then
reuses those matrices for both checkpoints.  It reads Train.csv only for the
required recent-origin comparison; Test.csv is read only if the frozen recent
gate passes and the explicitly requested 2009-01 transfer check is needed.

No Test predictions, submission files, calibration, or uploads are produced.
The 392-round fit is deliberately not early-stopped on outer labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_lgbm_core import DEFAULT_PARAMS
from src.availability import build_replay_observation_view
from src.metrics import TEST_H_WEIGHTS
from src.ml_features import (
    SOURCE_CORE_COLUMNS,
    SOURCE_HYDRO_HISTORY_COLUMNS,
    build_sampled_training_rows,
    horizon_rebalance_weights,
)
from src.neural_sequence import build_b3_feature_maps, build_b3_matrix
from src.observation_simulator import (
    build_mask_block_fold,
    build_template_replay_fold,
)
from src.regional_context import HYDRO_COLUMNS
from src.validation import build_test_mask_template


SEED = 20260908
BASE_ROUNDS = 98
EXTENDED_ROUNDS = 392
NUM_LEAVES = 63
MIN_DATA_IN_LEAF = 1000
DEFAULT_NUM_THREADS = 4
MAX_NUM_THREADS = 24
RECENT_ORIGINS = ("2014-04", "2014-12")
OLDER_ORIGIN = "2009-01"
END_MONTHS = {"2014-04": "2014-10", "2014-12": "2015-06"}
TRAIN_COLUMNS = [
    "sample_id", "time", "lat", "lon", "TWS_t", "month_sin", "month_cos",
    *HYDRO_COLUMNS, "target",
]
TEST_COLUMNS = ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_values(values: pd.Series | np.ndarray) -> str:
    if isinstance(values, pd.Series):
        data = values.astype(str).str.cat(sep="\n").encode("utf-8")
    else:
        data = np.ascontiguousarray(values).tobytes()
    return hashlib.sha256(data).hexdigest()


def _hash_frame(frame: pd.DataFrame) -> str:
    values = np.ascontiguousarray(frame.to_numpy(dtype=np.float32))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _memory() -> dict[str, float | None]:
    try:
        import psutil

        process = psutil.Process()
        return {
            "rss_gib": process.memory_info().rss / 1024**3,
            "available_gib": psutil.virtual_memory().available / 1024**3,
        }
    except Exception:
        return {"rss_gib": None, "available_gib": None}


class MemoryTracker:
    """Continuously retain peak resident memory without retaining samples."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._process = None
        self._psutil = None
        self._peak_rss_gib: float | None = None
        self._min_available_gib: float | None = None
        self._note: str | None = None
        try:
            import psutil

            self._psutil = psutil
            self._process = psutil.Process()
        except Exception as exc:
            self._note = f"memory sampling unavailable: {type(exc).__name__}: {exc}"

    def _sample(self) -> None:
        if self._process is None or self._psutil is None:
            return
        try:
            rss = self._process.memory_info().rss / 1024**3
            available = self._psutil.virtual_memory().available / 1024**3
            with self._lock:
                self._peak_rss_gib = max(self._peak_rss_gib or 0.0, rss)
                self._min_available_gib = (
                    available
                    if self._min_available_gib is None
                    else min(self._min_available_gib, available)
                )
        except Exception as exc:
            self._note = f"memory sampling unavailable: {type(exc).__name__}: {exc}"

    def _watch(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._sample()
        if self._process is not None:
            self._thread = threading.Thread(target=self._watch, daemon=True)
            self._thread.start()

    def snapshot(self) -> dict[str, float | None]:
        self._sample()
        if self._note is not None:
            return {"rss_gib": None, "available_gib": None, "note": self._note}
        with self._lock:
            return {
                **_memory(),
                "peak_rss_gib": self._peak_rss_gib,
                "min_available_gib": self._min_available_gib,
            }

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 2)


def _attach_training_labels(rows: pd.DataFrame, labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    joined = rows.loc[:, ["sample_id", "last_observed_TWS"]].merge(
        labels.loc[:, ["sample_id", "target"]],
        how="left",
        on="sample_id",
        validate="one_to_one",
        sort=False,
    )
    if joined["target"].isna().any():
        raise AssertionError("training row has no supplied target")
    target = joined["target"].to_numpy(dtype=np.float32)
    delta = target - joined["last_observed_TWS"].to_numpy(dtype=np.float32)
    return delta, target


def _metric(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if len(error) == 0:
        return {"rows": 0, "sse": 0.0, "raw_rmse": None, "mae": None, "bias": None}
    return {
        "rows": int(len(error)),
        "sse": float(np.sum(np.square(error))),
        "raw_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def _weighted_h1_7(target: np.ndarray, prediction: np.ndarray, horizons: np.ndarray) -> float | None:
    present = sorted(int(value) for value in np.unique(horizons))
    if present != list(range(1, 8)):
        return None
    weighted_mse = 0.0
    for horizon in range(1, 8):
        mask = horizons == horizon
        weighted_mse += TEST_H_WEIGHTS[horizon] * float(
            np.mean(np.square(prediction[mask].astype(np.float64) - target[mask].astype(np.float64)))
        )
    return float(np.sqrt(weighted_mse))


def _validation_summary(
    origin: str,
    checkpoint: int,
    target: np.ndarray,
    prediction: np.ndarray,
    ledger: pd.DataFrame,
) -> dict[str, Any]:
    horizons = ledger["h"].to_numpy(dtype=np.int16)
    official = (horizons >= 1) & (horizons <= 7)
    tail = ~official
    result: dict[str, Any] = {
        "origin": origin,
        "checkpoint": checkpoint,
        "rows": int(len(target)),
        "raw_rmse": _metric(target, prediction)["raw_rmse"],
        "mae": _metric(target, prediction)["mae"],
        "bias": _metric(target, prediction)["bias"],
        "sse": _metric(target, prediction)["sse"],
        "horizons_present": sorted(int(value) for value in np.unique(horizons)),
        "h1_7_rows": int(official.sum()),
        "h1_7_raw_rmse": None,
        "h1_7_sse": None,
        "h1_7_weighted_rmse": None,
        "h1_7_horizons_present": sorted(int(value) for value in np.unique(horizons[official])),
        "stress_h_gt7_rows": int(tail.sum()),
        "stress_h_gt7_raw_rmse": None,
        "stress_h_gt7_sse": None,
    }
    if official.any():
        official_metric = _metric(target[official], prediction[official])
        result["h1_7_raw_rmse"] = official_metric["raw_rmse"]
        result["h1_7_sse"] = official_metric["sse"]
        result["h1_7_weighted_rmse"] = _weighted_h1_7(target[official], prediction[official], horizons[official])
    if tail.any():
        tail_metric = _metric(target[tail], prediction[tail])
        result["stress_h_gt7_raw_rmse"] = tail_metric["raw_rmse"]
        result["stress_h_gt7_sse"] = tail_metric["sse"]
    return result


def _long_validation_frame(
    origin: str,
    model: str,
    checkpoint: int,
    ledger: pd.DataFrame,
    labels: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    frame = ledger.loc[:, ["sample_id", "source_date", "target_date", "h", "lat", "lon", "last_observed_date", "last_observed_TWS"]].copy()
    frame["origin"] = origin
    frame["model"] = model
    frame["checkpoint"] = checkpoint
    frame["target"] = labels.astype(np.float64)
    frame["prediction"] = prediction.astype(np.float64)
    frame["error"] = frame["prediction"] - frame["target"]
    frame["squared_error"] = np.square(frame["error"])
    frame["abs_error"] = np.abs(frame["error"])
    frame["source_month"] = pd.to_datetime(frame["source_date"]).dt.to_period("M").astype(str)
    frame["target_month"] = pd.to_datetime(frame["target_date"]).dt.to_period("M").astype(str)
    frame["cell_lat5"] = np.floor(frame["lat"].astype(float) / 5.0) * 5.0
    frame["cell_lon5"] = np.floor(frame["lon"].astype(float) / 5.0) * 5.0
    frame["scope"] = np.where(frame["h"].between(1, 7), "official_h1_7", "stress_h_gt7")
    return frame


def _aggregate_validation(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(
        ["origin", "model", "checkpoint", "scope", *group_columns],
        dropna=False,
        sort=True,
    ).agg(
        rows=("target", "size"),
        sse=("squared_error", "sum"),
        abs_error_sum=("abs_error", "sum"),
        error_sum=("error", "sum"),
    ).reset_index()
    grouped["raw_rmse"] = np.sqrt(grouped["sse"] / grouped["rows"])
    grouped["mae"] = grouped["abs_error_sum"] / grouped["rows"]
    grouped["bias"] = grouped["error_sum"] / grouped["rows"]
    return grouped


def _fold_for_origin(
    train: pd.DataFrame,
    origin: str,
    *,
    template: pd.DataFrame | None = None,
) -> tuple[Any, str]:
    if origin in END_MONTHS:
        fold = build_mask_block_fold(
            train,
            anchor_month=origin,
            end_month=END_MONTHS[origin],
            scenario_id=f"capacity_{origin.replace('-', '_')}",
            family="recent_mask_block",
        )
        return fold, "recent_development"
    if origin == OLDER_ORIGIN:
        if template is None:
            raise ValueError("2009-01 requires the target-blind Test schedule template")
        fold = build_template_replay_fold(
            train,
            template,
            start_month=origin,
            scenario_id="capacity_2009_01",
            family="development_template_replay",
        )
        return fold, "development"
    raise ValueError(f"unsupported capacity origin: {origin}")


def _d0_control(
    origin: str,
    d0_dir: Path | None,
    valid_x: pd.DataFrame,
    ledger: pd.DataFrame,
    target: np.ndarray,
    anchor: np.ndarray,
    iter98_prediction: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "origin": origin,
        "status": "missing_control",
        "requested_directory": None if d0_dir is None else str(d0_dir),
    }
    if d0_dir is None:
        return result
    model_path = d0_dir / "model.txt"
    oof_path = d0_dir / "oof.csv.gz"
    if not model_path.is_file():
        return result
    try:
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(model_path))
        feature_names = list(booster.feature_name())
        if feature_names != valid_x.columns.tolist():
            raise AssertionError("D0 control feature schema differs from current validation matrix")
        d0_prediction = ledger["last_observed_TWS"].to_numpy(dtype=np.float64) + booster.predict(valid_x)
        delta = d0_prediction - iter98_prediction
        result.update(
            {
                "status": "available",
                "model_path": str(model_path),
                "model_sha256": _sha256(model_path),
                "feature_count": int(len(feature_names)),
                "prediction_max_abs_difference_to_iter98": float(np.max(np.abs(delta))) if len(delta) else 0.0,
                "prediction_rmse_difference_to_iter98": float(np.sqrt(np.mean(np.square(delta)))) if len(delta) else 0.0,
                "prediction_within_1e-6": bool(not len(delta) or np.max(np.abs(delta)) <= 1e-6),
            }
        )
        if oof_path.is_file():
            common = [
                "sample_id", "source_date", "last_observed_date", "last_observed_TWS",
                "h", "lat", "lon", "target", "prediction",
            ]
            saved = pd.read_csv(oof_path, usecols=common)
            saved["sample_id"] = saved["sample_id"].astype(str)
            current = pd.DataFrame(
                {
                    "sample_id": ledger["sample_id"].astype(str).to_numpy(),
                    "source_date": pd.to_datetime(ledger["source_date"]).dt.strftime("%Y-%m-%d"),
                    "last_observed_date": pd.to_datetime(ledger["last_observed_date"]).dt.strftime("%Y-%m-%d"),
                    "last_observed_TWS": anchor.astype(np.float64),
                    "h": ledger["h"].to_numpy(dtype=np.int16),
                    "lat": ledger["lat"].to_numpy(dtype=np.float64),
                    "lon": ledger["lon"].to_numpy(dtype=np.float64),
                    "target": target.astype(np.float64),
                }
            )
            saved["source_date"] = pd.to_datetime(saved["source_date"]).dt.strftime("%Y-%m-%d")
            saved["last_observed_date"] = pd.to_datetime(saved["last_observed_date"]).dt.strftime("%Y-%m-%d")
            joined = current.merge(saved, on="sample_id", how="outer", suffixes=("_current", "_saved"), validate="one_to_one")
            if joined.isna().any(axis=None):
                raise AssertionError("D0 OOF control has a non-matching validation identity")
            result["oof_path"] = str(oof_path)
            result["oof_sha256"] = _sha256(oof_path)
            result["oof_rows"] = int(len(saved))
            result["oof_ids_equal"] = bool(_hash_values(current["sample_id"]) == _hash_values(saved["sample_id"]))
            result["oof_target_max_abs_difference"] = float(np.max(np.abs(joined["target_current"] - joined["target_saved"])))
            result["oof_anchor_max_abs_difference"] = float(np.max(np.abs(joined["last_observed_TWS_current"] - joined["last_observed_TWS_saved"])))
        return result
    except Exception as exc:
        result.update({"status": "control_error", "error": f"{type(exc).__name__}: {exc}"})
        raise


def _run_origin(
    *,
    train: pd.DataFrame,
    labels: pd.DataFrame,
    origin: str,
    template: pd.DataFrame | None,
    output_dir: Path,
    params: dict[str, Any],
    memory: MemoryTracker,
    log,
    d0_dir: Path | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    import lightgbm as lgb

    def emit(payload: dict[str, Any]) -> None:
        payload = {**payload, "memory": memory.snapshot()}
        print(json.dumps(payload, sort_keys=True, default=str), flush=True)
        log.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        log.flush()

    fold, role = _fold_for_origin(train, origin, template=template)
    source = train.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
    structural = train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
    labels_plain = labels.loc[:, ["sample_id", "target"]].copy()
    view = build_replay_observation_view(
        source,
        structural,
        ledger=fold.ledger,
        first_source_month=fold.spec.first_source_month,
        last_source_month=fold.spec.last_source_month,
    )
    cutoff = pd.Period(origin, freq="M") - 1
    sampled = build_sampled_training_rows(
        view.structural,
        view.source.loc[:, SOURCE_CORE_COLUMNS],
        max_target_month=cutoff,
        seed=SEED,
    )
    rows = sampled.rows
    y_delta, y_train_target = _attach_training_labels(rows, labels_plain)
    weights = np.asarray(horizon_rebalance_weights(rows["h"]), dtype=np.float32)
    maps = build_b3_feature_maps(view.source)
    emit({"phase": "features_start", "origin": origin, "training_rows": int(len(rows)), "validation_rows": int(len(fold.ledger))})
    x_train = build_b3_matrix(rows, view.source, view.structural, maps)
    x_valid = build_b3_matrix(fold.ledger, view.source, view.structural, maps)
    if x_train.columns.tolist() != x_valid.columns.tolist():
        raise AssertionError("training and validation B3 schemas differ")
    feature_names = x_train.columns.tolist()
    if len(feature_names) != 446:
        raise AssertionError(f"B3 schema must contain 446 features, found {len(feature_names)}")
    feature_fingerprint = {
        "feature_count": len(feature_names),
        "feature_names_hash": _hash_values(feature_names),
        "training_matrix_sha256": _hash_frame(x_train),
        "validation_matrix_sha256": _hash_frame(x_valid),
        "view_source_ids_hash": _hash_values(view.source["sample_id"]),
        "view_structural_ids_hash": _hash_values(view.structural["sample_id"]),
        "view_source_rows": int(len(view.source)),
        "view_structural_rows": int(len(view.structural)),
        "view_withheld_window_rows": int(view.withheld_window_rows),
    }
    training_fingerprint = {
        "target_cutoff": str(cutoff),
        "rows": int(len(rows)),
        "source_rows": int(len(train)),
        "dropped_missing_anchor": int(sampled.dropped_missing_anchor),
        "horizon_counts": {str(int(k)): int(v) for k, v in rows["h"].value_counts().sort_index().items()},
        "training_ids_hash": _hash_values(rows["sample_id"]),
        "training_target_hash": _hash_values(y_train_target),
        "training_delta_hash": _hash_values(y_delta),
        "training_weights_hash": _hash_values(weights),
    }
    validation_target = fold.labels["target"].to_numpy(dtype=np.float32)
    validation_anchor = fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float32)
    validation_fingerprint = {
        "rows": int(len(fold.ledger)),
        "validation_ids_hash": _hash_values(fold.ledger["sample_id"]),
        "validation_target_hash": _hash_values(validation_target),
        "validation_anchor_hash": _hash_values(validation_anchor),
        "validation_h_hash": _hash_values(fold.ledger["h"].to_numpy(dtype=np.int16)),
        "source_date_hash": _hash_values(pd.to_datetime(fold.ledger["source_date"]).dt.strftime("%Y-%m-%d")),
        "target_date_hash": _hash_values(pd.to_datetime(fold.ledger["target_date"]).dt.strftime("%Y-%m-%d")),
        "horizons_present": sorted(int(value) for value in fold.ledger["h"].unique()),
        "exclusions": fold.exclusions,
    }
    emit({"phase": "features_ready", "origin": origin, "training": training_fingerprint, "validation": validation_fingerprint, "feature_fingerprint": feature_fingerprint})

    train_set = lgb.Dataset(x_train, label=y_delta, weight=weights, feature_name=feature_names, free_raw_data=True)
    fit_started = time.perf_counter()
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=EXTENDED_ROUNDS,
        callbacks=[lgb.log_evaluation(50)],
    )
    fit_seconds = time.perf_counter() - fit_started
    emit({"phase": "fit_complete", "origin": origin, "fit_seconds": fit_seconds, "rounds": EXTENDED_ROUNDS})

    validation_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    checkpoint_paths: dict[str, str] = {}
    checkpoint_hashes: dict[str, str] = {}
    train_metrics: list[dict[str, Any]] = []
    iter98_prediction: np.ndarray | None = None
    for checkpoint in (BASE_ROUNDS, EXTENDED_ROUNDS):
        key = f"iter{checkpoint}"
        model_path = output_dir / f"model_{origin}_{key}.txt"
        booster.save_model(str(model_path), num_iteration=checkpoint)
        checkpoint_paths[key] = str(model_path)
        checkpoint_hashes[key] = _sha256(model_path)
        train_prediction = rows["last_observed_TWS"].to_numpy(dtype=np.float64) + booster.predict(x_train, num_iteration=checkpoint)
        valid_prediction = fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float64) + booster.predict(x_valid, num_iteration=checkpoint)
        train_metric = _metric(y_train_target, train_prediction)
        train_metric.update({"origin": origin, "checkpoint": checkpoint, "target_scale": "absolute_target"})
        train_metrics.append(train_metric)
        summary = _validation_summary(origin, checkpoint, validation_target, valid_prediction, fold.ledger)
        summary.update({"role": role, "training_target_cutoff": str(cutoff), "fit_seconds": fit_seconds})
        summaries.append(summary)
        validation_frames.append(_long_validation_frame(origin, key, checkpoint, fold.ledger, validation_target, valid_prediction))
        if checkpoint == BASE_ROUNDS:
            iter98_prediction = valid_prediction
        oof = fold.ledger.loc[:, ["sample_id", "source_date", "target_date", "last_observed_date", "last_observed_TWS", "h", "lat", "lon"]].copy()
        oof["target"] = validation_target
        oof["prediction"] = valid_prediction
        oof["model"] = key
        oof["training_ids_hash"] = training_fingerprint["training_ids_hash"]
        oof_path = output_dir / f"oof_{origin}_{key}.csv.gz"
        oof.to_csv(oof_path, index=False, compression="gzip")
        checkpoint_paths[f"oof_{key}"] = str(oof_path)
        checkpoint_hashes[f"oof_{key}"] = _sha256(oof_path)
    if iter98_prediction is None:
        raise AssertionError("iteration-98 prediction was not created")

    persistence_prediction = fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float64)
    persistence_frame = _long_validation_frame(origin, "persistence", 0, fold.ledger, validation_target, persistence_prediction)
    validation_frames.append(persistence_frame)
    persistence_summary = _validation_summary(origin, 0, validation_target, persistence_prediction, fold.ledger)
    persistence_summary.update({"role": role, "training_target_cutoff": str(cutoff), "fit_seconds": 0.0})
    summaries.append(persistence_summary)
    control = _d0_control(origin, d0_dir, x_valid, fold.ledger, validation_target, validation_anchor.astype(np.float64), iter98_prediction)
    origin_result = {
        "origin": origin,
        "role": role,
        "scenario": {
            "scenario_id": fold.spec.scenario_id,
            "family": fold.spec.family,
            "first_source_month": fold.spec.first_source_month,
            "last_source_month": fold.spec.last_source_month,
            "training_target_cutoff": fold.spec.training_target_cutoff,
            "visibility_rule": fold.spec.visibility_rule,
        },
        "training": training_fingerprint,
        "validation": validation_fingerprint,
        "feature_fingerprint": feature_fingerprint,
        "fit": {"rounds": EXTENDED_ROUNDS, "fit_seconds": fit_seconds, "num_threads": params["num_threads"], "num_leaves": NUM_LEAVES, "min_data_in_leaf": MIN_DATA_IN_LEAF},
        "model_checkpoints": checkpoint_paths,
        "checkpoint_hashes": checkpoint_hashes,
        "train_metrics": train_metrics,
        "validation_metrics": summaries,
        "d0_control": control,
        "no_test_predictions": True,
        "submission_written": False,
    }
    _atomic_json(output_dir / f"origin_{origin}_details.json", origin_result)
    del booster, train_set, x_train, x_valid, maps, view, source, structural, rows, y_delta, weights
    gc.collect()
    return origin_result, pd.concat(validation_frames, ignore_index=True), pd.DataFrame(train_metrics)


def _evaluate_gate(summaries: list[dict[str, Any]], monthly: pd.DataFrame) -> dict[str, Any]:
    recent_rows: list[dict[str, Any]] = []
    for origin in RECENT_ORIGINS:
        rows = [
            row for row in summaries
            if row["origin"] == origin and row["checkpoint"] in (BASE_ROUNDS, EXTENDED_ROUNDS)
        ]
        by_checkpoint = {int(row["checkpoint"]): row for row in rows}
        baseline = by_checkpoint[BASE_ROUNDS]
        extended = by_checkpoint[EXTENDED_ROUNDS]
        improvement = None
        if baseline.get("h1_7_raw_rmse") is not None and extended.get("h1_7_raw_rmse") is not None:
            improvement = float(baseline["h1_7_raw_rmse"] - extended["h1_7_raw_rmse"])
        recent_rows.append(
            {
                "origin": origin,
                "h1_7_horizons_present": baseline.get("h1_7_horizons_present"),
                "h1_7_complete": set(baseline.get("h1_7_horizons_present", [])) == set(range(1, 8)),
                "iter98_h1_7_raw_rmse": baseline.get("h1_7_raw_rmse"),
                "iter392_h1_7_raw_rmse": extended.get("h1_7_raw_rmse"),
                "improvement_iter98_minus_iter392": improvement,
                "improvement_gate_pass": bool(improvement is not None and improvement >= 0.005),
            }
        )
    month_rows = monthly.loc[
        (monthly["scope"] == "official_h1_7")
        & monthly["source_month"].notna()
        & (monthly["rows"] >= 1000)
    ].copy()
    pivot = month_rows.pivot_table(index=["origin", "source_month"], columns="model", values="raw_rmse", aggfunc="first").reset_index()
    regressions: list[dict[str, Any]] = []
    if "iter98" in pivot and "iter392" in pivot:
        pivot["delta_iter392_minus_iter98"] = pivot["iter392"] - pivot["iter98"]
        for row in pivot.to_dict(orient="records"):
            if float(row["delta_iter392_minus_iter98"]) > 0.02:
                regressions.append({**row, "regression_gate_pass": False})
    recent_pass = bool(
        len(recent_rows) == len(RECENT_ORIGINS)
        and all(row["improvement_gate_pass"] for row in recent_rows)
        and not regressions
    )
    return {
        "recent_origin_rows": recent_rows,
        "monthly_regressions_over_0.02": regressions,
        "recent_gate_pass": recent_pass,
        "thresholds": {
            "minimum_h1_7_raw_rmse_improvement_each_recent_origin": 0.005,
            "maximum_monthly_raw_rmse_regression_rows_at_least_1000": 0.02,
        },
        "note": "April h=4 is not imputed; its available h1-7 raw RMSE is reported with incomplete coverage.",
    }


def _evaluate_older_gate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        row for row in summaries
        if row["origin"] == OLDER_ORIGIN and row["checkpoint"] in (BASE_ROUNDS, EXTENDED_ROUNDS)
    ]
    if len(rows) != 2:
        return {"status": "not_available", "origin": OLDER_ORIGIN}
    by_checkpoint = {int(row["checkpoint"]): row for row in rows}
    baseline = by_checkpoint[BASE_ROUNDS]
    extended = by_checkpoint[EXTENDED_ROUNDS]
    baseline_rmse = baseline.get("h1_7_raw_rmse")
    extended_rmse = extended.get("h1_7_raw_rmse")
    if baseline_rmse is None or extended_rmse is None:
        return {
            "status": "not_available",
            "origin": OLDER_ORIGIN,
            "iter98_h1_7_raw_rmse": baseline_rmse,
            "iter392_h1_7_raw_rmse": extended_rmse,
        }
    delta = float(extended_rmse - baseline_rmse)
    return {
        "status": "evaluated",
        "origin": OLDER_ORIGIN,
        "iter98_h1_7_raw_rmse": float(baseline_rmse),
        "iter392_h1_7_raw_rmse": float(extended_rmse),
        "regression_iter392_minus_iter98": delta,
        "maximum_allowed_regression": 0.005,
        "gate_pass": bool(delta <= 0.005),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--d0-apr-dir", type=Path)
    parser.add_argument("--d0-dec-dir", type=Path)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit:
        raise RuntimeError({"expected_commit": args.expected_commit, "actual_commit": head})
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("refusing to run from a dirty checkout")
    available_cpus = os.cpu_count() or 1
    if args.num_threads < 1 or args.num_threads > MAX_NUM_THREADS:
        raise ValueError(f"--num-threads must be in [1, {MAX_NUM_THREADS}]")
    num_threads = min(int(args.num_threads), available_cpus)
    train_path = args.data_dir / "Train.csv"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "status": "running",
        "stage": "B3_CAPACITY_98_VS_392",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit": head,
        "branch": branch,
        "seed": SEED,
        "base_rounds": BASE_ROUNDS,
        "extended_rounds": EXTENDED_ROUNDS,
        "num_leaves": NUM_LEAVES,
        "min_data_in_leaf": MIN_DATA_IN_LEAF,
        "num_threads": num_threads,
        "requested_num_threads": int(args.num_threads),
        "origins": list(RECENT_ORIGINS),
        "older_origin_if_gate_passes": OLDER_ORIGIN,
        "no_test_predictions": True,
        "test_rows_read": 0,
        "test_labels_read": False,
        "submission_written": False,
        "data": {
            "train_path": str(train_path),
            "train_sha256": _sha256(train_path),
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": available_cpus,
            "packages": {name: _version(name) for name in ("numpy", "pandas", "lightgbm", "scikit-learn")},
        },
    }
    _atomic_json(args.output_dir / "manifest.json", manifest)
    memory = MemoryTracker()
    memory.start()
    started = time.perf_counter()
    try:
        train = pd.read_csv(train_path, usecols=TRAIN_COLUMNS)
        if train["sample_id"].duplicated().any() or train[["lat", "lon", "time"]].duplicated().any():
            raise AssertionError("Train IDs and location-month keys must be unique")
        if train["target"].isna().any():
            raise AssertionError("Train targets must be finite")
        labels = train.loc[:, ["sample_id", "target"]].copy()
        params = dict(
            DEFAULT_PARAMS,
            num_leaves=NUM_LEAVES,
            min_data_in_leaf=MIN_DATA_IN_LEAF,
            num_threads=num_threads,
            seed=SEED,
            feature_fraction_seed=SEED,
            bagging_seed=SEED,
            data_random_seed=SEED,
        )
        log_path = args.output_dir / "events.jsonl"
        log = log_path.open("a", encoding="utf-8")
        try:
            log.write(json.dumps({"phase": "start", "manifest": manifest}, sort_keys=True) + "\n")
            log.flush()
            origins: list[dict[str, Any]] = []
            validation_frames: list[pd.DataFrame] = []
            train_frames: list[pd.DataFrame] = []
            d0_dirs = {"2014-04": args.d0_apr_dir, "2014-12": args.d0_dec_dir}
            for origin in RECENT_ORIGINS:
                result, validation, train_metrics = _run_origin(
                    train=train,
                    labels=labels,
                    origin=origin,
                    template=None,
                    output_dir=args.output_dir,
                    params=params,
                    memory=memory,
                    log=log,
                    d0_dir=d0_dirs.get(origin),
                )
                origins.append(result)
                validation_frames.append(validation)
                train_frames.append(train_metrics)
            validation_long = pd.concat(validation_frames, ignore_index=True)
            validation_long.to_csv(args.output_dir / "validation_long.csv.gz", index=False, compression="gzip")
            train_metrics = pd.concat(train_frames, ignore_index=True)
            train_metrics.to_csv(args.output_dir / "train_metrics.csv", index=False)
            monthly = _aggregate_validation(validation_long, ["source_month"])
            by_horizon = _aggregate_validation(validation_long, ["h"])
            by_geo5 = _aggregate_validation(validation_long, ["source_month", "cell_lat5", "cell_lon5"])
            overall = _aggregate_validation(validation_long, [])
            monthly.to_csv(args.output_dir / "validation_by_source_month.csv", index=False)
            by_horizon.to_csv(args.output_dir / "validation_by_horizon.csv", index=False)
            by_geo5.to_csv(args.output_dir / "validation_by_geo5.csv", index=False)
            overall.to_csv(args.output_dir / "validation_overall.csv", index=False)
            summary_rows = [row for origin in origins for row in origin["validation_metrics"]]
            gate = _evaluate_gate(summary_rows, monthly)
            _atomic_json(args.output_dir / "gate.json", gate)
            older_result: dict[str, Any] | None = None
            older_gate: dict[str, Any] = {"status": "not_run", "origin": OLDER_ORIGIN}
            if gate["recent_gate_pass"]:
                test_path = args.data_dir / "Test.csv"
                test = pd.read_csv(test_path, usecols=TEST_COLUMNS)
                template = build_test_mask_template(test)
                manifest["test_rows_read"] = int(len(test))
                older_result, older_validation, older_train_metrics = _run_origin(
                    train=train,
                    labels=labels,
                    origin=OLDER_ORIGIN,
                    template=template,
                    output_dir=args.output_dir,
                    params=params,
                    memory=memory,
                    log=log,
                    d0_dir=None,
                )
                origins.append(older_result)
                validation_frames.append(older_validation)
                train_frames.append(older_train_metrics)
                validation_long = pd.concat(validation_frames, ignore_index=True)
                train_metrics = pd.concat(train_frames, ignore_index=True)
                validation_long.to_csv(args.output_dir / "validation_long.csv.gz", index=False, compression="gzip")
                train_metrics.to_csv(args.output_dir / "train_metrics.csv", index=False)
                _aggregate_validation(validation_long, ["source_month"]).to_csv(args.output_dir / "validation_by_source_month.csv", index=False)
                _aggregate_validation(validation_long, ["h"]).to_csv(args.output_dir / "validation_by_horizon.csv", index=False)
                _aggregate_validation(validation_long, ["source_month", "cell_lat5", "cell_lon5"]).to_csv(args.output_dir / "validation_by_geo5.csv", index=False)
                _aggregate_validation(validation_long, []).to_csv(args.output_dir / "validation_overall.csv", index=False)
                summary_rows = [row for origin in origins for row in origin["validation_metrics"]]
                older_gate = _evaluate_older_gate(summary_rows)
                gate["older_transfer_gate"] = older_gate
                _atomic_json(args.output_dir / "gate.json", gate)
            manifest.update(
                {
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "memory_peak": memory.snapshot(),
                    "origins_completed": [row["origin"] for row in origins],
                    "recent_gate": gate,
                    "older_transfer_gate": older_gate,
                    "older_transfer_check_run": older_result is not None,
                    "older_transfer_check_origin": OLDER_ORIGIN if older_result is not None else None,
                    "artifacts": {
                        "manifest": str(args.output_dir / "manifest.json"),
                        "validation_long": str(args.output_dir / "validation_long.csv.gz"),
                        "train_metrics": str(args.output_dir / "train_metrics.csv"),
                        "validation_by_source_month": str(args.output_dir / "validation_by_source_month.csv"),
                        "validation_by_horizon": str(args.output_dir / "validation_by_horizon.csv"),
                        "validation_by_geo5": str(args.output_dir / "validation_by_geo5.csv"),
                        "validation_overall": str(args.output_dir / "validation_overall.csv"),
                        "gate": str(args.output_dir / "gate.json"),
                    },
                }
            )
            _atomic_json(args.output_dir / "manifest.json", manifest)
            print(json.dumps(manifest, sort_keys=True, default=str), flush=True)
        finally:
            log.close()
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "memory_peak": memory.snapshot(),
            }
        )
        _atomic_json(args.output_dir / "manifest.json", manifest)
        raise
    finally:
        memory.stop()


if __name__ == "__main__":
    main()
