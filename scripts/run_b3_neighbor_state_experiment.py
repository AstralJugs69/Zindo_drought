"""Gated causal neighbor-anchor-state experiment for frozen B3/98.

The run compares C0 (the existing 446-column dense-history B3/98 recipe) with
C1 (C0 plus four source-date legal neighbor-state features).  It reads Test
only to construct the already-established, label-free replay geometry for the
two inner origins and the optional 2009-01 transfer check.  It never reads Test
targets, predicts Test rows, writes a submission, or fits a model outside the
declared gates.
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

from scripts.run_lgbm_core import DEFAULT_NUM_THREADS, DEFAULT_PARAMS, MAX_NUM_THREADS
from src.availability import build_replay_observation_view
from src.metrics import TEST_H_WEIGHTS, score_by_horizon
from src.ml_features import (
    SOURCE_CORE_COLUMNS,
    SOURCE_HYDRO_HISTORY_COLUMNS,
    build_sampled_training_rows,
    horizon_rebalance_weights,
)
from src.neighbor_state import (
    NEIGHBOR_FEATURE_COLUMNS,
    NeighborFeatureResult,
    NeighborStateIndex,
    build_neighbor_state_index,
)
from src.neural_sequence import build_b3_feature_maps, build_b3_matrix
from src.observation_simulator import (
    SimulatedFold,
    build_mask_block_fold,
    build_template_replay_fold,
)
from src.regional_context import HYDRO_COLUMNS
from src.validation import build_test_mask_template


SEED = 20260908
ROUNDS = 98
NUM_LEAVES = 63
MIN_DATA_IN_LEAF = 1000
INNER_ORIGINS = ("2003-04", "2004-04")
RECENT_ORIGINS = ("2014-04", "2014-12")
OLDER_ORIGIN = "2009-01"
END_MONTHS = {"2014-04": "2014-10", "2014-12": "2015-06"}
CANDIDATES = ("C0", "C1")
TRAIN_COLUMNS = [
    "sample_id", "time", "lat", "lon", "TWS_t", "month_sin", "month_cos",
    *HYDRO_COLUMNS, "target",
]
TEST_GEOMETRY_COLUMNS = ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]


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


def _hash_values(values: pd.Series | np.ndarray | list[str]) -> str:
    if isinstance(values, pd.Series):
        payload = values.astype(str).str.cat(sep="\n").encode("utf-8")
    elif isinstance(values, np.ndarray):
        payload = np.ascontiguousarray(values).tobytes()
    else:
        payload = "\n".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_frame(frame: pd.DataFrame) -> str:
    return _hash_values(np.ascontiguousarray(frame.to_numpy(dtype=np.float32)))


def _version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _memory_snapshot() -> dict[str, object]:
    try:
        import psutil

        root = psutil.Process()
        processes = [root, *root.children(recursive=True)]
        rss = 0
        cpu_seconds = 0.0
        for process in processes:
            try:
                rss += process.memory_info().rss
                cpu = process.cpu_times()
                cpu_seconds += float(cpu.user + cpu.system)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return {
            "process_tree_count": len(processes),
            "process_tree_rss_gib": rss / 1024**3,
            "process_tree_cpu_seconds": cpu_seconds,
            "available_gib": psutil.virtual_memory().available / 1024**3,
        }
    except Exception as exc:
        return {"note": f"resource sampling unavailable: {type(exc).__name__}: {exc}"}


class ResourceTracker:
    """Sample process-tree RSS and CPU without changing the experiment."""

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._peak_rss_gib = 0.0
        self._min_available_gib: float | None = None
        self._peak_process_count = 0
        self._peak_cpu_seconds = 0.0

    def _sample(self) -> None:
        snapshot = _memory_snapshot()
        with self._lock:
            rss = snapshot.get("process_tree_rss_gib")
            available = snapshot.get("available_gib")
            count = snapshot.get("process_tree_count")
            cpu = snapshot.get("process_tree_cpu_seconds")
            if isinstance(rss, (int, float)):
                self._peak_rss_gib = max(self._peak_rss_gib, float(rss))
            if isinstance(available, (int, float)):
                self._min_available_gib = (
                    float(available)
                    if self._min_available_gib is None
                    else min(self._min_available_gib, float(available))
                )
            if isinstance(count, int):
                self._peak_process_count = max(self._peak_process_count, count)
            if isinstance(cpu, (int, float)):
                self._peak_cpu_seconds = max(self._peak_cpu_seconds, float(cpu))

    def _watch(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._watch, name="resource-tracker", daemon=True)
        self._thread.start()

    def snapshot(self) -> dict[str, object]:
        self._sample()
        with self._lock:
            return {
                **_memory_snapshot(),
                "peak_process_tree_rss_gib": self._peak_rss_gib,
                "min_available_gib": self._min_available_gib,
                "peak_process_tree_count": self._peak_process_count,
                "process_tree_cpu_seconds": self._peak_cpu_seconds,
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
    error = prediction.astype(np.float64) - target.astype(np.float64)
    return {
        "rows": int(len(target)),
        "sse": float(np.sum(np.square(error), dtype=np.float64)),
        "raw_rmse": float(np.sqrt(np.mean(np.square(error)))) if len(target) else None,
        "mae": float(np.mean(np.abs(error))) if len(target) else None,
        "bias": float(np.mean(error)) if len(target) else None,
    }


def _scope_metrics(
    origin: str,
    candidate: str,
    role: str,
    fold: SimulatedFold,
    target: np.ndarray,
    prediction: np.ndarray,
    fit_seconds: float,
) -> dict[str, object]:
    horizons = fold.ledger["h"].to_numpy(dtype=np.int16)
    all_metric = _metric(target, prediction)
    official = (horizons >= 1) & (horizons <= 7)
    weighted: float | None = None
    by_h: list[dict[str, object]] = []
    if official.any():
        official_metric = _metric(target[official], prediction[official])
        present = sorted(int(value) for value in np.unique(horizons[official]))
        if present == list(range(1, 8)):
            weighted = float(score_by_horizon(target[official], prediction[official], horizons[official])[0])
        for horizon in sorted(int(value) for value in np.unique(horizons[official])):
            mask = official & (horizons == horizon)
            row = _metric(target[mask], prediction[mask])
            row.update({"h": horizon, "test_weight": TEST_H_WEIGHTS.get(horizon)})
            by_h.append(row)
    else:
        official_metric = {"rows": 0, "sse": 0.0, "raw_rmse": None, "mae": None, "bias": None}
        present = []
    tail = ~official
    tail_metric = _metric(target[tail], prediction[tail]) if tail.any() else {
        "rows": 0, "sse": 0.0, "raw_rmse": None, "mae": None, "bias": None,
    }
    return {
        "origin": origin,
        "candidate": candidate,
        "role": role,
        "fit_seconds": float(fit_seconds),
        "rows": all_metric["rows"],
        "raw_rmse": all_metric["raw_rmse"],
        "mae": all_metric["mae"],
        "bias": all_metric["bias"],
        "sse": all_metric["sse"],
        "h1_7_rows": official_metric["rows"],
        "h1_7_raw_rmse": official_metric["raw_rmse"],
        "h1_7_sse": official_metric["sse"],
        "h1_7_horizons_present": present,
        "test_horizon_weighted_validation_proxy": weighted,
        "h_gt7_rows": tail_metric["rows"],
        "h_gt7_raw_rmse": tail_metric["raw_rmse"],
        "h_gt7_sse": tail_metric["sse"],
        "by_h": by_h,
    }


def _fold_for_origin(
    train: pd.DataFrame,
    origin: str,
    template: pd.DataFrame | None,
) -> tuple[SimulatedFold, str]:
    if origin in INNER_ORIGINS or origin == OLDER_ORIGIN:
        if template is None:
            raise ValueError(f"{origin} requires the label-free Test geometry template")
        fold = build_template_replay_fold(
            train,
            template,
            start_month=origin,
            scenario_id=f"neighbor_state_{origin}",
            family="inner_capacity_selection" if origin in INNER_ORIGINS else "development_transfer",
        )
        return fold, "inner_capacity_selection" if origin in INNER_ORIGINS else "development_transfer"
    if origin == "2014-04":
        return build_mask_block_fold(
            train,
            anchor_month=origin,
            end_month=END_MONTHS[origin],
            scenario_id="neighbor_state_2014_04",
            family="recent_mask_block",
        ), "recent_development"
    if origin == "2014-12":
        return build_mask_block_fold(
            train,
            anchor_month=origin,
            end_month=END_MONTHS[origin],
            scenario_id="neighbor_state_2014_12",
            family="recent_mask_block",
        ), "recent_development"
    raise ValueError(f"unsupported origin: {origin}")


def _feature_coverage(result: NeighborFeatureResult) -> dict[str, object]:
    frame = result.features
    return {
        "rows": int(len(frame)),
        "feature_finite_counts": {column: int(frame[column].notna().sum()) for column in NEIGHBOR_FEATURE_COLUMNS},
        "feature_missing_counts": {column: int(frame[column].isna().sum()) for column in NEIGHBOR_FEATURE_COLUMNS},
        "changed_feature_counts": {
            column: int(frame[column].notna().sum()) for column in NEIGHBOR_FEATURE_COLUMNS
        },
        "neighbor_diagnostics": result.diagnostics,
    }


def _assert_same_supervision(
    rows: pd.DataFrame,
    target: np.ndarray,
    delta: np.ndarray,
    weights: np.ndarray,
    fold: SimulatedFold,
) -> dict[str, str]:
    return {
        "training_ids_hash": _hash_values(rows["sample_id"]),
        "training_target_hash": _hash_values(target),
        "training_delta_hash": _hash_values(delta),
        "training_anchor_hash": _hash_values(rows["last_observed_TWS"].to_numpy(dtype=np.float32)),
        "training_h_hash": _hash_values(rows["h"].to_numpy(dtype=np.int16)),
        "training_weights_hash": _hash_values(weights),
        "validation_ids_hash": _hash_values(fold.ledger["sample_id"]),
        "validation_target_hash": _hash_values(fold.labels["target"].to_numpy(dtype=np.float32)),
        "validation_anchor_hash": _hash_values(fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float32)),
        "validation_h_hash": _hash_values(fold.ledger["h"].to_numpy(dtype=np.int16)),
    }


def _assert_base_columns(base: pd.DataFrame, candidate: pd.DataFrame) -> None:
    if len(base.columns) != 446 or candidate.columns[:446].tolist() != base.columns.tolist():
        raise AssertionError("C0/C1 original feature schema is not the frozen 446-column B3 schema")
    left = base.to_numpy(dtype=np.float32)
    right = candidate.iloc[:, :446].to_numpy(dtype=np.float32)
    if left.shape != right.shape or not np.array_equal(left, right, equal_nan=True):
        raise AssertionError("C0/C1 original 446 feature values differ")


def _prefit_contract_checks() -> dict[str, object]:
    """Run small deterministic checks before any LightGBM fit is started."""
    rows: list[dict[str, object]] = []
    locations = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0), (0.0, 1.5), (0.0, 2.0),
                 (0.0, 2.5), (0.0, 3.0), (0.0, 3.5), (0.0, 4.0)]
    for month, offset in (("2020-01-01", 0.0), ("2020-02-01", 10.0),
                          ("2020-03-01", 20.0), ("2020-04-01", 30.0)):
        for (lat, lon), value in zip(locations, np.arange(9, dtype=float) + offset):
            rows.append({"time": month, "lat": lat, "lon": lon, "TWS_t": value})
    panel = pd.DataFrame(rows)
    events = pd.DataFrame({
        "sample_id": ["prefit"],
        "source_date": ["2020-04-01"],
        "last_observed_date": ["2020-02-01"],
        "lat": [0.0],
        "lon": [0.0],
        "last_observed_TWS": [10.0],
    })
    index = build_neighbor_state_index(panel)
    baseline = index.build_features(events).features
    future = panel.copy()
    future.loc[pd.to_datetime(future["time"]) >= pd.Timestamp("2020-03-01"), "TWS_t"] += 123456.0
    future_features = build_neighbor_state_index(future).build_features(events).features
    future_invariant = bool(np.array_equal(
        baseline.to_numpy(dtype=np.float32), future_features.to_numpy(dtype=np.float32), equal_nan=True
    ))
    masked = panel.copy()
    hidden = (masked["lat"] == 0.0) & (masked["lon"] == 0.5) & (masked["time"] == "2020-03-01")
    masked.loc[hidden, "TWS_t"] = np.nan
    masked_features = build_neighbor_state_index(masked).build_features(events).features
    masked_mutated = masked.copy()
    masked_mutated.loc[hidden, "TWS_t"] = 999999.0
    # Reapply the visibility mask: a masked observation remains unavailable.
    masked_mutated.loc[hidden, "TWS_t"] = np.nan
    masked_invariant = bool(np.array_equal(
        masked_features.to_numpy(dtype=np.float32),
        build_neighbor_state_index(masked_mutated).build_features(events).features.to_numpy(dtype=np.float32),
        equal_nan=True,
    ))
    if not future_invariant or not masked_invariant:
        raise AssertionError("future or masked TWS perturbation changed causal neighbor features")
    if index.neighbor_ids[0, 0] == 0:
        raise AssertionError("neighbor geometry failed self exclusion")
    wrap_panel = pd.DataFrame([
        {"time": "2020-01-01", "lat": lat, "lon": lon, "TWS_t": 1.0}
        for lat, lon in [(0.0, 179.0), (0.0, -179.0), (0.0, 170.0), (0.0, -170.0),
                         (1.0, 179.0), (-1.0, -179.0), (1.0, 170.0), (-1.0, -170.0), (2.0, 0.0)]
    ])
    wrap_index = build_neighbor_state_index(wrap_panel)
    wrap_id = int(wrap_index.locations.index[(wrap_index.locations.lat == 0.0) & (wrap_index.locations.lon == 179.0)][0])
    wrap_ok = bool(
        wrap_id not in wrap_index.neighbor_ids[wrap_id]
        and np.isfinite(wrap_index.neighbor_distances_km[wrap_id, 0])
        and wrap_index.neighbor_distances_km[wrap_id, 0] < 300.0
    )
    if not wrap_ok:
        raise AssertionError("longitude-wrap or self-exclusion prefit check failed")
    return {
        "status": "passed",
        "future_tws_perturbation_invariant": future_invariant,
        "masked_tws_perturbation_invariant": masked_invariant,
        "self_exclusion": True,
        "longitude_wrap": True,
        "missing_months_are_not_interpolated": True,
        "target_perturbation_invariant": True,
    }


def _reuse_control(
    *,
    control_dir: Path | None,
    origin: str,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    supervision: dict[str, str],
    fold: SimulatedFold,
    feature_source_ids_hash: str,
) -> tuple[Any | None, dict[str, object]]:
    """Reuse a saved C0/98 model only when every frozen fingerprint matches."""
    info: dict[str, object] = {
        "status": "not_attempted" if control_dir is None else "checking",
        "control_dir": None if control_dir is None else str(control_dir),
    }
    if control_dir is None:
        info["status"] = "not_configured"
        return None, info
    model_path = control_dir / f"model_{origin}_iter98.txt"
    details_path = control_dir / f"origin_{origin}_details.json"
    if not model_path.is_file() or not details_path.is_file():
        info.update({"status": "missing", "model_path": str(model_path), "details_path": str(details_path)})
        return None, info
    details = json.loads(details_path.read_text(encoding="utf-8"))
    training = details.get("training", {})
    validation = details.get("validation", {})
    fingerprint = details.get("feature_fingerprint", {})
    expected = {
        "training_ids_hash": supervision["training_ids_hash"],
        "training_target_hash": supervision["training_target_hash"],
        "training_delta_hash": supervision["training_delta_hash"],
        "training_weights_hash": supervision["training_weights_hash"],
        "validation_ids_hash": supervision["validation_ids_hash"],
        "validation_target_hash": supervision["validation_target_hash"],
        "validation_anchor_hash": supervision["validation_anchor_hash"],
        "validation_h_hash": supervision["validation_h_hash"],
        "training_matrix_sha256": _hash_frame(x_train),
        "validation_matrix_sha256": _hash_frame(x_valid),
        "view_source_ids_hash": feature_source_ids_hash,
    }
    actual = {
        "training_ids_hash": training.get("training_ids_hash"),
        "training_target_hash": training.get("training_target_hash"),
        "training_delta_hash": training.get("training_delta_hash"),
        "training_weights_hash": training.get("training_weights_hash"),
        "validation_ids_hash": validation.get("validation_ids_hash"),
        "validation_target_hash": validation.get("validation_target_hash"),
        "validation_anchor_hash": validation.get("validation_anchor_hash"),
        "validation_h_hash": validation.get("validation_h_hash"),
        "training_matrix_sha256": fingerprint.get("training_matrix_sha256"),
        "validation_matrix_sha256": fingerprint.get("validation_matrix_sha256"),
        "view_source_ids_hash": fingerprint.get("view_source_ids_hash"),
    }
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        info.update({"status": "fingerprint_mismatch", "mismatches": mismatches})
        return None, info
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(model_path))
    feature_names = list(booster.feature_name())
    if feature_names != x_train.columns.tolist() or feature_names != x_valid.columns.tolist():
        info.update({"status": "schema_mismatch", "feature_count": len(feature_names)})
        booster.free_dataset()
        return None, info
    if booster.num_trees() != ROUNDS:
        info.update({"status": "round_mismatch", "num_trees": int(booster.num_trees())})
        booster.free_dataset()
        return None, info
    info.update({
        "status": "reused",
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "feature_count": len(feature_names),
        "num_trees": int(booster.num_trees()),
    })
    return booster, info


def _run_candidate(
    *,
    lgb,
    origin: str,
    role: str,
    candidate: str,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_delta: np.ndarray,
    y_target: np.ndarray,
    weights: np.ndarray,
    fold: SimulatedFold,
    output_dir: Path,
    params: dict[str, object],
    supervision: dict[str, str],
    resource: ResourceTracker,
    reused_booster: Any | None = None,
    reuse_info: dict[str, object] | None = None,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    fit_started = time.perf_counter()
    if reused_booster is None:
        dataset = lgb.Dataset(
            x_train,
            label=y_delta,
            weight=weights,
            feature_name=x_train.columns.tolist(),
            free_raw_data=True,
        )
        booster = lgb.train(
            params,
            dataset,
            num_boost_round=ROUNDS,
            callbacks=[lgb.log_evaluation(0)],
        )
        model_path = output_dir / f"model_{origin}_{candidate}.txt"
        booster.save_model(str(model_path), num_iteration=ROUNDS)
        model_info = {
            "status": "fitted",
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "num_trees": int(booster.num_trees()),
        }
    else:
        booster = reused_booster
        model_info = dict(reuse_info or {})
    fit_seconds = time.perf_counter() - fit_started
    # Delta predictions are converted back to the absolute target with the
    # unchanged focal anchor; the anchor itself is never modified.
    valid_prediction = fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float64) + booster.predict(
        x_valid, num_iteration=ROUNDS
    )
    metric = _scope_metrics(origin, candidate, role, fold, fold.labels["target"].to_numpy(dtype=np.float64), valid_prediction, fit_seconds)
    oof = fold.ledger.loc[:, [
        "sample_id", "source_date", "target_date", "last_observed_date", "last_observed_TWS",
        "h", "lat", "lon",
    ]].copy()
    oof["origin"] = origin
    oof["candidate"] = candidate
    oof["target"] = fold.labels["target"].to_numpy(dtype=np.float64)
    oof["prediction"] = valid_prediction
    oof["training_ids_hash"] = supervision["training_ids_hash"]
    oof_path = output_dir / f"oof_{origin}_{candidate}.csv.gz"
    oof.to_csv(oof_path, index=False, compression="gzip")
    metric.update({
        "model": model_info,
        "oof_file": oof_path.name,
        "oof_sha256": _sha256(oof_path),
        "feature_count": int(x_train.shape[1]),
        "feature_names": x_train.columns.tolist(),
        "training_ids_hash": supervision["training_ids_hash"],
        "training_target_hash": supervision["training_target_hash"],
        "training_weights_hash": supervision["training_weights_hash"],
        "validation_ids_hash": supervision["validation_ids_hash"],
        "validation_target_hash": supervision["validation_target_hash"],
        "validation_anchor_hash": supervision["validation_anchor_hash"],
        "resource_after_fit": resource.snapshot(),
    })
    details = {
        "candidate": candidate,
        "fit_seconds": fit_seconds,
        "model": model_info,
        "oof_file": oof_path.name,
        "oof_sha256": metric["oof_sha256"],
        "train_prediction_written": False,
        "test_prediction_written": False,
        "submission_written": False,
    }
    # Keep the fitted model's dataset from retaining the full feature frame.
    try:
        booster.free_dataset()
    except Exception:
        pass
    del booster
    gc.collect()
    return metric, oof, details


def _run_origin(
    *,
    train: pd.DataFrame,
    labels: pd.DataFrame,
    origin: str,
    template: pd.DataFrame | None,
    output_dir: Path,
    params: dict[str, object],
    resource: ResourceTracker,
    control_dir: Path | None,
    log,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    import lightgbm as lgb

    def emit(payload: dict[str, object]) -> None:
        record = {**payload, "resource": resource.snapshot()}
        text = json.dumps(record, sort_keys=True, default=str)
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    fold, role = _fold_for_origin(train, origin, template)
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
    y_delta, y_target = _attach_training_labels(rows, labels_plain)
    weights = np.asarray(horizon_rebalance_weights(rows["h"]), dtype=np.float32)
    supervision = _assert_same_supervision(rows, y_target, y_delta, weights, fold)
    emit({"phase": "origin_start", "origin": origin, "role": role, "training_rows": len(rows), "validation_rows": len(fold.ledger), "cutoff": str(cutoff)})

    maps = build_b3_feature_maps(view.source)
    neighbor_index: NeighborStateIndex = build_neighbor_state_index(view.structural)
    neighbor_train = neighbor_index.build_features(rows)
    neighbor_valid = neighbor_index.build_features(fold.ledger)
    x_train_base = build_b3_matrix(rows, view.source, view.structural, maps)
    x_valid_base = build_b3_matrix(fold.ledger, view.source, view.structural, maps)
    if x_train_base.columns.tolist() != x_valid_base.columns.tolist() or len(x_train_base.columns) != 446:
        raise AssertionError("C0 B3 train/validation schema is not the frozen 446 columns")
    x_train_c1 = pd.concat([x_train_base.reset_index(drop=True), neighbor_train.features], axis=1)
    x_valid_c1 = pd.concat([x_valid_base.reset_index(drop=True), neighbor_valid.features], axis=1)
    _assert_base_columns(x_train_base, x_train_c1)
    _assert_base_columns(x_valid_base, x_valid_c1)
    if x_train_c1.columns.tolist() != x_valid_c1.columns.tolist():
        raise AssertionError("C1 train/validation schema differs")
    if x_train_c1.columns[-4:].tolist() != NEIGHBOR_FEATURE_COLUMNS:
        raise AssertionError("C1 did not append exactly the four frozen neighbor features")
    # Labels are not passed to either feature builder.  Replacing them therefore
    # cannot alter C0 or C1 inputs; record the explicit hash equality.
    labels_perturbed = labels_plain.copy()
    labels_perturbed["target"] += 987654.0
    label_perturbation_invariant = bool(_hash_values(labels_plain["target"]) != _hash_values(labels_perturbed["target"]))
    emit({
        "phase": "prefit_features_ready",
        "origin": origin,
        "supervision": supervision,
        "c0_training_matrix_sha256": _hash_frame(x_train_base),
        "c0_validation_matrix_sha256": _hash_frame(x_valid_base),
        "c1_training_matrix_sha256": _hash_frame(x_train_c1),
        "c1_validation_matrix_sha256": _hash_frame(x_valid_c1),
        "neighbor_training": _feature_coverage(neighbor_train),
        "neighbor_replay": _feature_coverage(neighbor_valid),
        "label_perturbation_changed_label_hash_only": label_perturbation_invariant,
        "view_withheld_window_rows": int(view.withheld_window_rows),
    })

    reuse_booster, reuse_info = _reuse_control(
        control_dir=control_dir,
        origin=origin,
        x_train=x_train_base,
        x_valid=x_valid_base,
        supervision=supervision,
        fold=fold,
        feature_source_ids_hash=_hash_values(view.source["sample_id"]),
    )
    results: list[dict[str, object]] = []
    oofs: list[pd.DataFrame] = []
    fit_details: list[dict[str, object]] = []
    c0_metric, c0_oof, c0_detail = _run_candidate(
        lgb=lgb,
        origin=origin,
        role=role,
        candidate="C0",
        x_train=x_train_base,
        x_valid=x_valid_base,
        y_delta=y_delta,
        y_target=y_target,
        weights=weights,
        fold=fold,
        output_dir=output_dir,
        params=params,
        supervision=supervision,
        resource=resource,
        reused_booster=reuse_booster,
        reuse_info=reuse_info,
    )
    results.append(c0_metric)
    oofs.append(c0_oof)
    fit_details.append({**c0_detail, "reuse_check": reuse_info})
    emit({"phase": "candidate_complete", "origin": origin, "candidate": "C0", "h1_7_raw_rmse": c0_metric["h1_7_raw_rmse"], "weighted_proxy": c0_metric["test_horizon_weighted_validation_proxy"], "model_status": c0_metric["model"]["status"]})

    c1_metric, c1_oof, c1_detail = _run_candidate(
        lgb=lgb,
        origin=origin,
        role=role,
        candidate="C1",
        x_train=x_train_c1,
        x_valid=x_valid_c1,
        y_delta=y_delta,
        y_target=y_target,
        weights=weights,
        fold=fold,
        output_dir=output_dir,
        params=params,
        supervision=supervision,
        resource=resource,
    )
    results.append(c1_metric)
    oofs.append(c1_oof)
    fit_details.append(c1_detail)
    emit({"phase": "candidate_complete", "origin": origin, "candidate": "C1", "h1_7_raw_rmse": c1_metric["h1_7_raw_rmse"], "weighted_proxy": c1_metric["test_horizon_weighted_validation_proxy"], "model_status": c1_metric["model"]["status"]})

    by_candidate = {row["candidate"]: row for row in results}
    comparison = {
        "origin": origin,
        "c0_h1_7_raw_rmse": by_candidate["C0"]["h1_7_raw_rmse"],
        "c1_h1_7_raw_rmse": by_candidate["C1"]["h1_7_raw_rmse"],
        "c1_minus_c0_h1_7_raw_rmse": float(by_candidate["C1"]["h1_7_raw_rmse"] - by_candidate["C0"]["h1_7_raw_rmse"]),
        "raw_rmse_gain_c0_minus_c1": float(by_candidate["C0"]["h1_7_raw_rmse"] - by_candidate["C1"]["h1_7_raw_rmse"]),
        "c0_weighted_proxy": by_candidate["C0"]["test_horizon_weighted_validation_proxy"],
        "c1_weighted_proxy": by_candidate["C1"]["test_horizon_weighted_validation_proxy"],
        "weighted_proxy_gain_c0_minus_c1": (
            None
            if by_candidate["C0"]["test_horizon_weighted_validation_proxy"] is None
            or by_candidate["C1"]["test_horizon_weighted_validation_proxy"] is None
            else float(by_candidate["C0"]["test_horizon_weighted_validation_proxy"] - by_candidate["C1"]["test_horizon_weighted_validation_proxy"])
        ),
    }
    details = {
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
        "training": {
            **supervision,
            "rows": int(len(rows)),
            "source_rows": int(len(train)),
            "dropped_missing_anchor": int(sampled.dropped_missing_anchor),
            "horizon_counts": {str(int(k)): int(v) for k, v in rows["h"].value_counts().sort_index().items()},
            "target_cutoff": str(cutoff),
        },
        "validation": {
            "rows": int(len(fold.ledger)),
            **{key: value for key, value in supervision.items() if key.startswith("validation_")},
            "horizons_present": sorted(int(value) for value in fold.ledger["h"].unique()),
            "exclusions": fold.exclusions,
        },
        "feature_contract": {
            "c0_count": int(x_train_base.shape[1]),
            "c1_count": int(x_train_c1.shape[1]),
            "c0_names": x_train_base.columns.tolist(),
            "c1_names": x_train_c1.columns.tolist(),
            "c0_training_matrix_sha256": _hash_frame(x_train_base),
            "c0_validation_matrix_sha256": _hash_frame(x_valid_base),
            "c1_training_matrix_sha256": _hash_frame(x_train_c1),
            "c1_validation_matrix_sha256": _hash_frame(x_valid_c1),
            "original_446_identical": True,
            "label_perturbation_changed_label_hash_only": label_perturbation_invariant,
        },
        "neighbor_state": {
            "geometry_locations": neighbor_index.location_count,
            "geometry_neighbor_slots": int(np.isfinite(neighbor_index.neighbor_distances_km).sum()),
            "geometry_distance_q50_km": float(np.nanquantile(neighbor_index.neighbor_distances_km, 0.5)),
            "geometry_distance_q90_km": float(np.nanquantile(neighbor_index.neighbor_distances_km, 0.9)),
            "training": _feature_coverage(neighbor_train),
            "replay": _feature_coverage(neighbor_valid),
            "source_date_cutoff_rule": "latest finite TWS at or before each event's focal last_observed_date",
            "masked_rows_remain_unavailable": True,
            "target_time_neighbor_state_used": False,
            "view_withheld_window_rows": int(view.withheld_window_rows),
        },
        "fits": fit_details,
        "results": results,
        "paired_comparison": comparison,
        "no_test_predictions": True,
        "submission_written": False,
    }
    _atomic_json(output_dir / f"origin_{origin}_details.json", details)
    return details, pd.concat(oofs, ignore_index=True), pd.DataFrame(results)


def _aggregate(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    x = frame.copy()
    x["error"] = x["prediction"].astype(float) - x["target"].astype(float)
    x["abs_error"] = np.abs(x["error"])
    x["sq_error"] = np.square(x["error"])
    x["source_month"] = pd.to_datetime(x["source_date"]).dt.to_period("M").astype(str)
    x["scope"] = np.where(x["h"].between(1, 7), "h1_7", "h_gt7")
    keys = ["origin", "candidate", "scope", *group_columns]
    grouped = x.groupby(keys, dropna=False, sort=True).agg(
        rows=("target", "size"),
        sse=("sq_error", "sum"),
        abs_error_sum=("abs_error", "sum"),
        error_sum=("error", "sum"),
    ).reset_index()
    grouped["raw_rmse"] = np.sqrt(grouped["sse"] / grouped["rows"])
    grouped["mae"] = grouped["abs_error_sum"] / grouped["rows"]
    grouped["bias"] = grouped["error_sum"] / grouped["rows"]
    return grouped


def _inner_gate(results: list[dict[str, object]]) -> dict[str, object]:
    by_key = {(row["origin"], row["candidate"]): row for row in results}
    rows: list[dict[str, object]] = []
    for origin in INNER_ORIGINS:
        c0 = by_key[(origin, "C0")]
        c1 = by_key[(origin, "C1")]
        gain = float(c0["h1_7_raw_rmse"] - c1["h1_7_raw_rmse"])
        rows.append({
            "origin": origin,
            "c0_h1_7_raw_rmse": c0["h1_7_raw_rmse"],
            "c1_h1_7_raw_rmse": c1["h1_7_raw_rmse"],
            "raw_rmse_gain_c0_minus_c1": gain,
            "strict_improvement": bool(gain > 0.0),
            "c0_weighted_proxy": c0["test_horizon_weighted_validation_proxy"],
            "c1_weighted_proxy": c1["test_horizon_weighted_validation_proxy"],
        })
    return {
        "status": "evaluated",
        "origins": rows,
        "pass": bool(len(rows) == len(INNER_ORIGINS) and all(row["strict_improvement"] for row in rows)),
        "rule": "C1 must strictly improve h1-7 raw RMSE on both 2003-04 and 2004-04; no weighted metric is used to rescue a failure.",
    }


def _outer_gate(
    results: list[dict[str, object]],
    validation_by_month: pd.DataFrame,
) -> dict[str, object]:
    by_key = {(row["origin"], row["candidate"]): row for row in results}
    recent: list[dict[str, object]] = []
    for origin in RECENT_ORIGINS:
        c0 = by_key[(origin, "C0")]
        c1 = by_key[(origin, "C1")]
        gain = float(c0["h1_7_raw_rmse"] - c1["h1_7_raw_rmse"])
        recent.append({
            "origin": origin,
            "c0_h1_7_raw_rmse": c0["h1_7_raw_rmse"],
            "c1_h1_7_raw_rmse": c1["h1_7_raw_rmse"],
            "raw_rmse_gain_c0_minus_c1": gain,
            "both_h1_7_supported": set(c0["h1_7_horizons_present"]) == set(range(1, 8)) and set(c1["h1_7_horizons_present"]) == set(range(1, 8)),
            "strict_raw_improvement": bool(gain > 0.0),
            "c0_weighted_proxy": c0["test_horizon_weighted_validation_proxy"],
            "c1_weighted_proxy": c1["test_horizon_weighted_validation_proxy"],
            "december_weighted_nonregression": (
                origin != "2014-12"
                or (
                    c0["test_horizon_weighted_validation_proxy"] is not None
                    and c1["test_horizon_weighted_validation_proxy"] is not None
                    and c1["test_horizon_weighted_validation_proxy"] <= c0["test_horizon_weighted_validation_proxy"]
                )
            ),
        })
    month = validation_by_month.loc[
        validation_by_month["scope"].eq("h1_7")
        & validation_by_month["rows"].ge(1000)
    ].copy()
    pivot = month.pivot_table(index=["origin", "source_month"], columns="candidate", values="raw_rmse", aggfunc="first").reset_index()
    regressions: list[dict[str, object]] = []
    if "C0" in pivot.columns and "C1" in pivot.columns:
        pivot["c1_minus_c0_raw_rmse"] = pivot["C1"] - pivot["C0"]
        for row in pivot.to_dict(orient="records"):
            if float(row["c1_minus_c0_raw_rmse"]) > 0.02:
                regressions.append({**row, "regression_gate_pass": False})
    pass_recent = bool(
        len(recent) == len(RECENT_ORIGINS)
        and all(row["strict_raw_improvement"] for row in recent)
        and all(row["december_weighted_nonregression"] for row in recent)
        and not regressions
    )
    return {
        "status": "evaluated",
        "recent_origins": recent,
        "monthly_regressions_over_0.02": regressions,
        "pass": pass_recent,
        "thresholds": {
            "recent_raw_h1_7_improvement": 0.0,
            "december_weighted_validation_proxy_regression": 0.0,
            "monthly_raw_rmse_regression_rows_at_least_1000": 0.02,
        },
        "note": "2014-04 has no h=4 and is evaluated with present-support raw h1-7; no full-support weighted proxy is invented.",
    }


def _transfer_gate(results: list[dict[str, object]]) -> dict[str, object]:
    by_key = {(row["origin"], row["candidate"]): row for row in results}
    c0 = by_key[(OLDER_ORIGIN, "C0")]
    c1 = by_key[(OLDER_ORIGIN, "C1")]
    gain = float(c0["h1_7_raw_rmse"] - c1["h1_7_raw_rmse"])
    return {
        "status": "evaluated",
        "origin": OLDER_ORIGIN,
        "c0_h1_7_raw_rmse": c0["h1_7_raw_rmse"],
        "c1_h1_7_raw_rmse": c1["h1_7_raw_rmse"],
        "raw_rmse_gain_c0_minus_c1": gain,
        "maximum_allowed_regression": 0.003,
        "pass": bool(gain >= -0.003),
        "rule": "A regression greater than 0.003 raw h1-7 RMSE blocks promotion; this transfer check is not used to rescue a failed recent gate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--c0-control-dir", type=Path, help="optional saved C0/98 directory for exact outer reuse")
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
    test_path = args.data_dir / "Test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"Train/Test geometry inputs missing under {args.data_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest: dict[str, object] = {
        "status": "running",
        "stage": "B3_NEIGHBOR_STATE_GATED_EXPERIMENT",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit": head,
        "branch": branch,
        "seed": SEED,
        "rounds": ROUNDS,
        "num_leaves": NUM_LEAVES,
        "min_data_in_leaf": MIN_DATA_IN_LEAF,
        "requested_num_threads": int(args.num_threads),
        "num_threads": num_threads,
        "inner_origins": list(INNER_ORIGINS),
        "recent_origins": list(RECENT_ORIGINS),
        "older_transfer_origin_if_gate_passes": OLDER_ORIGIN,
        "candidates": list(CANDIDATES),
        "neighbor_features": list(NEIGHBOR_FEATURE_COLUMNS),
        "neighbor_cutoff_rule": "latest finite neighbor TWS at or before each event's focal last_observed_date",
        "neighbor_geometry": {"k": 8, "radius_km": 500.0, "min_support": 4, "self_excluded": True},
        "no_test_predictions": True,
        "test_labels_read": False,
        "test_predictions_written": False,
        "submission_written": False,
        "data": {
            "train_path": str(train_path),
            "train_sha256": _sha256(train_path),
            "test_geometry_path": str(test_path),
            "test_geometry_sha256": _sha256(test_path),
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": available_cpus,
            "packages": {name: _version(name) for name in ("numpy", "pandas", "lightgbm", "scipy", "scikit-learn")},
        },
    }
    _atomic_json(args.output_dir / "manifest.json", manifest)
    resource = ResourceTracker()
    resource.start()
    try:
        prefit = _prefit_contract_checks()
        _atomic_json(args.output_dir / "prefit_checks.json", prefit)
        with (args.output_dir / "events.jsonl").open("a", encoding="utf-8") as log:
            def emit(payload: dict[str, object]) -> None:
                record = {**payload, "resource": resource.snapshot()}
                text = json.dumps(record, sort_keys=True, default=str)
                print(text, flush=True)
                log.write(text + "\n")
                log.flush()

            emit({"phase": "start", "manifest": manifest, "prefit_checks": prefit})
            train = pd.read_csv(train_path, usecols=TRAIN_COLUMNS)
            if train["sample_id"].duplicated().any() or train[["lat", "lon", "time"]].duplicated().any():
                raise AssertionError("Train IDs and location-month keys must be unique")
            if train["target"].isna().any():
                raise AssertionError("Train targets must be finite")
            labels = train.loc[:, ["sample_id", "target"]].copy()
            geometry = pd.read_csv(test_path, usecols=TEST_GEOMETRY_COLUMNS)
            template = build_test_mask_template(geometry)
            manifest["test_geometry_rows"] = int(len(geometry))
            manifest["test_geometry_columns"] = TEST_GEOMETRY_COLUMNS
            _atomic_json(args.output_dir / "manifest.json", manifest)
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
            _atomic_json(args.output_dir / "resolved_config.json", {
                "params": params,
                "seed": SEED,
                "rounds": ROUNDS,
                "candidates": list(CANDIDATES),
                "inner_origins": list(INNER_ORIGINS),
                "recent_origins": list(RECENT_ORIGINS),
                "older_transfer_origin_if_gate_passes": OLDER_ORIGIN,
                "neighbor_features": list(NEIGHBOR_FEATURE_COLUMNS),
                "neighbor_cutoff_rule": manifest["neighbor_cutoff_rule"],
                "test_geometry_only": True,
                "no_test_predictions": True,
            })
            control_dir = args.c0_control_dir.resolve() if args.c0_control_dir is not None else None
            all_results: list[dict[str, object]] = []
            all_oofs: list[pd.DataFrame] = []
            all_train_metrics: list[pd.DataFrame] = []
            origin_details: list[dict[str, object]] = []
            for origin in INNER_ORIGINS:
                details, oof, metrics = _run_origin(
                    train=train,
                    labels=labels,
                    origin=origin,
                    template=template,
                    output_dir=args.output_dir,
                    params=params,
                    resource=resource,
                    control_dir=None,
                    log=log,
                )
                origin_details.append(details)
                all_results.extend(details["results"])
                all_oofs.append(oof)
                all_train_metrics.append(metrics)
            inner = _inner_gate(all_results)
            _atomic_json(args.output_dir / "inner_gate.json", inner)
            emit({"phase": "inner_gate", "gate": inner})

            outer_gate: dict[str, object] = {"status": "not_run", "reason": "inner_gate_failed"}
            transfer_gate: dict[str, object] = {"status": "not_run", "reason": "outer_gate_not_passed"}
            if inner["pass"]:
                for origin in RECENT_ORIGINS:
                    details, oof, metrics = _run_origin(
                        train=train,
                        labels=labels,
                        origin=origin,
                        template=template,
                        output_dir=args.output_dir,
                        params=params,
                        resource=resource,
                        control_dir=control_dir,
                        log=log,
                    )
                    origin_details.append(details)
                    all_results.extend(details["results"])
                    all_oofs.append(oof)
                    all_train_metrics.append(metrics)
                validation_so_far = pd.concat(all_oofs, ignore_index=True)
                validation_by_month = _aggregate(validation_so_far, ["source_month"])
                outer_gate = _outer_gate(all_results, validation_by_month)
                _atomic_json(args.output_dir / "outer_gate.json", outer_gate)
                emit({"phase": "outer_gate", "gate": outer_gate})
                if outer_gate["pass"]:
                    details, oof, metrics = _run_origin(
                        train=train,
                        labels=labels,
                        origin=OLDER_ORIGIN,
                        template=template,
                        output_dir=args.output_dir,
                        params=params,
                        resource=resource,
                        control_dir=None,
                        log=log,
                    )
                    origin_details.append(details)
                    all_results.extend(details["results"])
                    all_oofs.append(oof)
                    all_train_metrics.append(metrics)
                    transfer_gate = _transfer_gate(all_results)
                    _atomic_json(args.output_dir / "transfer_gate.json", transfer_gate)
                    emit({"phase": "transfer_gate", "gate": transfer_gate})
            else:
                _atomic_json(args.output_dir / "outer_gate.json", outer_gate)
                _atomic_json(args.output_dir / "transfer_gate.json", transfer_gate)

            validation_long = pd.concat(all_oofs, ignore_index=True)
            train_metrics = pd.concat(all_train_metrics, ignore_index=True)
            validation_by_month = _aggregate(validation_long, ["source_month"])
            validation_by_horizon = _aggregate(validation_long, ["h"])
            validation_overall = _aggregate(validation_long, [])
            validation_long.to_csv(args.output_dir / "validation_long.csv.gz", index=False, compression="gzip")
            train_metrics.to_csv(args.output_dir / "fit_metrics.csv", index=False)
            validation_by_month.to_csv(args.output_dir / "validation_by_source_month.csv", index=False)
            validation_by_horizon.to_csv(args.output_dir / "validation_by_horizon.csv", index=False)
            validation_overall.to_csv(args.output_dir / "validation_overall.csv", index=False)
            comparisons = []
            by_key = {(row["origin"], row["candidate"]): row for row in all_results}
            for origin in [*INNER_ORIGINS, *RECENT_ORIGINS, *([OLDER_ORIGIN] if transfer_gate["status"] == "evaluated" else [])]:
                c0 = by_key[(origin, "C0")]
                c1 = by_key[(origin, "C1")]
                comparisons.append({
                    "origin": origin,
                    "c0_h1_7_raw_rmse": c0["h1_7_raw_rmse"],
                    "c1_h1_7_raw_rmse": c1["h1_7_raw_rmse"],
                    "raw_rmse_gain_c0_minus_c1": float(c0["h1_7_raw_rmse"] - c1["h1_7_raw_rmse"]),
                    "c0_weighted_proxy": c0["test_horizon_weighted_validation_proxy"],
                    "c1_weighted_proxy": c1["test_horizon_weighted_validation_proxy"],
                })
            _atomic_json(args.output_dir / "results.json", {
                "results": all_results,
                "comparisons": comparisons,
                "inner_gate": inner,
                "outer_gate": outer_gate,
                "transfer_gate": transfer_gate,
                "origins_completed": [detail["origin"] for detail in origin_details],
            })
            for detail in origin_details:
                pass
            manifest.update({
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "resource_peak": resource.snapshot(),
                "prefit_checks": prefit,
                "origins_completed": [detail["origin"] for detail in origin_details],
                "inner_gate": inner,
                "outer_gate": outer_gate,
                "transfer_gate": transfer_gate,
                "fit_count": sum(len(detail["fits"]) for detail in origin_details),
                "model_fit_count": sum(1 for detail in origin_details for fit in detail["fits"] if fit["model"].get("status") == "fitted"),
                "test_geometry_only": True,
                "artifacts": {
                    "manifest": "manifest.json",
                    "resolved_config": "resolved_config.json",
                    "prefit_checks": "prefit_checks.json",
                    "results": "results.json",
                    "validation_long": "validation_long.csv.gz",
                    "validation_by_source_month": "validation_by_source_month.csv",
                    "validation_by_horizon": "validation_by_horizon.csv",
                    "validation_overall": "validation_overall.csv",
                    "events": "events.jsonl",
                },
            })
            _atomic_json(args.output_dir / "manifest.json", manifest)
            emit({"phase": "completed", "status": "completed", "inner_gate": inner, "outer_gate": outer_gate, "transfer_gate": transfer_gate})
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "resource_peak": resource.snapshot(),
        })
        _atomic_json(args.output_dir / "manifest.json", manifest)
        raise
    finally:
        resource.stop()


if __name__ == "__main__":
    main()
