"""Paired causal local/regional hydrological trajectory ablation (Kaggle only)."""
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
import threading
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiment import _attach_delta, _score
from scripts.run_lgbm_core import DEFAULT_NUM_THREADS, DEFAULT_PARAMS
from src.hydro_trajectory import (
    attach_trajectory_features,
    build_hydro_trajectory_map,
    build_regional_trajectory_map,
)
from src.ml_features import (
    SOURCE_CORE_COLUMNS,
    SOURCE_HYDRO_HISTORY_COLUMNS,
    build_hydro_gap_safe_feature_matrix,
    build_sampled_training_rows,
    horizon_rebalance_weights,
)
from src.observation_simulator import SimulatedFold, build_mask_block_fold, build_template_replay_fold
from src.regional_context import HYDRO_COLUMNS, attach_regional_features, build_regional_context, regional_bytes_hash
from src.validation import build_test_mask_template

SEED = 20260908
ROUNDS = 98
OUTER_ORIGINS = ("2007-09", "2009-01", "2014-04", "2014-12")
INNER_ORIGINS = ("2003-04", "2004-04")
ALL_ORIGINS = (*INNER_ORIGINS, *OUTER_ORIGINS)
CANDIDATES = ("B0", "B1_local", "B2_regional", "B3_both")


def _json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _memory() -> dict[str, object]:
    try:
        import psutil
        process = psutil.Process()
        return {
            "rss_gib": process.memory_info().rss / 1024**3,
            "available_gib": psutil.virtual_memory().available / 1024**3,
        }
    except Exception as exc:
        return {"note": f"memory unavailable: {type(exc).__name__}: {exc}"}


class MemoryTracker:
    """Continuously retain peak resident memory without retaining samples."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._note: str | None = None
        self._process = None
        self._psutil = None
        self._peak_rss_gib: float | None = None
        self._min_available_gib: float | None = None
        try:
            import psutil
            self._psutil = psutil
            self._process = psutil.Process()
        except Exception as exc:
            self._note = f"memory unavailable: {type(exc).__name__}: {exc}"

    def _sample(self) -> None:
        if self._process is None or self._psutil is None:
            return
        try:
            rss_gib = self._process.memory_info().rss / 1024**3
            available_gib = self._psutil.virtual_memory().available / 1024**3
            with self._lock:
                self._peak_rss_gib = max(self._peak_rss_gib or 0.0, rss_gib)
                self._min_available_gib = available_gib if self._min_available_gib is None else min(
                    self._min_available_gib, available_gib
                )
        except Exception as exc:
            self._note = f"memory sampling unavailable: {type(exc).__name__}: {exc}"

    def _watch(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._sample()
        if self._process is not None:
            self._thread = threading.Thread(target=self._watch, name="memory-tracker", daemon=True)
            self._thread.start()

    def snapshot(self) -> dict[str, object]:
        self._sample()
        if self._note is not None:
            return {"note": self._note}
        current = _memory()
        with self._lock:
            return {
                **current,
                "peak_rss_gib": self._peak_rss_gib,
                "min_available_gib": self._min_available_gib,
            }

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 2)


def _fold(
    train: pd.DataFrame,
    template: pd.DataFrame,
    origin: str,
    *,
    include_stress_tail: bool = False,
) -> tuple[SimulatedFold, str]:
    """Build the declared replay fold.

    The default preserves the historical competition-horizon slice for the
    December-2014 stress block.  Gap-augmentation diagnostics may opt into the
    complete block so that h>7 rows remain available as a separately reported
    stress tail; this never changes the fitting cutoff or the official h=1..7
    score.
    """
    if origin in INNER_ORIGINS:
        return build_template_replay_fold(
            train, template, start_month=origin,
            scenario_id=f"trajectory_inner_{origin}", family="inner_capacity_selection",
        ), "inner_capacity_selection"
    if origin == "2014-04":
        return build_mask_block_fold(
            train, anchor_month=origin, end_month="2014-10",
            scenario_id="trajectory_recent_2014_04", family="recent_mask_block",
        ), "recent_development"
    if origin == "2014-12":
        full = build_mask_block_fold(
            train, anchor_month=origin, end_month="2015-06",
            scenario_id="trajectory_recent_2014_12", family="recent_mask_block",
        )
        if include_stress_tail:
            return full, "recent_development"
        keep = full.ledger.h.between(1, 7).to_numpy()
        return SimulatedFold(
            full.spec,
            full.ledger.loc[keep].reset_index(drop=True),
            full.labels.loc[keep].reset_index(drop=True),
            dict(full.exclusions, outside_competition_horizon=int((~keep).sum())),
        ), "recent_development"
    return build_template_replay_fold(
        train, template, start_month=origin,
        scenario_id=f"trajectory_{origin}", family="development",
    ), "development"


def _feature_frame(
    candidate: str,
    ledger: pd.DataFrame,
    source: pd.DataFrame,
    structural: pd.DataFrame,
    regional: pd.DataFrame,
    local_trajectory: pd.DataFrame,
    regional_trajectory: pd.DataFrame,
) -> pd.DataFrame:
    base = build_hydro_gap_safe_feature_matrix(ledger, source, structural)
    current_regional = attach_regional_features(ledger.sample_id, regional)
    blocks = [base.reset_index(drop=True), current_regional.reset_index(drop=True)]
    if candidate in {"B1_local", "B3_both"}:
        blocks.append(attach_trajectory_features(ledger.sample_id, local_trajectory))
    if candidate in {"B2_regional", "B3_both"}:
        blocks.append(attach_trajectory_features(ledger.sample_id, regional_trajectory))
    out = pd.concat(blocks, axis=1)
    if out.columns.duplicated().any():
        raise AssertionError(f"{candidate} feature schema has duplicate names")
    if np.isinf(out.to_numpy(dtype=np.float64)).any():
        raise AssertionError(f"{candidate} feature matrix contains infinite values")
    return out.reset_index(drop=True)


def _hash_ids(sample_ids: pd.Series) -> str:
    return hashlib.sha256(sample_ids.astype(str).str.cat(sep="\n").encode("utf-8")).hexdigest()


def _emit(log, payload: object) -> None:
    text = json.dumps(payload, sort_keys=True)
    print(text, flush=True)
    log.write(text + "\n")
    log.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal hydrological trajectory B0-B3 ablations on Kaggle.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--origins", nargs="+", choices=ALL_ORIGINS, default=list(OUTER_ORIGINS))
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args()
    if len(set(args.origins)) != len(args.origins) or len(set(args.candidates)) != len(args.candidates):
        raise ValueError("origins and candidates must not contain duplicates")

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
        "status": "running", "stage": "HYDROLOGICAL_TRAJECTORY_ABLATION", "commit": head,
        "started_at": datetime.now(timezone.utc).isoformat(), "no_test_predictions": True,
        "seed": SEED, "rounds": args.rounds, "origins": args.origins, "candidates": args.candidates,
        "python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
    }
    _json(args.output_dir / "manifest.json", manifest)
    results: list[dict[str, object]] = []
    memory_tracker = MemoryTracker()
    memory_tracker.start()
    try:
        with (args.output_dir / "console.log").open("w", encoding="utf-8") as log:
            _emit(log, {"phase": "start", "commit": head, "memory": memory_tracker.snapshot()})
            import lightgbm as lgb

            train_cols = ["sample_id", "time", "lat", "lon", "TWS_t", "target", "month_sin", "month_cos", *HYDRO_COLUMNS]
            test_cols = ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked", *HYDRO_COLUMNS]
            train = pd.read_csv(args.data_dir / "Train.csv", usecols=train_cols)
            test = pd.read_csv(args.data_dir / "Test.csv", usecols=test_cols)
            template = build_test_mask_template(test[["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]])
            structural = train[["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
            source = train.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
            labels = train[["sample_id", "target"]].copy()
            regional_source = train[["sample_id", "time", "lat", "lon", *HYDRO_COLUMNS]].copy()

            _emit(log, {"phase": "build_causal_trajectory_maps", "rows": int(len(source)), "memory": memory_tracker.snapshot()})
            regional = build_regional_context(regional_source)
            local_trajectory = build_hydro_trajectory_map(regional_source)
            regional_trajectory = build_regional_trajectory_map(regional_source, regional, local_trajectory)
            _emit(log, {"phase": "trajectory_maps_ready", "local_features": int(local_trajectory.shape[1] - 1),
                        "regional_features": int(regional_trajectory.shape[1] - 1), "memory": memory_tracker.snapshot()})

            # Structural Test parity only: construct every feature schema but never fit/predict Test rows.
            test_ledger = test[["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]].copy()
            from src.availability import build_test_availability_ledger
            test_ledger = build_test_availability_ledger(test_ledger).rename(columns={"ID": "sample_id"})
            test_source = test.rename(columns={"ID": "sample_id"})[["sample_id", "time", "lat", "lon", *HYDRO_COLUMNS]].copy()
            test_month = pd.to_datetime(test_source.time).dt.month.to_numpy(dtype=np.float64)
            test_source["month_sin"] = np.sin(2.0 * np.pi * test_month / 12.0).astype(np.float32)
            test_source["month_cos"] = np.cos(2.0 * np.pi * test_month / 12.0).astype(np.float32)
            test_structural = test.rename(columns={"ID": "sample_id"})[["sample_id", "time", "lat", "lon", "TWS_t"]]
            test_regional_source = test_source[["sample_id", "time", "lat", "lon", *HYDRO_COLUMNS]]
            test_regional = build_regional_context(test_regional_source)
            test_local = build_hydro_trajectory_map(test_regional_source)
            test_regional_trajectory = build_regional_trajectory_map(test_regional_source, test_regional, test_local)
            test_contract = {}
            for candidate in args.candidates:
                matrix = _feature_frame(candidate, test_ledger, test_source, test_structural, test_regional, test_local, test_regional_trajectory)
                test_contract[candidate] = {"feature_count": int(matrix.shape[1]), "feature_names": matrix.columns.tolist(),
                                            "finite_or_missing": bool(~np.isinf(matrix.to_numpy(dtype=np.float64)).any())}
            _json(args.output_dir / "test_feature_contract.json", test_contract)
            del test_local, test_regional_trajectory, test_regional, test_source, test_structural, test_ledger, test

            params = dict(DEFAULT_PARAMS, num_leaves=63, min_data_in_leaf=1000, num_threads=min(DEFAULT_NUM_THREADS, os.cpu_count() or 1),
                          seed=SEED, feature_fraction_seed=SEED, bagging_seed=SEED, data_random_seed=SEED)
            for origin in args.origins:
                fold, role = _fold(train, template, origin)
                start = pd.Period(origin, freq="M")
                sampled = build_sampled_training_rows(structural, source[SOURCE_CORE_COLUMNS], max_target_month=start - 1, seed=SEED)
                rows = sampled.rows
                y_train = _attach_delta(rows, labels)
                weights = horizon_rebalance_weights(rows.h)
                matrices_train = {candidate: _feature_frame(candidate, rows, source, structural, regional, local_trajectory, regional_trajectory)
                                  for candidate in args.candidates}
                matrices_valid = {candidate: _feature_frame(candidate, fold.ledger, source, structural, regional, local_trajectory, regional_trajectory)
                                  for candidate in args.candidates}
                row_hash = _hash_ids(rows.sample_id)
                coverage_hash = _hash_ids(fold.ledger.sample_id)
                label_hash = hashlib.sha256(np.ascontiguousarray(y_train).tobytes()).hexdigest()
                _emit(log, {"phase": "origin_features_ready", "origin": origin, "role": role, "training_rows": int(len(rows)),
                            "validation_rows": int(len(fold.ledger)), "row_hash": row_hash, "coverage_hash": coverage_hash,
                            "h_counts": {str(k): int(v) for k, v in fold.ledger.h.value_counts().sort_index().items()},
                            "memory": memory_tracker.snapshot()})
                for candidate in args.candidates:
                    fit_started = time.perf_counter()
                    x_train, x_valid = matrices_train[candidate], matrices_valid[candidate]
                    if test_contract[candidate]["feature_names"] != x_train.columns.tolist() or x_train.columns.tolist() != x_valid.columns.tolist():
                        raise AssertionError(f"{candidate} train/validation/Test feature schemas differ")
                    model = lgb.train(params, lgb.Dataset(x_train, label=y_train, weight=weights, feature_name=list(x_train.columns)),
                                      num_boost_round=args.rounds, callbacks=[lgb.log_evaluation(0)])
                    prediction = fold.ledger.last_observed_TWS.to_numpy(dtype=np.float64) + model.predict(x_valid)
                    score, oof = _score(candidate, fold, role, prediction)
                    model_path = args.output_dir / f"model_{origin}_{candidate}.txt"
                    oof_path = args.output_dir / f"oof_{origin}_{candidate}.csv.gz"
                    model.save_model(str(model_path))
                    oof["training_row_hash"] = row_hash
                    oof.to_csv(oof_path, index=False, compression="gzip")
                    score.update({
                        "origin": origin, "candidate": candidate, "rounds": args.rounds,
                        "training_rows": int(len(rows)), "training_row_hash": row_hash,
                        "training_label_hash": label_hash, "coverage_hash": coverage_hash,
                        "feature_count": int(x_train.shape[1]), "feature_names": x_train.columns.tolist(),
                        "regional_context_hash": regional_bytes_hash(attach_regional_features(rows.sample_id, regional)),
                        "elapsed_seconds": time.perf_counter() - fit_started,
                        "model_file": model_path.name, "oof_file": oof_path.name,
                    })
                    results.append(score)
                    _json(args.output_dir / "metrics.partial.json", {"results": results})
                    _emit(log, {"phase": "fit_complete", "origin": origin, "candidate": candidate,
                                "raw_rmse": score["raw_rmse"], "weighted_rmse": score["weighted_rmse"],
                                "elapsed_seconds": score["elapsed_seconds"], "memory": memory_tracker.snapshot()})
                    model.free_dataset()
                del matrices_train, matrices_valid

            comparisons = []
            by_key = {(r["origin"], r["candidate"]): r for r in results}
            if "B0" in args.candidates:
                for origin in args.origins:
                    baseline = by_key[(origin, "B0")]
                    for candidate in args.candidates:
                        if candidate == "B0":
                            continue
                        candidate_row = by_key[(origin, candidate)]
                        comparisons.append({
                            "origin": origin, "candidate": candidate,
                            "raw_rmse_gain_vs_B0": float(baseline["raw_rmse"] - candidate_row["raw_rmse"]),
                            "weighted_rmse_gain_vs_B0": float(baseline["weighted_rmse"] - candidate_row["weighted_rmse"]),
                        })
            _json(args.output_dir / "metrics.json", {"results": results, "paired_comparisons": comparisons,
                                                       "test_contract": test_contract})
            manifest.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
                             "elapsed_seconds": time.perf_counter() - started, "result_rows": len(results),
                             "memory_peak": memory_tracker.snapshot(),
                             "output_checksums": {str(p.relative_to(args.output_dir)): _sha256(p)
                                                  for p in args.output_dir.rglob("*") if p.is_file() and p.name != "manifest.json"}})
            _json(args.output_dir / "manifest.json", manifest)
            package = shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir)
            manifest["package"] = {"path": package, "bytes": Path(package).stat().st_size, "sha256": _sha256(Path(package))}
            _json(args.output_dir / "manifest.json", manifest)
            _emit(log, {"status": "completed", "output_dir": str(args.output_dir), "package": package, "results": len(results)})
    except Exception as exc:
        manifest.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(),
                         "elapsed_seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}",
                         "memory_peak": memory_tracker.snapshot()})
        _json(args.output_dir / "manifest.json", manifest)
        raise
    finally:
        memory_tracker.stop()


if __name__ == "__main__":
    main()
