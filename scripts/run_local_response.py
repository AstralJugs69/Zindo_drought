"""Fixed, matched local-response versus R01 experiment; no Test predictions.

The 2014-12 block is a recent stress check, not untouched historical evidence.
Its h=1..7 slice is specified before model scores; the >7 tail is reported apart.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiment import _attach_delta, _coverage_hash, _json, _preflight, _score, _sha256
from scripts.run_lgbm_core import DEFAULT_NUM_THREADS, DEFAULT_PARAMS
from src.local_response import LocalResponse, fit_local_response
from src.ml_features import HYDRO_GAP_SAFE_FEATURE_COLUMNS, SOURCE_CORE_COLUMNS, SOURCE_HYDRO_HISTORY_COLUMNS, build_sampled_training_rows, build_hydro_gap_safe_feature_matrix, horizon_rebalance_weights
from src.observation_simulator import SimulatedFold, build_mask_block_fold, build_template_replay_fold
from src.metrics import raw_rmse
from src.validation import build_test_mask_template


def _config_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_state(expected_commit: str | None) -> tuple[str, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError(f"Refusing to run from a dirty repository checkout:\n{dirty}")
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(f"Commit mismatch: expected {expected_commit}, got {commit}")
    return commit, branch


class _Console:
    """Mirrors concise, flushed runner progress to a durable per-run log."""

    def __init__(self, path: Path):
        self._handle = path.open("w", encoding="utf-8")

    def emit(self, *items: object) -> None:
        line = " ".join(str(item) for item in items)
        print(line, flush=True)
        self._handle.write(line + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=173)
    parser.add_argument("--alpha", type=float, default=30)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--expected-commit", help="Fail closed unless the clean checkout is this full SHA.")
    args = parser.parse_args()
    if args.rounds <= 0 or args.alpha <= 0:
        raise ValueError("rounds and alpha must be positive")
    import lightgbm as lgb
    out = args.output_dir.resolve()
    commit, branch = _git_state(args.expected_commit)
    out.mkdir(parents=True, exist_ok=False)
    params = dict(DEFAULT_PARAMS, num_threads=min(DEFAULT_NUM_THREADS, os.cpu_count() or 1), seed=args.seed, feature_fraction_seed=args.seed,
                  bagging_seed=args.seed, data_random_seed=args.seed)
    config: dict[str, object] = {
        "run_id": out.name, "data_dir": str(args.data_dir.resolve()), "output_dir": str(out),
        "rounds": args.rounds, "alpha": args.alpha, "seed": args.seed,
        "expected_commit": args.expected_commit, "hydro_gap_safe_feature_order": HYDRO_GAP_SAFE_FEATURE_COLUMNS,
        "lightgbm_params": params, "candidates": ["persistence", "r01_lgbm", "global_linear", "local_response", "equal_blend"],
        "origins": ["2007-09", "2009-01", "2014-12"],
        "recent_rule": "score only predeclared h=1..7 IDs; report h>7 tail without clipping",
        "selection": "fixed before scores; no early stopping",
    }
    _json(out / "config.json", config)
    manifest: dict[str, object] = {
        "run_id": out.name, "commit": commit, "branch": branch, "config_hash": _config_hash(config),
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "seed": args.seed, "rounds": args.rounds, "alpha": args.alpha,
        "lightgbm_params": params, "feature_schema": list(HYDRO_GAP_SAFE_FEATURE_COLUMNS),
        "resolved_config": config,
    }
    _json(out / "manifest.json", manifest)
    console = _Console(out / "console.log")
    started = time.perf_counter()
    try:
        manifest["preflight"] = _preflight(args.data_dir)
        train = pd.read_csv(args.data_dir / "Train.csv")
        test = pd.read_csv(args.data_dir / "Test.csv")
        template = build_test_mask_template(test)
        structural = train[["sample_id", "time", "lat", "lon", "TWS_t"]]
        source = train[SOURCE_HYDRO_HISTORY_COLUMNS]
        labels = train[["sample_id", "target"]]
        summaries: list[dict[str, object]] = []
        recent_report: dict[str, object] | None = None
        training_counts: dict[str, int] = {}
        scenario_hashes: dict[str, str] = {}
        coverage_hashes: dict[str, str] = {}
        for origin in ["2007-09", "2009-01", "2014-12"]:
            console.emit("FOLD_START", origin)
            if origin == "2014-12":
                full = build_mask_block_fold(train, anchor_month=origin, end_month="2015-06", scenario_id="recent_2014_12")
                keep = full.ledger.h.between(1, 7).to_numpy()
                tail = full.ledger.loc[~keep].copy()
                tail["truth"] = full.labels.loc[~keep, "target"].to_numpy()
                tail.to_csv(out / "recent_h_gt7_tail.csv.gz", index=False, compression="gzip")
                excluded = dict(full.exclusions, outside_competition_horizon=int((~keep).sum()))
                fold = SimulatedFold(full.spec, full.ledger.loc[keep].reset_index(drop=True), full.labels.loc[keep].reset_index(drop=True), excluded)
                full_anchor = full.ledger.last_observed_TWS.to_numpy(dtype=np.float64)
                tail_anchor = full_anchor[~keep]
                tail_truth = full.labels.loc[~keep, "target"].to_numpy(dtype=np.float64)
                recent_report = {
                    "full_rows": int(len(full.ledger)), "scored_h1_to_h7_rows": int(keep.sum()),
                    "tail_h_gt7_rows": int((~keep).sum()), "tail_horizon_counts": {
                        str(k): int(v) for k, v in full.ledger.loc[~keep, "h"].value_counts().sort_index().items()
                    },
                    "full_persistence_raw_rmse": raw_rmse(full.labels["target"], full_anchor),
                    "tail_persistence_raw_rmse": raw_rmse(tail_truth, tail_anchor),
                    "tail_persistence_bias": float(np.mean(tail_anchor - tail_truth)),
                    "tail_persistence_mae": float(np.mean(np.abs(tail_anchor - tail_truth))),
                }
            else:
                fold = build_template_replay_fold(train, template, start_month=origin, scenario_id=f"replay_{origin}", family="development")
            sampled = build_sampled_training_rows(structural, source[SOURCE_CORE_COLUMNS], max_target_month=pd.Period(origin, "M") - 1)
            rows = sampled.rows
            training_counts[origin] = len(rows)
            scenario_hashes[origin] = fold.spec.digest()
            coverage_hashes[origin] = _coverage_hash(fold.ledger)
            y = _attach_delta(rows, labels)
            w = horizon_rebalance_weights(rows.h)
            x = build_hydro_gap_safe_feature_matrix(rows, source, structural)
            v = build_hydro_gap_safe_feature_matrix(fold.ledger, source, structural)
            console.emit("TRAIN_ROWS", len(rows), "VALID_ROWS", len(v), "SPEC", fold.spec.digest())
            booster = lgb.train(params, lgb.Dataset(x, label=y, weight=w), num_boost_round=args.rounds)
            booster.save_model(str(out / f"lgbm_{origin}.txt"))
            console.emit("LGBM_FIT_DONE", origin)
            local = fit_local_response(x, y, w, alpha=args.alpha)
            local.save(out / f"local_{origin}.npz")
            np.testing.assert_allclose(local.predict(v.iloc[:100]), LocalResponse.load(out / f"local_{origin}.npz").predict(v.iloc[:100]), atol=1e-10)
            anchor = fold.ledger.last_observed_TWS.to_numpy(dtype=np.float64)
            gb = anchor + booster.predict(v)
            lr = anchor + local.predict(v)
            candidates = {"persistence": anchor, "r01_lgbm": gb,
                          "global_linear": anchor + local.predict(v, local=False),
                          "local_response": lr, "equal_blend": (gb + lr) / 2}
            for name, pred in candidates.items():
                score, oof = _score(name, fold, "recent_stress" if origin == "2014-12" else "development", pred)
                score.update(candidate=name, origin=origin, training_rows=len(rows))
                summaries.append(score)
                oof.to_csv(out / f"oof_{origin}_{name}.csv.gz", index=False, compression="gzip")
                console.emit("RESULT", json.dumps({k: score[k] for k in ["origin", "candidate", "rows", "raw_rmse", "weighted_rmse", "bias"]}, sort_keys=True))
            _json(out / "metrics.json", {"folds": summaries, "recent_stress": recent_report})
            del x, v, y, w, rows, sampled, local, booster, candidates, fold
            gc.collect()
        manifest.update(
            status="completed",
            elapsed_seconds=time.perf_counter() - started,
            completed_at=datetime.now(timezone.utc).isoformat(),
            training_counts=training_counts,
            scenario_hashes=scenario_hashes,
            coverage_hashes=coverage_hashes,
            recent_stress=recent_report,
        )
        manifest["output_checksums"] = {p.name: _sha256(p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json"}
        console.emit("RUN_COMPLETE", str(out))
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        console.emit("RUN_FAILED", type(exc).__name__, str(exc))
        raise
    finally:
        _json(out / "manifest.json", manifest)
        console.close()


if __name__ == "__main__":
    main()
