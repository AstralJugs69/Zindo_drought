"""Artifact-first experiment entry point for the Kaggle notebook runner.

Initially this provides the gated non-training preflight and R00 persistence
baseline.  It never imports LightGBM, so it is safe to run before model work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import predict_persistence
from src.metrics import raw_rmse, score_by_horizon
from src.observation_simulator import build_mask_block_fold, build_template_replay_fold
from src.validation import build_test_mask_template, find_exact_template_starts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _preflight(data_dir: Path) -> dict[str, object]:
    required = [data_dir / name for name in ("Train.csv", "Test.csv", "SampleSubmission.csv")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required CSVs not found: {missing}")
    headers = {path.name: pd.read_csv(path, nrows=3).columns.tolist() for path in required}
    sample = pd.read_csv(data_dir / "SampleSubmission.csv", nrows=3)
    if sample.columns.tolist() != ["ID", "Target"]:
        raise AssertionError("SampleSubmission schema must be exactly ID,Target")
    try:
        import psutil  # Kaggle normally includes it.
        memory = psutil.virtual_memory()._asdict()
    except ImportError:
        memory = {"available": None, "note": "psutil unavailable"}
    version_names = ("numpy", "pandas", "lightgbm", "pyarrow")
    packages: dict[str, str | None] = {}
    for name in version_names:
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory": memory,
        "disk": shutil.disk_usage(data_dir)._asdict(),
        "packages": packages,
        "csv_headers": headers,
        "csv_hashes": {path.name: _sha256(path) for path in required},
    }


def _score_persistence(fold, role: str) -> tuple[dict[str, object], pd.DataFrame]:
    y = fold.labels["target"].to_numpy(dtype=np.float64)
    p = predict_persistence(fold.ledger)
    present_h = sorted(int(v) for v in fold.ledger["h"].unique())
    weighted = None
    by_h = None
    if present_h == list(range(1, 8)):
        weighted, horizon = score_by_horizon(y, p, fold.ledger["h"])
        by_h = horizon.reset_index().to_dict(orient="records")
    row = {
        "scenario_id": fold.spec.scenario_id, "family": fold.spec.family, "role": role,
        "spec_hash": fold.spec.digest(), "training_target_cutoff": fold.spec.training_target_cutoff,
        "rows": int(len(fold.ledger)), "horizons_present": present_h,
        "raw_rmse": raw_rmse(y, p), "weighted_rmse": weighted,
        "exclusions": fold.exclusions, "by_h": by_h,
    }
    oof = pd.DataFrame({
        "sample_id": fold.ledger["sample_id"], "scenario_id": fold.spec.scenario_id,
        "source_date": fold.ledger["source_date"], "h": fold.ledger["h"],
        "anchor_date": fold.ledger["last_observed_date"], "anchor_tws": fold.ledger["last_observed_TWS"],
        "truth": y, "prediction": p, "model": "R00_persistence", "role": role,
    })
    return row, oof


def _r00(data_dir: Path, out_dir: Path) -> dict[str, object]:
    columns = ["sample_id", "time", "lat", "lon", "TWS_t", "target"]
    train = pd.read_csv(data_dir / "Train.csv", usecols=columns)
    test = pd.read_csv(data_dir / "Test.csv", usecols=["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"])
    template = build_test_mask_template(test)
    starts = find_exact_template_starts(train, template)
    september = [start for start in starts if start.month == 9]
    if len(september) < 2:
        raise AssertionError(f"Expected at least two September template starts, found {september}")
    replay_specs = [
        ("exact_latest", starts[-1], "exact_template_replay", "development"),
        ("season_aligned_sep_a", september[-2], "season_aligned_template_replay", "development"),
        ("season_aligned_sep_b", september[-1], "season_aligned_template_replay", "development"),
    ]
    # This is a declared stress/coverage scenario, not a promotion metric. Missing
    # historical source months may make h exceed seven; it is reported verbatim.
    stress_specs = [("confirm_2014_12_to_2015_06", "2014-12", "2015-06", "confirmation")]
    rows: list[dict[str, object]] = []
    oof: list[pd.DataFrame] = []
    for scenario_id, start, family, role in replay_specs:
        fold = build_template_replay_fold(train, template, start_month=start, scenario_id=scenario_id, family=family)
        row, frame = _score_persistence(fold, role)
        rows.append(row)
        oof.append(frame)
    for scenario_id, anchor, end, role in stress_specs:
        fold = build_mask_block_fold(train, anchor_month=anchor, end_month=end, scenario_id=scenario_id)
        row, frame = _score_persistence(fold, role)
        rows.append(row)
        oof.append(frame)
    oof_frame = pd.concat(oof, ignore_index=True)
    oof_frame.to_csv(out_dir / "oof.csv.gz", index=False, compression="gzip")
    return {"candidate": "R00_persistence", "scenarios": rows, "oof": "oof.csv.gz"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gated drought validation experiments with preserved artifacts.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("preflight", "r00"), required=True)
    args = parser.parse_args()
    data_dir, out_dir = args.data_dir.resolve(), args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip():
        raise RuntimeError("Refusing to run from a dirty repository checkout")
    manifest = {"run_id": args.run_id, "mode": args.mode, "started_at": started, "commit": head, "branch": branch}
    _json(out_dir / "config.json", {
        "data_dir": str(data_dir), "output_dir": str(out_dir),
        "run_id": args.run_id, "mode": args.mode,
    })
    try:
        manifest["preflight"] = _preflight(data_dir)
        if args.mode == "r00":
            metrics = _r00(data_dir, out_dir)
        else:
            metrics = {"status": "preflight_complete"}
        _json(out_dir / "metrics.json", metrics)
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        _json(out_dir / "manifest.json", manifest)
        print(json.dumps({"run_id": args.run_id, "status": "completed", "output_dir": str(out_dir)}, indent=2))
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _json(out_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
