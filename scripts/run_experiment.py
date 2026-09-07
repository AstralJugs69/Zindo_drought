"""Artifact-first runner for the repaired causal validation program.

R00 is a no-fit legal persistence baseline. R01 keeps the old independent
horizon sampler as a comparator; R02 replaces it with transplanted historical
observation schedules; R03 adds simulator-derived recent visible history.
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
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_lgbm_core import DEFAULT_PARAMS
from src.baselines import predict_persistence
from src.metrics import raw_rmse, score_by_horizon
from src.ml_features import (
    HYDRO_GAP_SAFE_FEATURE_COLUMNS, SOURCE_CORE_COLUMNS,
    SOURCE_HYDRO_HISTORY_COLUMNS, VISIBLE_HISTORY_FEATURE_COLUMNS,
    build_hydro_gap_safe_feature_matrix, build_sampled_training_rows,
    build_visible_history_feature_matrix, horizon_rebalance_weights,
)
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
    if headers["SampleSubmission.csv"] != ["ID", "Target"]:
        raise AssertionError("SampleSubmission schema must be exactly ID,Target")
    try:
        import psutil
        memory = psutil.virtual_memory()._asdict()
    except ImportError:
        memory = {"available": None, "note": "psutil unavailable"}
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "lightgbm", "pyarrow"):
        try:
            packages[name] = getattr(__import__(name), "__version__", "unknown")
        except ImportError:
            packages[name] = None
    return {
        "python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "memory": memory, "disk": shutil.disk_usage(data_dir)._asdict(), "packages": packages,
        "csv_headers": headers, "csv_hashes": {path.name: _sha256(path) for path in required},
    }


def _coverage_hash(ledger: pd.DataFrame) -> str:
    values = ledger["sample_id"].astype(str).sort_values().str.cat(sep="\n")
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


def _slice_metrics(frame: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    x = frame.copy()
    x["abs_error"] = np.abs(x["prediction"] - x["truth"])
    x["sq_error"] = np.square(x["prediction"] - x["truth"])
    x["error"] = x["prediction"] - x["truth"]
    x["source_month"] = pd.to_datetime(x["source_date"]).dt.month.astype(int)
    x["region"] = ((np.floor(x["lat"] / 10) * 10).astype(int).astype(str) + ":" +
                   (np.floor(x["lon"] / 10) * 10).astype(int).astype(str))
    try:
        x["target_quartile"] = pd.qcut(x["truth"], q=4, duplicates="drop").astype(str)
    except ValueError:
        x["target_quartile"] = "all"

    def summarize(column: str) -> list[dict[str, object]]:
        grouped = x.groupby(column, observed=True).agg(
            rows=("truth", "size"), mse=("sq_error", "mean"), mae=("abs_error", "mean"), bias=("error", "mean")
        ).reset_index()
        grouped["rmse"] = np.sqrt(grouped.pop("mse"))
        return grouped.to_dict(orient="records")

    return {"by_source_month": summarize("source_month"), "by_region": summarize("region"),
            "by_target_range": summarize("target_quartile")}


def _score(name: str, fold, role: str, prediction: np.ndarray) -> tuple[dict[str, object], pd.DataFrame]:
    y = fold.labels["target"].to_numpy(dtype=np.float64)
    h = fold.ledger["h"]
    present_h = sorted(int(v) for v in h.unique())
    weighted, by_h = (None, None)
    if present_h == list(range(1, 8)):
        weighted, horizon = score_by_horizon(y, prediction, h)
        by_h = horizon.reset_index().to_dict(orient="records")
    oof = pd.DataFrame({
        "sample_id": fold.ledger["sample_id"], "scenario_id": fold.spec.scenario_id,
        "source_date": fold.ledger["source_date"], "h": h, "anchor_date": fold.ledger["last_observed_date"],
        "anchor_tws": fold.ledger["last_observed_TWS"], "lat": fold.ledger["lat"], "lon": fold.ledger["lon"],
        "truth": y, "prediction": prediction, "model": name, "role": role,
    })
    row: dict[str, object] = {
        "scenario_id": fold.spec.scenario_id, "family": fold.spec.family, "role": role,
        "spec_hash": fold.spec.digest(), "coverage_hash": _coverage_hash(fold.ledger),
        "training_target_cutoff": fold.spec.training_target_cutoff, "rows": int(len(oof)),
        "horizons_present": present_h, "raw_rmse": raw_rmse(y, prediction), "weighted_rmse": weighted,
        "mae": float(np.mean(np.abs(prediction - y))), "bias": float(np.mean(prediction - y)),
        "exclusions": fold.exclusions, "by_h": by_h, **_slice_metrics(oof),
    }
    return row, oof


def _development_folds(train: pd.DataFrame, template: pd.DataFrame):
    starts = find_exact_template_starts(train, template)
    september = [start for start in starts if start.month == 9]
    if len(september) < 2:
        raise AssertionError(f"Expected at least two September template starts, found {september}")
    definitions = [
        ("exact_latest", starts[-1], "exact_template_replay", "development"),
        ("season_aligned_sep_a", september[-2], "season_aligned_template_replay", "development"),
        ("season_aligned_sep_b", september[-1], "season_aligned_template_replay", "development"),
    ]
    return [(build_template_replay_fold(train, template, start_month=start, scenario_id=sid, family=family), role)
            for sid, start, family, role in definitions]


def _r00(train: pd.DataFrame, template: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    scored = []
    oofs = []
    for fold, role in _development_folds(train, template):
        row, oof = _score("R00_persistence", fold, role, predict_persistence(fold.ledger))
        scored.append(row); oofs.append(oof)
    confirm = build_mask_block_fold(train, anchor_month="2014-12", end_month="2015-06",
                                    scenario_id="confirm_2014_12_to_2015_06")
    row, oof = _score("R00_persistence", confirm, "confirmation", predict_persistence(confirm.ledger))
    scored.append(row); oofs.append(oof)
    pd.concat(oofs, ignore_index=True).to_csv(out_dir / "oof.csv.gz", index=False, compression="gzip")
    return {"candidate": "R00_persistence", "scenarios": scored, "oof": "oof.csv.gz"}


def _attach_delta(rows: pd.DataFrame, labels: pd.DataFrame) -> np.ndarray:
    joined = rows.loc[:, ["sample_id", "last_observed_TWS"]].merge(
        labels.loc[:, ["sample_id", "target"]], on="sample_id", how="left", validate="one_to_one", sort=False
    )
    if joined["target"].isna().any():
        raise AssertionError("Training row is missing its label")
    return joined["target"].to_numpy(dtype=np.float32) - joined["last_observed_TWS"].to_numpy(dtype=np.float32)


def _scenario_training_rows(train: pd.DataFrame, template: pd.DataFrame, eval_start: pd.Period) -> tuple[pd.DataFrame, list[str]]:
    """Choose non-overlapping historical Test-template blocks before the cutoff."""
    max_offset = int(template["offset_months"].max())
    legal = [start for start in find_exact_template_starts(train, template) if start + max_offset <= eval_start - 2]
    selected: list[pd.Period] = []
    last_end: pd.Period | None = None
    for start in legal:
        if last_end is None or start > last_end:
            selected.append(start)
            last_end = start + max_offset
    if not selected:
        raise AssertionError(f"No completed historical template blocks before {eval_start}")
    folds = [build_template_replay_fold(train, template, start_month=start,
             scenario_id=f"train_template_{start}", family="training_template_replay") for start in selected]
    rows = pd.concat([fold.ledger for fold in folds], ignore_index=True)
    if rows["sample_id"].duplicated().any():
        raise AssertionError("Training template blocks unexpectedly duplicate a sample")
    if (pd.to_datetime(rows["source_date"]).dt.to_period("M") + 1 >= eval_start).any():
        raise AssertionError("Scenario-trained label reaches evaluation source month")
    return rows, [str(start) for start in selected]


def _model_run(mode: str, train: pd.DataFrame, template: pd.DataFrame, out_dir: Path, *, rounds: int, seed: int) -> dict[str, object]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM is required for R01-R03; run on Kaggle.") from exc
    structural = train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
    source = train.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
    labels = train.loc[:, ["sample_id", "target"]].copy()
    params = dict(DEFAULT_PARAMS)
    params.update({"seed": seed, "feature_fraction_seed": seed, "bagging_seed": seed,
                   "data_random_seed": seed, "num_threads": os.cpu_count() or 1})
    feature_names = VISIBLE_HISTORY_FEATURE_COLUMNS if mode == "r03" else HYDRO_GAP_SAFE_FEATURE_COLUMNS
    build_features = build_visible_history_feature_matrix if mode == "r03" else build_hydro_gap_safe_feature_matrix
    all_rows: list[dict[str, object]] = []
    all_oof: list[pd.DataFrame] = []
    model_dir = out_dir / "models"; model_dir.mkdir()
    for fold, role in _development_folds(train, template):
        start = pd.Period(fold.spec.first_source_month, freq="M")
        if mode == "r01":
            sampled = build_sampled_training_rows(structural, source.loc[:, SOURCE_CORE_COLUMNS],
                                                  max_target_month=start - 1, seed=seed)
            train_rows = sampled.rows
            training_schedule = {"kind": "legacy_independent_h_sampler", "horizon_counts": sampled.horizon_counts,
                                 "dropped_missing_anchor": sampled.dropped_missing_anchor}
        else:
            train_rows, starts = _scenario_training_rows(train, template, start)
            training_schedule = {"kind": "non_overlapping_transplanted_test_schedules", "starts": starts,
                                 "rows": int(len(train_rows))}
        y_train = _attach_delta(train_rows, labels)
        weights = horizon_rebalance_weights(train_rows["h"])
        x_train = build_features(train_rows, source, structural)
        x_valid = build_features(fold.ledger, source, structural)
        booster = lgb.train(params, lgb.Dataset(x_train, label=y_train, weight=weights, feature_name=feature_names),
                            num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
        model_path = model_dir / f"{mode}_{fold.spec.scenario_id}.txt"
        booster.save_model(str(model_path))
        prediction = fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float64) + booster.predict(x_valid)
        row, oof = _score(mode.upper(), fold, role, prediction)
        row.update({"rounds": rounds, "feature_names": feature_names, "training_schedule": training_schedule,
                    "training_rows": int(len(train_rows)), "model_file": str(model_path.relative_to(out_dir))})
        all_rows.append(row); all_oof.append(oof)
    pd.concat(all_oof, ignore_index=True).to_csv(out_dir / "oof.csv.gz", index=False, compression="gzip")
    return {"candidate": mode.upper(), "rounds": rounds, "seed": seed, "feature_names": feature_names,
            "scenarios": all_rows, "oof": "oof.csv.gz", "models": "models/"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gated drought validation experiments with preserved artifacts.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("preflight", "r00", "r01", "r02", "r03"), required=True)
    parser.add_argument("--rounds", type=int, default=173, help="Frozen rounds for matched R01-R03 comparisons.")
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    data_dir, out_dir = args.data_dir.resolve(), args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip():
        raise RuntimeError("Refusing to run from a dirty repository checkout")
    config = {"data_dir": str(data_dir), "output_dir": str(out_dir), "run_id": args.run_id,
              "mode": args.mode, "rounds": args.rounds, "seed": args.seed}
    _json(out_dir / "config.json", config)
    manifest: dict[str, object] = {"run_id": args.run_id, "mode": args.mode, "started_at": datetime.now(timezone.utc).isoformat(),
                                   "commit": head, "branch": branch, "config_hash": _sha256(out_dir / "config.json")}
    started = time.perf_counter()
    try:
        manifest["preflight"] = _preflight(data_dir)
        if args.mode == "preflight":
            metrics: dict[str, object] = {"status": "preflight_complete"}
        else:
            usecols = ["sample_id", "time", "lat", "lon", "TWS_t", "month_sin", "month_cos",
                       "SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t", "target"]
            train = pd.read_csv(data_dir / "Train.csv", usecols=usecols)
            test = pd.read_csv(data_dir / "Test.csv", usecols=["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"])
            template = build_test_mask_template(test)
            metrics = _r00(train, template, out_dir) if args.mode == "r00" else _model_run(
                args.mode, train, template, out_dir, rounds=args.rounds, seed=args.seed)
        metrics["elapsed_seconds"] = time.perf_counter() - started
        _json(out_dir / "metrics.json", metrics)
        manifest["status"] = "completed"; manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["output_checksums"] = {str(p.relative_to(out_dir)): _sha256(p) for p in out_dir.rglob("*") if p.is_file()}
        _json(out_dir / "manifest.json", manifest)
        print(json.dumps({"run_id": args.run_id, "status": "completed", "output_dir": str(out_dir)}, indent=2))
    except Exception as exc:
        manifest["status"] = "failed"; manifest["error"] = f"{type(exc).__name__}: {exc}"
        _json(out_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
