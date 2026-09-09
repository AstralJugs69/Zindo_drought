"""Reproducible, read-only diagnostics for the B3 leaderboard gap.

The runner is intentionally split into independent stages.  A stage reads only
the columns it needs, writes compact JSON/CSV summaries, and exits; this keeps
the large Train/Test panels and feature maps out of long-lived processes.  No
stage fits a model, writes a competition submission, or reads Test labels.

Typical Kaggle usage (one command per process)::

    python -u scripts/run_leaderboard_failure_investigation.py \
      --stage integrity --data-dir /kaggle/input/datasets/cashgenenator/drought \
      --run-dir /kaggle/working/drought_runs/leaderboard_failure_YYYYMMDDTHHMMSSZ

The heavier ``full_fit`` and ``sample_inference`` stages only score the already
saved B3 model.  Their results are summaries, never new prediction files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_features import (
    HYDRO_GAP_SAFE_FEATURE_COLUMNS,
    SOURCE_CORE_COLUMNS,
    SOURCE_HYDRO_HISTORY_COLUMNS,
    build_sampled_training_rows,
    horizon_rebalance_weights,
)
from src.metrics import TEST_H_WEIGHTS
from src.neural_sequence import build_b3_feature_maps, build_b3_matrix
from src.availability import build_replay_observation_view
from src.observation_simulator import build_mask_block_fold
from src.regional_context import HYDRO_COLUMNS
from scripts.run_b3_full_submission import (
    _attach_delta,
    _build_internal_train,
    _build_test_visibility,
    _namespace,
)


TRAIN_BASE_COLUMNS = ["sample_id", "time", "lat", "lon", "TWS_t", "target"]
TRAIN_SOURCE_COLUMNS = list(SOURCE_HYDRO_HISTORY_COLUMNS)
TRAIN_ALL_COLUMNS = [*TRAIN_BASE_COLUMNS, *[c for c in HYDRO_COLUMNS if c not in TRAIN_BASE_COLUMNS], "month_sin", "month_cos"]
TEST_BASE_COLUMNS = ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]
TEST_SOURCE_COLUMNS = ["ID", "time", "lat", "lon", "month_sin", "month_cos", *HYDRO_COLUMNS]
EXPECTED_TEST_ROWS = 280_961
EXPECTED_SUBMISSION_SHA = "62f1876ee7a26dfc5289c68d724353e9b89185d146eb7e8335a827e923424e7c"
FIT_DIR_DEFAULT = Path("/kaggle/working/drought_runs/b3_full_submission_20260909T010000Z")


def _json(path: Path, payload: object) -> None:
    """Write strict JSON, converting numpy scalars and periods first."""
    def convert(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, (pd.Timestamp, pd.Period)):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_values(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    first = True
    for value in values:
        if not first:
            digest.update(b"\n")
        first = False
        digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()


def _memory() -> dict[str, float | None]:
    try:
        import psutil

        process = psutil.Process()
        vm = psutil.virtual_memory()
        return {"rss_gib": process.memory_info().rss / 1024**3, "available_gib": vm.available / 1024**3}
    except Exception:
        return {"rss_gib": None, "available_gib": None}


def _period_number(values: pd.Series) -> pd.Series:
    dt = pd.to_datetime(values)
    return (dt.dt.year * 12 + dt.dt.month).astype("int64")


def _period_label(value: int) -> str:
    year = (int(value) - 1) // 12
    month = (int(value) - 1) % 12 + 1
    return f"{year:04d}-{month:02d}"


def _period_labels(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series([_period_label(int(v)) for v in values], index=getattr(values, "index", None), dtype="string")


def _finite_count(values: pd.Series) -> int:
    return int(np.isfinite(values.to_numpy(dtype=np.float64, na_value=np.nan)).sum())


def _metric_arrays(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if len(truth) != len(prediction):
        raise ValueError("metric arrays have different lengths")
    if not len(truth):
        return {"rows": 0, "mse": None, "rmse": None, "mae": None, "bias": None, "centered_std": None, "decomposition_error": None}
    error = prediction - truth
    bias = float(np.mean(error))
    centered = error - bias
    mse = float(np.mean(np.square(error)))
    return {
        "rows": int(len(error)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "bias": bias,
        "centered_std": float(np.std(error)),
        "decomposition_error": float(mse - (bias * bias + float(np.var(centered)))),
    }


def _weighted_rmse(frame: pd.DataFrame, prediction_col: str, truth_col: str = "target") -> float | None:
    """Official h=1..7 mixture; returns None if a horizon is absent."""
    if not set(range(1, 8)).issubset(set(frame["h"].astype(int))):
        return None
    weighted_mse = 0.0
    for horizon, weight in TEST_H_WEIGHTS.items():
        mask = frame["h"].to_numpy(dtype=np.int16) == int(horizon)
        error = frame.loc[mask, prediction_col].to_numpy(dtype=np.float64) - frame.loc[mask, truth_col].to_numpy(dtype=np.float64)
        weighted_mse += float(weight) * float(np.mean(np.square(error)))
    return float(np.sqrt(weighted_mse))


def _stage_start(run_dir: Path, stage: str, expected_commit: str | None) -> tuple[Path, dict[str, Any], float]:
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"{stage}.json"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite completed stage output: {out}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if expected_commit and head != expected_commit:
        raise RuntimeError({"expected_commit": expected_commit, "actual_commit": head})
    started = time.perf_counter()
    manifest = {
        "status": "running",
        "stage": stage,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit": head,
        "branch": branch,
        "memory_start": _memory(),
        "read_only": True,
        "test_labels_used": False,
        "model_fit": False,
        "submission_written": False,
    }
    _json(out, manifest)
    print(json.dumps({"stage": stage, "status": "running", "commit": head, "memory": _memory()}, sort_keys=True), flush=True)
    return out, manifest, started


def _stage_finish(path: Path, manifest: dict[str, Any], started: float, **details: Any) -> None:
    manifest.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started, "memory_end": _memory(), **details})
    _json(path, manifest)
    print(json.dumps({"stage": manifest["stage"], "status": "completed", "elapsed_seconds": manifest["elapsed_seconds"], **details}, sort_keys=True), flush=True)


def _stage_fail(path: Path, manifest: dict[str, Any], started: float, exc: BaseException) -> None:
    manifest.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}", "memory_end": _memory()})
    _json(path, manifest)


def _paths(data_dir: Path, submission: Path | None) -> dict[str, Path]:
    out = {name: data_dir / name for name in ("Train.csv", "Test.csv", "SampleSubmission.csv")}
    if submission is not None:
        out["submission.csv"] = submission
    return out


def _load_train(data_dir: Path, *, hydro: bool = False) -> pd.DataFrame:
    columns = list(TRAIN_BASE_COLUMNS)
    if hydro:
        columns = list(dict.fromkeys([*columns, *TRAIN_SOURCE_COLUMNS, "month_sin", "month_cos"]))
    return pd.read_csv(data_dir / "Train.csv", usecols=columns)


def _load_test(data_dir: Path, *, hydro: bool = False) -> pd.DataFrame:
    columns = list(TEST_BASE_COLUMNS)
    if hydro:
        columns = list(dict.fromkeys([*columns, *TEST_SOURCE_COLUMNS]))
    return pd.read_csv(data_dir / "Test.csv", usecols=columns)


def _join_identity(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], output: pd.DataFrame | None = None) -> dict[str, Any]:
    """Return explicit cardinality evidence for a keyed diagnostic join."""
    if left.duplicated(keys).any():
        raise AssertionError(f"left side is not unique on {keys}")
    if right.duplicated(keys).any():
        raise AssertionError(f"right side is not unique on {keys}")
    result = output if output is not None else left.merge(right, on=keys, how="left", validate="one_to_one", sort=False)
    id_column = keys[0]
    return {
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "output_rows": int(len(result)),
        "left_unique_keys": int(left[keys].drop_duplicates().shape[0]),
        "right_unique_keys": int(right[keys].drop_duplicates().shape[0]),
        "output_unique_keys": int(result[keys].drop_duplicates().shape[0]),
        "unmatched_left_keys": int((~left.set_index(keys).index.isin(right.set_index(keys).index)).sum()),
        "identity_key": id_column,
    }


def _next_calendar_alignment(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join each supplied label to the exact next calendar-month TWS row."""
    required = {"sample_id", "time", "lat", "lon", "TWS_t", "target"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"train missing columns: {sorted(missing)}")
    x = train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t", "target"]].copy()
    x["source_period"] = _period_number(x["time"])
    if x.duplicated(["lat", "lon", "source_period"]).any():
        raise AssertionError("train has duplicate location/calendar rows")
    lookup = x.loc[:, ["lat", "lon", "source_period", "TWS_t"]].rename(columns={"source_period": "target_period", "TWS_t": "next_calendar_TWS"})
    left = x.loc[:, ["sample_id", "lat", "lon", "source_period", "target"]].copy()
    left["target_period"] = left["source_period"] + 1
    joined = left.merge(lookup, on=["lat", "lon", "target_period"], how="left", validate="many_to_one", sort=False)
    matched = joined["next_calendar_TWS"].notna()
    diff = joined.loc[matched, "target"].to_numpy(dtype=np.float64) - joined.loc[matched, "next_calendar_TWS"].to_numpy(dtype=np.float64)
    evidence = {"left_rows": int(len(left)), "right_rows": int(len(lookup)), "output_rows": int(len(joined)), "matched_rows": int(matched.sum()), "unmatched_rows": int((~matched).sum()), "duplicate_location_month_rows": int(x.duplicated(["lat", "lon", "source_period"]).sum()), "max_abs_difference": None if not len(diff) else float(np.max(np.abs(diff))), "exact_zero_difference": bool(not len(diff) or np.allclose(diff, 0.0, atol=1e-12, rtol=0.0))}
    return joined, evidence


def _file_identity(path: Path, *, id_column: str | None = None, numeric_columns: list[str] | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    rows = 0
    ids: list[str] = []
    nonfinite: dict[str, int] = {c: 0 for c in (numeric_columns or [])}
    usecols = list(dict.fromkeys([*( [id_column] if id_column else []), *(numeric_columns or [])]))
    for chunk in pd.read_csv(path, usecols=usecols or None, chunksize=100_000):
        rows += len(chunk)
        if id_column:
            ids.extend(chunk[id_column].astype(str).tolist())
        for column in nonfinite:
            if column in chunk:
                nonfinite[column] += int((~np.isfinite(chunk[column].to_numpy(dtype=np.float64, na_value=np.nan))).sum())
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
        "header": header,
        "rows": rows,
        "nonfinite": nonfinite,
    }
    if id_column:
        result.update({"id_column": id_column, "id_hash": _hash_values(ids), "duplicate_ids": int(pd.Series(ids).duplicated().sum())})
    return result


def stage_integrity(args: argparse.Namespace) -> None:
    path, manifest, started = _stage_start(args.run_dir, "integrity", args.expected_commit)
    try:
        data_dir = args.data_dir.resolve()
        paths = _paths(data_dir, args.submission.resolve() if args.submission else None)
        identities = {
            "Train.csv": _file_identity(paths["Train.csv"], id_column="sample_id", numeric_columns=["lat", "lon", "TWS_t", "target"]),
            "Test.csv": _file_identity(paths["Test.csv"], id_column="ID", numeric_columns=["lat", "lon", "TWS_t", *HYDRO_COLUMNS]),
            "SampleSubmission.csv": _file_identity(paths["SampleSubmission.csv"], id_column="ID", numeric_columns=["Target"]),
        }
        if "submission.csv" in paths:
            identities["submission.csv"] = _file_identity(paths["submission.csv"], id_column="ID", numeric_columns=["Target"])
        test_ids = pd.read_csv(paths["Test.csv"], usecols=["ID"])["ID"].astype(str)
        sample_ids = pd.read_csv(paths["SampleSubmission.csv"], usecols=["ID"])["ID"].astype(str)
        exact_template = bool(len(test_ids) == len(sample_ids) and test_ids.tolist() == sample_ids.tolist())
        submission_match: dict[str, Any] = {"available": "submission.csv" in paths}
        if "submission.csv" in paths:
            sub = pd.read_csv(paths["submission.csv"], usecols=["ID", "Target"])
            sub_ids = sub["ID"].astype(str)
            submission_match.update({
                "rows": int(len(sub)),
                "columns": sub.columns.tolist(),
                "exact_sample_order": bool(sub_ids.tolist() == sample_ids.tolist()),
                "exact_sample_set": bool(set(sub_ids) == set(sample_ids)),
                "finite_target": bool(np.isfinite(sub["Target"].to_numpy(dtype=np.float64, na_value=np.nan)).all()),
                "duplicate_ids": int(sub_ids.duplicated().sum()),
            })
        result = {
            "data_dir": str(data_dir),
            "files": identities,
            "exact_test_sample_submission_order": exact_template,
            "test_rows_expected": EXPECTED_TEST_ROWS,
            "test_rows_match_expected": int(len(test_ids)) == EXPECTED_TEST_ROWS,
            "submission_expected_sha256": EXPECTED_SUBMISSION_SHA,
            "submission_sha256_matches_recorded": identities.get("submission.csv", {}).get("sha256") == EXPECTED_SUBMISSION_SHA,
            "submission": submission_match,
            "join_identity": {"test_id_rows": int(len(test_ids)), "sample_id_rows": int(len(sample_ids)), "test_unique": int(test_ids.nunique()), "sample_unique": int(sample_ids.nunique())},
        }
        _json(args.run_dir / "integrity_details.json", result)
        _stage_finish(path, manifest, started, details_path=str(args.run_dir / "integrity_details.json"), integrity=result)
    except Exception as exc:
        _stage_fail(path, manifest, started, exc)
        raise


def stage_target_alignment(args: argparse.Namespace) -> None:
    path, manifest, started = _stage_start(args.run_dir, "target_alignment", args.expected_commit)
    try:
        train = _load_train(args.data_dir)
        joined, evidence = _next_calendar_alignment(train)
        left = joined.loc[:, ["sample_id", "lat", "lon", "source_period", "target"]].copy()
        duplicate_keys = int(evidence["duplicate_location_month_rows"])
        matched = joined["next_calendar_TWS"].notna()
        difference = joined.loc[matched, "target"].to_numpy(dtype=np.float64) - joined.loc[matched, "next_calendar_TWS"].to_numpy(dtype=np.float64)
        by_month: list[dict[str, Any]] = []
        for source_period, group in joined.groupby("source_period", sort=True):
            m = group["next_calendar_TWS"].notna().to_numpy()
            d = group.loc[m, "target"].to_numpy(dtype=np.float64) - group.loc[m, "next_calendar_TWS"].to_numpy(dtype=np.float64)
            by_month.append({"source_month": _period_label(int(source_period)), "rows": int(len(group)), "matched": int(m.sum()), "unmatched": int((~m).sum()), "max_abs_difference": None if not len(d) else float(np.max(np.abs(d))), "exact_zero_difference": bool(len(d) == 0 or np.allclose(d, 0.0, atol=1e-12, rtol=0.0))})
        table = pd.DataFrame(by_month)
        table.to_csv(args.run_dir / "target_alignment_by_source_month.csv", index=False)
        result = {
            "train_rows": int(len(train)),
            "duplicate_location_month_rows": duplicate_keys,
            "matched_next_calendar_rows": int(matched.sum()),
            "unmatched_next_calendar_rows": int((~matched).sum()),
            "max_abs_target_next_calendar_difference": None if not len(difference) else float(np.max(np.abs(difference))),
            "mean_abs_target_next_calendar_difference": None if not len(difference) else float(np.mean(np.abs(difference))),
            "exact_identity_at_1e-12": bool(len(difference) == 0 or np.allclose(difference, 0.0, atol=1e-12, rtol=0.0)),
            "row_join": {**evidence, "output_unique_sample_id": int(joined["sample_id"].nunique())},
            "by_source_month_csv": str(args.run_dir / "target_alignment_by_source_month.csv"),
        }
        _json(args.run_dir / "target_alignment_details.json", result)
        _stage_finish(path, manifest, started, details_path=str(args.run_dir / "target_alignment_details.json"), alignment=result)
    except Exception as exc:
        _stage_fail(path, manifest, started, exc)
        raise


def _target_summary(group: pd.DataFrame) -> dict[str, Any]:
    target = group["target"].to_numpy(dtype=np.float64)
    current = group["TWS_t"].to_numpy(dtype=np.float64)
    delta = target - current
    bias = float(np.mean(delta))
    centered = delta - bias
    out: dict[str, Any] = {
        "rows": int(len(group)),
        "locations": int(group[["lat", "lon"]].drop_duplicates().shape[0]),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)),
        "target_q05": float(np.quantile(target, 0.05)),
        "target_q50": float(np.quantile(target, 0.50)),
        "target_q95": float(np.quantile(target, 0.95)),
        "current_TWS_mean": float(np.mean(current)),
        "current_TWS_std": float(np.std(current)),
        "current_TWS_q05": float(np.quantile(current, 0.05)),
        "current_TWS_q50": float(np.quantile(current, 0.50)),
        "current_TWS_q95": float(np.quantile(current, 0.95)),
        "target_minus_current_rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "target_minus_current_mae": float(np.mean(np.abs(delta))),
        "target_minus_current_bias": bias,
        "target_minus_current_centered_std": float(np.std(delta)),
        "rmse2_minus_bias2_minus_centered_var": float(np.mean(np.square(delta)) - (bias * bias + np.var(centered))),
        "share_abs_change_gt_1": float(np.mean(np.abs(delta) > 1.0)),
        "share_abs_change_gt_2": float(np.mean(np.abs(delta) > 2.0)),
        "share_abs_change_gt_3": float(np.mean(np.abs(delta) > 3.0)),
    }
    for variable in HYDRO_COLUMNS:
        out[f"finite_{variable}"] = float(np.isfinite(group[variable].to_numpy(dtype=np.float64, na_value=np.nan)).mean()) if variable in group else None
    return out


def stage_target_behavior(args: argparse.Namespace) -> None:
    path, manifest, started = _stage_start(args.run_dir, "target_behavior", args.expected_commit)
    try:
        train = _load_train(args.data_dir, hydro=True)
        train["source_period"] = _period_number(train["time"])
        dt = pd.to_datetime(train["time"])
        train["source_year"] = dt.dt.year.astype(int)
        train["calendar_month"] = dt.dt.month.astype(int)
        monthly: list[dict[str, Any]] = []
        for source_period, group in train.groupby("source_period", sort=True):
            row = {"source_month": _period_label(int(source_period)), "source_year": int(group["source_year"].iloc[0]), "calendar_month": int(group["calendar_month"].iloc[0]), **_target_summary(group)}
            monthly.append(row)
        monthly_frame = pd.DataFrame(monthly)
        monthly_frame.to_csv(args.run_dir / "target_by_source_month.csv", index=False)

        selected = train.loc[train["source_year"].isin([2012, 2013, 2014, 2015]) & train["calendar_month"].isin([1, 2, 6])].copy()
        comparison_rows: list[dict[str, Any]] = []
        for (year, month), group in selected.groupby(["source_year", "calendar_month"], sort=True):
            comparison_rows.append({"comparison": "year_month", "source_year": int(year), "calendar_month": int(month), **_target_summary(group)})
        panel = selected.loc[:, ["lat", "lon", "calendar_month", "source_year", "target", "TWS_t"]].copy()
        panel["delta"] = panel["target"] - panel["TWS_t"]
        # One value per location/year/month is expected; averaging duplicates
        # makes the comparison explicit rather than silently dropping them.
        wide = panel.groupby(["lat", "lon", "calendar_month", "source_year"], sort=False)["delta"].mean().unstack("source_year")
        prior_columns = [c for c in [2012, 2013, 2014] if c in wide.columns]
        # The source-year level is in the columns after unstack; filter rows
        # with 2015 and retain the same location/calendar keys from the index.
        if 2015 in wide.columns and prior_columns:
            panel_2015 = wide.loc[wide[2015].notna() & wide[prior_columns].notna().any(axis=1), [*prior_columns, 2015]].copy()
            panel_2015["prior_mean"] = panel_2015[prior_columns].mean(axis=1)
            panel_2015["difference_2015_minus_prior_mean"] = panel_2015[2015] - panel_2015["prior_mean"]
            for month, group in panel_2015.groupby(level="calendar_month", sort=True):
                difference = group["difference_2015_minus_prior_mean"].to_numpy(dtype=np.float64)
                comparison_rows.append({"comparison": "matched_location_calendar_month", "source_year": 2015, "calendar_month": int(month), "rows": int(len(group)), "locations": int(len(group)), "prior_years": ",".join(map(str, prior_columns)), "difference_rmse": float(np.sqrt(np.mean(np.square(difference)))), "difference_bias": float(np.mean(difference)), "difference_centered_std": float(np.std(difference))})
            panel_2015.reset_index().to_csv(args.run_dir / "target_matched_panel_2015.csv", index=False)
        comparison_frame = pd.DataFrame(comparison_rows)
        comparison_frame.to_csv(args.run_dir / "target_comparison.csv", index=False)

        focus = train.loc[(train["source_year"] == 2015) & train["calendar_month"].isin([1, 2, 6])].copy()
        focus["cell_lat5"] = np.floor(focus["lat"].astype(float) / 5.0) * 5.0
        focus["cell_lon5"] = np.floor(focus["lon"].astype(float) / 5.0) * 5.0
        focus["delta"] = focus["target"] - focus["TWS_t"]
        cells: list[pd.DataFrame] = []
        for (month, lat5, lon5), group in focus.groupby(["calendar_month", "cell_lat5", "cell_lon5"], sort=True):
            delta = group["delta"].to_numpy(dtype=np.float64)
            cells.append(pd.DataFrame([{"source_year": 2015, "calendar_month": int(month), "cell_lat5": float(lat5), "cell_lon5": float(lon5), "rows": int(len(group)), "locations": int(group[["lat", "lon"]].drop_duplicates().shape[0]), "sse": float(np.sum(np.square(delta))), "rmse": float(np.sqrt(np.mean(np.square(delta)))), "bias": float(np.mean(delta)), "abs_change_gt_1": float(np.mean(np.abs(delta) > 1.0)), "abs_change_gt_2": float(np.mean(np.abs(delta) > 2.0)), "abs_change_gt_3": float(np.mean(np.abs(delta) > 3.0))}]))
        if cells:
            cell_frame = pd.concat(cells, ignore_index=True)
            cell_frame = cell_frame.sort_values(["calendar_month", "sse", "cell_lat5", "cell_lon5"], ascending=[True, False, True, True], kind="mergesort")
            cell_frame["month_total_sse"] = cell_frame.groupby("calendar_month")["sse"].transform("sum")
            cell_frame["sse_share"] = cell_frame["sse"] / cell_frame["month_total_sse"].replace(0, np.nan)
            cell_frame["cumulative_sse_share"] = cell_frame.groupby("calendar_month")["sse"].cumsum() / cell_frame["month_total_sse"].replace(0, np.nan)
            cell_frame["cumulative_rows"] = cell_frame.groupby("calendar_month")["rows"].cumsum()
            cell_frame["cumulative_row_share"] = cell_frame["cumulative_rows"] / cell_frame.groupby("calendar_month")["rows"].transform("sum")
        else:
            cell_frame = pd.DataFrame(columns=["source_year", "calendar_month", "cell_lat5", "cell_lon5", "rows", "locations", "sse", "rmse", "bias", "abs_change_gt_1", "abs_change_gt_2", "abs_change_gt_3", "month_total_sse", "sse_share", "cumulative_sse_share", "cumulative_rows", "cumulative_row_share"])
        cell_frame.to_csv(args.run_dir / "target_cells_2015.csv", index=False)

        source_periods = train["source_period"].drop_duplicates().sort_values().to_numpy(dtype=np.int64)
        location_count = int(train[["lat", "lon"]].drop_duplicates().shape[0])
        global_span = int(source_periods.max() - source_periods.min() + 1) if len(source_periods) else 0
        result = {
            "train_rows": int(len(train)),
            "locations": location_count,
            "source_months": int(len(source_periods)),
            "source_month_min": _period_label(int(source_periods.min())),
            "source_month_max": _period_label(int(source_periods.max())),
            "duplicate_location_month_rows": int(train.duplicated(["lat", "lon", "source_period"]).sum()),
            "location_month_missingness": {"global_calendar_span_months": global_span, "location_count_times_span": int(location_count * global_span), "observed_rows": int(len(train)), "unobserved_location_month_slots_lower_bound": int(max(0, location_count * global_span - len(train)))},
            "2015_focus_rows": int(len(focus)),
            "2015_focus_months": sorted(int(x) for x in focus["calendar_month"].unique()),
            "monthly_csv": str(args.run_dir / "target_by_source_month.csv"),
            "comparison_csv": str(args.run_dir / "target_comparison.csv"),
            "cells_csv": str(args.run_dir / "target_cells_2015.csv"),
            "matched_panel_csv": str(args.run_dir / "target_matched_panel_2015.csv") if (args.run_dir / "target_matched_panel_2015.csv").exists() else None,
        }
        _json(args.run_dir / "target_behavior_details.json", result)
        _stage_finish(path, manifest, started, details_path=str(args.run_dir / "target_behavior_details.json"), behavior=result)
    except Exception as exc:
        _stage_fail(path, manifest, started, exc)
        raise


def _find_oof_files(oof_root: Path) -> list[Path]:
    files = sorted(path for path in oof_root.rglob("oof.csv.gz") if "_D0_" in path.parent.name)
    if not files:
        raise FileNotFoundError(f"No D0 oof.csv.gz files below {oof_root}")
    return files


def _oof_group_table(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        b3 = _metric_arrays(group["target"].to_numpy(), group["prediction"].to_numpy())
        persistence = _metric_arrays(group["target"].to_numpy(), group["persistence"].to_numpy())
        row = {column: value for column, value in zip(group_columns, key)}
        row.update({f"b3_{name}": value for name, value in b3.items()})
        row.update({f"persistence_{name}": value for name, value in persistence.items()})
        row["b3_minus_persistence_rmse"] = None if b3["rmse"] is None or persistence["rmse"] is None else float(b3["rmse"] - persistence["rmse"])
        row["b3_minus_persistence_sse"] = float(np.sum(np.square(group["prediction"].to_numpy(dtype=np.float64) - group["target"].to_numpy(dtype=np.float64))) - np.sum(np.square(group["persistence"].to_numpy(dtype=np.float64) - group["target"].to_numpy(dtype=np.float64))))
        rows.append(row)
    return pd.DataFrame(rows)


def stage_oof_decomposition(args: argparse.Namespace) -> None:
    path, manifest, started = _stage_start(args.run_dir, "oof_decomposition", args.expected_commit)
    try:
        files = _find_oof_files(args.oof_root.resolve())
        frames: list[pd.DataFrame] = []
        for file in files:
            origin = file.parent.name.split("_D0_", 1)[-1]
            columns = ["sample_id", "source_date", "last_observed_date", "last_observed_TWS", "h", "lat", "lon", "target", "prediction", "anchor_age_months", "geo5", "calendar_block"]
            frame = pd.read_csv(file, usecols=lambda c: c in columns)
            missing = set(columns).difference(frame.columns)
            if missing:
                raise ValueError(f"{file} missing OOF columns: {sorted(missing)}")
            frame["origin"] = origin
            frame["source_month"] = pd.to_datetime(frame["source_date"]).dt.to_period("M").astype(str)
            frame["target_month"] = (pd.to_datetime(frame["source_date"]) + pd.offsets.MonthBegin(1)).dt.to_period("M").astype(str)
            frame["h"] = frame["h"].astype(int)
            frame["persistence"] = frame["last_observed_TWS"].astype(float)
            frames.append(frame)
            print(json.dumps({"stage": "oof_decomposition", "file": str(file), "origin": origin, "rows": len(frame), "memory": _memory()}, sort_keys=True), flush=True)
        all_oof = pd.concat(frames, ignore_index=True, sort=False)
        overall: list[dict[str, Any]] = []
        for origin, group in all_oof.groupby("origin", sort=True):
            b3 = _metric_arrays(group["target"], group["prediction"])
            persistence = _metric_arrays(group["target"], group["persistence"])
            overall.append({"origin": origin, "h1_7_weighted_rmse": _weighted_rmse(group.loc[group["h"].between(1, 7)], "prediction"), "h1_7_rows": int(group["h"].between(1, 7).sum()), "h_gt7_rows": int((group["h"] > 7).sum()), **{f"b3_{k}": v for k, v in b3.items()}, **{f"persistence_{k}": v for k, v in persistence.items()}, "b3_minus_persistence_rmse": None if b3["rmse"] is None or persistence["rmse"] is None else float(b3["rmse"] - persistence["rmse"])})
        pooled_b3 = _metric_arrays(all_oof["target"], all_oof["prediction"])
        pooled_persistence = _metric_arrays(all_oof["target"], all_oof["persistence"])
        overall.append({"origin": "POOLED", "h1_7_weighted_rmse": _weighted_rmse(all_oof.loc[all_oof["h"].between(1, 7)], "prediction"), "h1_7_rows": int(all_oof["h"].between(1, 7).sum()), "h_gt7_rows": int((all_oof["h"] > 7).sum()), **{f"b3_{k}": v for k, v in pooled_b3.items()}, **{f"persistence_{k}": v for k, v in pooled_persistence.items()}, "b3_minus_persistence_rmse": float(pooled_b3["rmse"] - pooled_persistence["rmse"])})
        equal_origin_mse = float(np.mean([row["b3_mse"] for row in overall if row["origin"] != "POOLED" and row["b3_mse"] is not None]))
        overall.append({"origin": "EQUAL_ORIGIN_MSE", "b3_mse": equal_origin_mse, "b3_rmse": float(np.sqrt(equal_origin_mse)), "origins": int(len(files)), "interpretation": "mean of origin MSEs; not an independent sample count"})
        pd.DataFrame(overall).to_csv(args.run_dir / "oof_overall.csv", index=False)
        htable = _oof_group_table(all_oof, ["origin", "h"])
        htable.to_csv(args.run_dir / "oof_by_horizon.csv", index=False)
        source_table = _oof_group_table(all_oof, ["origin", "source_month", "target_month"])
        source_table.to_csv(args.run_dir / "oof_by_source_month.csv", index=False)
        age_table = _oof_group_table(all_oof, ["origin", "anchor_age_months"])
        age_table.to_csv(args.run_dir / "oof_by_anchor_age.csv", index=False)
        geo_table = _oof_group_table(all_oof, ["origin", "geo5"])
        geo_table.to_csv(args.run_dir / "oof_by_geo5.csv", index=False)
        late = all_oof.loc[all_oof["origin"].eq("2014_12") | all_oof["target_month"].isin(["2015-01", "2015-02", "2015-06"])].copy()
        late_table = _oof_group_table(late, ["origin", "source_month", "target_month", "h"])
        late_table.to_csv(args.run_dir / "oof_late_months.csv", index=False)
        cells = all_oof.loc[all_oof["target_month"].isin(["2015-01", "2015-02", "2015-06"])].copy()
        if len(cells):
            cells["cell_lat5"] = np.floor(cells["lat"].astype(float) / 5.0) * 5.0
            cells["cell_lon5"] = np.floor(cells["lon"].astype(float) / 5.0) * 5.0
            cell_table = _oof_group_table(cells, ["origin", "target_month", "cell_lat5", "cell_lon5"])
        else:
            cell_table = pd.DataFrame()
        cell_table.to_csv(args.run_dir / "oof_late_cells.csv", index=False)
        overlap_rows: list[dict[str, Any]] = []
        keys_by_origin: dict[str, set[str]] = {}
        months_by_origin: dict[str, set[str]] = {}
        for origin, group in all_oof.groupby("origin", sort=True):
            keys_by_origin[origin] = set(group["lat"].astype(str) + "|" + group["lon"].astype(str) + "|" + group["target_month"].astype(str))
            months_by_origin[origin] = set(group["target_month"].astype(str))
        origins = sorted(keys_by_origin)
        for i, first in enumerate(origins):
            for second in origins[i + 1:]:
                overlap_rows.append({"origin_a": first, "origin_b": second, "physical_target_key_overlap": int(len(keys_by_origin[first] & keys_by_origin[second])), "calendar_month_overlap": int(len(months_by_origin[first] & months_by_origin[second]))})
        pd.DataFrame(overlap_rows).to_csv(args.run_dir / "oof_overlap.csv", index=False)
        result = {"files": [str(f) for f in files], "origins": [f.parent.name.split("_D0_", 1)[-1] for f in files], "rows_total": int(len(all_oof)), "pooled_rows_are_overlapping": True, "overall_csv": str(args.run_dir / "oof_overall.csv"), "horizon_csv": str(args.run_dir / "oof_by_horizon.csv"), "source_month_csv": str(args.run_dir / "oof_by_source_month.csv"), "age_csv": str(args.run_dir / "oof_by_anchor_age.csv"), "geo_csv": str(args.run_dir / "oof_by_geo5.csv"), "late_month_csv": str(args.run_dir / "oof_late_months.csv"), "late_cell_csv": str(args.run_dir / "oof_late_cells.csv"), "overlap_csv": str(args.run_dir / "oof_overlap.csv"), "pooled_b3": pooled_b3, "pooled_persistence": pooled_persistence}
        _json(args.run_dir / "oof_decomposition_details.json", result)
        _stage_finish(path, manifest, started, details_path=str(args.run_dir / "oof_decomposition_details.json"), decomposition=result)
    except Exception as exc:
        _stage_fail(path, manifest, started, exc)
        raise


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _aggregate_error_frame(frame: pd.DataFrame, group_columns: list[str], *, error_col: str = "error", persistence_error_col: str = "persistence_error") -> pd.DataFrame:
    """Aggregate errors into additive sufficient statistics.

    Means are deliberately not emitted here.  A later grouping may combine
    batches or cells with unequal row counts, so carrying a batch mean and then
    summing it would apply the wrong denominator.  ``_finalize_error_metrics``
    is the only place that derives RMSE, MAE, bias, and centered spread.
    """
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        error = group[error_col].to_numpy(dtype=np.float64)
        persistence = group[persistence_error_col].to_numpy(dtype=np.float64)
        values = {column: value for column, value in zip(group_columns, key)}
        values.update({
            "rows": int(len(group)),
            "b3_sse": float(np.sum(np.square(error))),
            "b3_abs_error_sum": float(np.sum(np.abs(error))),
            "b3_error_sum": float(np.sum(error)),
            "persistence_sse": float(np.sum(np.square(persistence))),
            "persistence_abs_error_sum": float(np.sum(np.abs(persistence))),
            "persistence_error_sum": float(np.sum(persistence)),
        })
        rows.append(values)
    return pd.DataFrame(rows)


def _finalize_error_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive row-normalized metrics from additive error statistics."""
    out = frame.copy()
    if out.empty:
        return out
    rows = out["rows"].to_numpy(dtype=np.float64)
    if np.any(rows <= 0):
        raise AssertionError("error aggregates must have positive row counts")
    b3_mse = out["b3_sse"].to_numpy(dtype=np.float64) / rows
    persistence_mse = out["persistence_sse"].to_numpy(dtype=np.float64) / rows
    b3_bias = out["b3_error_sum"].to_numpy(dtype=np.float64) / rows
    persistence_bias = out["persistence_error_sum"].to_numpy(dtype=np.float64) / rows
    out["b3_rmse"] = np.sqrt(np.maximum(b3_mse, 0.0))
    out["b3_mae"] = out["b3_abs_error_sum"].to_numpy(dtype=np.float64) / rows
    out["b3_bias"] = b3_bias
    out["b3_centered_std"] = np.sqrt(np.maximum(b3_mse - np.square(b3_bias), 0.0))
    out["persistence_rmse"] = np.sqrt(np.maximum(persistence_mse, 0.0))
    out["persistence_mae"] = out["persistence_abs_error_sum"].to_numpy(dtype=np.float64) / rows
    out["persistence_bias"] = persistence_bias
    out["persistence_centered_std"] = np.sqrt(np.maximum(persistence_mse - np.square(persistence_bias), 0.0))
    out["b3_minus_persistence_rmse"] = out["b3_rmse"] - out["persistence_rmse"]
    out["b3_minus_persistence_sse"] = out["b3_sse"] - out["persistence_sse"]
    out["rmse2_minus_bias2_minus_centered_var"] = b3_mse - (
        np.square(b3_bias) + np.square(out["b3_centered_std"].to_numpy(dtype=np.float64))
    )
    return out


def _tag_training_exposure(
    replay: pd.DataFrame,
    training_events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Tag replay rows by exact ``(sample_id, anchor date, h)`` exposure.

    A source ID occurring in the full-training sample is not sufficient to call
    a replay row in-sample: deterministic horizon sampling can assign a
    different anchor task to that same source event.  The anchor date and
    horizon are therefore part of the join identity.
    """
    required_replay = {"sample_id", "last_observed_date", "h"}
    required_training = {"sample_id", "last_observed_date", "h"}
    if missing := required_replay.difference(replay.columns):
        raise ValueError(f"replay missing exposure columns: {sorted(missing)}")
    if missing := required_training.difference(training_events.columns):
        raise ValueError(f"training events missing exposure columns: {sorted(missing)}")
    left = replay.copy()
    right = training_events.loc[:, ["sample_id", "last_observed_date", "h"]].copy()
    left["sample_id"] = left["sample_id"].astype(str)
    right["sample_id"] = right["sample_id"].astype(str)
    left["_anchor_key"] = pd.to_datetime(left["last_observed_date"]).dt.strftime("%Y-%m-%d")
    right["_anchor_key"] = pd.to_datetime(right["last_observed_date"]).dt.strftime("%Y-%m-%d")
    left["h"] = left["h"].astype(int)
    right["h"] = right["h"].astype(int)
    key = ["sample_id", "_anchor_key", "h"]
    if right.duplicated(key).any():
        raise AssertionError("full training exposure keys are not unique")
    source_ids = set(right["sample_id"])
    tagged = left.merge(
        right.loc[:, key].assign(_exact_training_exposure=True),
        on=key,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    tagged["source_id_in_full_training"] = tagged["sample_id"].isin(source_ids)
    tagged["exact_training_exposure"] = tagged["_exact_training_exposure"].eq(True)
    tagged["exposure_status"] = np.select(
        [
            tagged["exact_training_exposure"].to_numpy(dtype=bool),
            tagged["source_id_in_full_training"].to_numpy(dtype=bool),
        ],
        ["exact_exposure", "different_anchor"],
        default="source_id_not_in_full_training",
    )
    tagged = tagged.drop(columns=["_anchor_key", "_exact_training_exposure"])
    counts = {
        "replay_rows": int(len(tagged)),
        "source_id_in_full_training_rows": int(tagged["source_id_in_full_training"].sum()),
        "exact_exposure_rows": int(tagged["exact_training_exposure"].sum()),
        "different_anchor_rows": int((tagged["exposure_status"] == "different_anchor").sum()),
        "source_id_not_in_full_training_rows": int((tagged["exposure_status"] == "source_id_not_in_full_training").sum()),
    }
    if sum(counts[key] for key in ("exact_exposure_rows", "different_anchor_rows", "source_id_not_in_full_training_rows")) != counts["replay_rows"]:
        raise AssertionError("exposure status counts do not partition replay rows")
    return tagged, counts


def _comparison_metric_row(
    frame: pd.DataFrame,
    *,
    scope: str,
    group_values: dict[str, Any],
) -> dict[str, Any]:
    """Return A/B/C/persistence metrics for one paired subset."""
    row: dict[str, Any] = {"scope": scope, **group_values, "rows": int(len(frame))}
    row["h1_7_rows"] = int(frame["h"].between(1, 7).sum())
    row["h_gt7_rows"] = int((frame["h"] > 7).sum())
    for label in ("a", "b", "c", "persistence"):
        prediction = frame[f"{label}_prediction"].to_numpy(dtype=np.float64)
        metrics = _metric_arrays(frame["target"].to_numpy(dtype=np.float64), prediction)
        row[f"{label}_sse"] = float(metrics["mse"] * metrics["rows"]) if metrics["mse"] is not None else None
        row[f"{label}_rmse"] = metrics["rmse"]
        row[f"{label}_mae"] = metrics["mae"]
        row[f"{label}_bias"] = metrics["bias"]
        row[f"{label}_centered_std"] = metrics["centered_std"]
        row[f"{label}_decomposition_error"] = metrics["decomposition_error"]
        row[f"{label}_h1_7_weighted_rmse"] = _weighted_rmse(
            frame.loc[frame["h"].between(1, 7)], f"{label}_prediction"
        )
    row["b_minus_a_rmse"] = None if row["b_rmse"] is None or row["a_rmse"] is None else float(row["b_rmse"] - row["a_rmse"])
    row["c_minus_b_rmse"] = None if row["c_rmse"] is None or row["b_rmse"] is None else float(row["c_rmse"] - row["b_rmse"])
    row["b_minus_a_sse"] = None if row["b_sse"] is None or row["a_sse"] is None else float(row["b_sse"] - row["a_sse"])
    row["c_minus_b_sse"] = None if row["c_sse"] is None or row["b_sse"] is None else float(row["c_sse"] - row["b_sse"])
    return row


def _comparison_metric_table(
    frame: pd.DataFrame,
    group_columns: list[str],
    scopes: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, subset in scopes.items():
        if subset.empty:
            continue
        if group_columns:
            grouped = subset.groupby(group_columns, sort=True, dropna=False)
            groups: Iterable[tuple[tuple[Any, ...], pd.DataFrame]] = grouped
        else:
            groups = [((), subset)]
        for key, group in groups:
            if not isinstance(key, tuple):
                key = (key,)
            values = {column: value for column, value in zip(group_columns, key)}
            rows.append(_comparison_metric_row(group, scope=scope, group_values=values))
    return pd.DataFrame(rows)


def _source_view_month_table(
    sparse_source: pd.DataFrame,
    dense_source: pd.DataFrame,
    *,
    first_source_month: str,
    last_source_month: str,
) -> pd.DataFrame:
    """Summarize the source-row support used by sparse and dense feature views."""
    first = pd.Period(first_source_month, freq="M")
    last = pd.Period(last_source_month, freq="M")
    months = pd.period_range(first, last, freq="M")

    def summarize(source: pd.DataFrame, prefix: str) -> pd.DataFrame:
        x = source.loc[:, ["time", "lat", "lon"]].copy()
        x["source_month"] = pd.to_datetime(x["time"]).dt.to_period("M")
        x["cell_lat5"] = np.floor(x["lat"].astype(float) / 5.0) * 5.0
        x["cell_lon5"] = np.floor(x["lon"].astype(float) / 5.0) * 5.0
        x["_location_key"] = x["lat"].astype(str) + "|" + x["lon"].astype(str)
        x["_region5_key"] = x["cell_lat5"].astype(str) + "|" + x["cell_lon5"].astype(str)
        grouped = x.groupby("source_month", sort=True).agg(
            rows=("source_month", "size"),
            locations=("_location_key", "nunique"),
            region5_cells=("_region5_key", "nunique"),
        )
        grouped.index = pd.PeriodIndex(grouped.index, freq="M")
        grouped = grouped.reindex(months, fill_value=0).reset_index().rename(columns={"index": "source_month"})
        grouped["source_month"] = grouped["source_month"].astype(str)
        return grouped.rename(columns={column: f"{prefix}_{column}" for column in ("rows", "locations", "region5_cells")})

    sparse = summarize(sparse_source, "sparse")
    dense = summarize(dense_source, "dense")
    out = sparse.merge(dense, on="source_month", how="outer", validate="one_to_one", sort=True)
    out["dense_minus_sparse_rows"] = out["dense_rows"] - out["sparse_rows"]
    out["dense_minus_sparse_locations"] = out["dense_locations"] - out["sparse_locations"]
    out["dense_minus_sparse_region5_cells"] = out["dense_region5_cells"] - out["sparse_region5_cells"]
    return out


def _cell_metric_table(frame: pd.DataFrame, scopes: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Build all-cell and top-five-per-model SSE tables plus conservation checks."""
    all_parts: list[pd.DataFrame] = []
    checks: list[dict[str, Any]] = []
    for scope, subset in scopes.items():
        if subset.empty:
            continue
        grouped = subset.groupby(
            ["source_month", "target_month", "cell_lat5", "cell_lon5"],
            sort=True,
            dropna=False,
        )
        rows: list[dict[str, Any]] = []
        for key, group in grouped:
            source_month, target_month, cell_lat5, cell_lon5 = key
            row: dict[str, Any] = {
                "scope": scope,
                "source_month": source_month,
                "target_month": target_month,
                "cell_lat5": cell_lat5,
                "cell_lon5": cell_lon5,
                "rows": int(len(group)),
            }
            for label in ("a", "b", "c", "persistence"):
                row[f"{label}_sse"] = float(np.sum(np.square(group[f"{label}_prediction"].to_numpy(dtype=np.float64) - group["target"].to_numpy(dtype=np.float64))))
            rows.append(row)
        cells = pd.DataFrame(rows)
        for label in ("a", "b", "c", "persistence"):
            cells[f"{label}_sse_rank"] = cells.groupby("source_month")[f"{label}_sse"].rank(method="first", ascending=False).astype(int)
        all_parts.append(cells)
        for (source_month, target_month), group in subset.groupby(["source_month", "target_month"], sort=True):
            for label in ("a", "b", "c", "persistence"):
                total = float(np.sum(np.square(group[f"{label}_prediction"].to_numpy(dtype=np.float64) - group["target"].to_numpy(dtype=np.float64))))
                cell_total = float(cells.loc[cells["source_month"].eq(source_month) & cells["target_month"].eq(target_month), f"{label}_sse"].sum())
                checks.append({"scope": scope, "source_month": source_month, "target_month": target_month, "model": label, "row_sse": total, "cell_sse": cell_total, "abs_difference": abs(total - cell_total), "ok": bool(np.isclose(total, cell_total, atol=1e-8, rtol=1e-12))})
    all_cells = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    if all_cells.empty:
        return all_cells, all_cells, checks
    rank_columns = [f"{label}_sse_rank" for label in ("a", "b", "c", "persistence")]
    top = all_cells.loc[all_cells[rank_columns].min(axis=1) <= 5].copy()
    top = top.sort_values(["scope", "source_month", "a_sse_rank", "b_sse_rank", "c_sse_rank"], kind="mergesort")
    return all_cells, top, checks


def _tree_gain_groups(booster: Any, feature_names: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize saved LightGBM split gains descriptively by feature family."""
    dump = booster.dump_model()
    gains: dict[str, float] = {}
    counts: dict[str, int] = {}
    leaf_counts: list[int] = []

    def family(name: str) -> str:
        lower = name.lower()
        if "last_observed_tws" in lower or lower.startswith("tws_") or "tws_anchor" in lower:
            return "legal_tws_anchor_or_history"
        if any(token in lower for token in ("reg", "dev", "count", "coverage", "regional")):
            return "regional_context_or_support"
        if any(token in lower for token in ("spei", "soil", "gap_delta", "trail", "drying", "recovery", "slope")):
            return "local_hydro_or_trajectory"
        if any(token in lower for token in ("lat", "lon", "geo", "h")):
            return "geography_or_horizon"
        if "month" in lower or "sin" in lower or "cos" in lower:
            return "calendar"
        return "other"

    def walk(node: dict[str, Any]) -> None:
        split = node.get("split_feature")
        gain = float(node.get("split_gain", 0.0) or 0.0)
        if split is not None:
            if isinstance(split, (int, np.integer)) and feature_names and 0 <= int(split) < len(feature_names):
                name = feature_names[int(split)]
            else:
                name = str(split)
            key = family(name)
            gains[key] = gains.get(key, 0.0) + gain
            counts[key] = counts.get(key, 0) + 1
        for child in node.get("left_child",), node.get("right_child",):
            if isinstance(child, dict):
                walk(child)

    def leaves(node: dict[str, Any]) -> int:
        left = node.get("left_child")
        right = node.get("right_child")
        if not isinstance(left, dict) and not isinstance(right, dict):
            return 1
        return (leaves(left) if isinstance(left, dict) else 0) + (leaves(right) if isinstance(right, dict) else 0)

    for tree in dump.get("tree_info", []):
        root = tree.get("tree_structure", {})
        if isinstance(root, dict):
            walk(root)
            leaf_counts.append(leaves(root))
    total = sum(gains.values())
    table = pd.DataFrame([{"feature_group": key, "split_count": int(counts.get(key, 0)), "split_gain": float(value), "split_gain_share": (float(value) / total if total else None)} for key, value in sorted(gains.items(), key=lambda kv: (-kv[1], kv[0]))])
    metadata = {
        "tree_count": int(len(dump.get("tree_info", []))),
        "total_split_gain": float(total),
        "leaf_counts": leaf_counts,
        "total_leaves": int(sum(leaf_counts)),
        "mean_leaves": float(np.mean(leaf_counts)) if leaf_counts else None,
        "min_leaves": int(min(leaf_counts)) if leaf_counts else None,
        "max_leaves": int(max(leaf_counts)) if leaf_counts else None,
    }
    return table, metadata


def stage_full_fit(args: argparse.Namespace) -> None:
    path, manifest, started = _stage_start(args.run_dir, "full_fit", args.expected_commit)
    try:
        import lightgbm as lgb

        fit_dir = args.model_dir.resolve()
        model_path = fit_dir / "model.txt"
        config_path = fit_dir / "resolved_config.json"
        schema_path = fit_dir / "feature_schema.json"
        for required in (model_path, config_path, schema_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        resolved = json.loads(config_path.read_text(encoding="utf-8"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        train = _load_train(args.data_dir, hydro=True)
        train_structural, train_source, train_labels = _build_internal_train(train)
        plain_structural = train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
        plain_source = train.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
        plain_labels = train.loc[:, ["sample_id", "target"]].copy()
        max_target_month = (_period_number(train["time"]) + 1).max()
        sampled = build_sampled_training_rows(plain_structural, plain_source.loc[:, SOURCE_CORE_COLUMNS], max_target_month=_period_label(int(max_target_month)), seed=20260908)
        rows_plain = sampled.rows
        y_delta = _attach_delta(rows_plain, plain_labels).astype(np.float32)
        weights = np.asarray(horizon_rebalance_weights(rows_plain["h"]), dtype=np.float32)
        expected_training = resolved.get("training", {})
        training_hash_checks = {
            "row_id_hash": {"actual": _hash_values(rows_plain["sample_id"]), "expected": expected_training.get("row_id_hash")},
            "label_hash": {"actual": _array_hash(y_delta), "expected": expected_training.get("label_hash")},
            "weight_hash": {"actual": _array_hash(weights), "expected": expected_training.get("weight_hash")},
            "rows": {"actual": int(len(rows_plain)), "expected": expected_training.get("rows")},
            "dropped_missing_anchor": {"actual": int(sampled.dropped_missing_anchor), "expected": expected_training.get("dropped_missing_anchor")},
        }
        if any(item["expected"] is not None and item["actual"] != item["expected"] for item in training_hash_checks.values()):
            raise AssertionError({"training_hash_checks": training_hash_checks})
        rows = rows_plain.copy()
        rows["sample_id"] = _namespace("tr__", rows["sample_id"])
        maps = build_b3_feature_maps(train_source)
        booster = lgb.Booster(model_file=str(model_path))
        feature_names = list(schema.get("feature_names", []))
        if not feature_names:
            raise AssertionError("saved feature schema has no feature_names")
        aggregate_parts: list[pd.DataFrame] = []
        total_sse = total_persistence_sse = total_abs = total_bias = 0.0
        for start_index in range(0, len(rows), int(args.batch_size)):
            stop_index = min(start_index + int(args.batch_size), len(rows))
            chunk = rows.iloc[start_index:stop_index].reset_index(drop=True)
            x = build_b3_matrix(chunk, train_source, train_structural, maps)
            if x.columns.tolist() != feature_names:
                raise AssertionError("saved model feature schema differs from rebuilt full-fit features")
            pred_delta = np.asarray(booster.predict(x), dtype=np.float64)
            truth_delta = y_delta[start_index:stop_index].astype(np.float64)
            error = pred_delta - truth_delta
            persistence_error = -truth_delta
            source_period = rows_plain.iloc[start_index:stop_index]["source_period"].astype(str).to_numpy()
            source_year = np.asarray([int(value[:4]) for value in source_period], dtype=np.int16)
            geo5 = (np.floor(rows_plain.iloc[start_index:stop_index]["lat"].to_numpy(dtype=float) / 5.0) * 5.0).astype(float).astype(str) + "," + (np.floor(rows_plain.iloc[start_index:stop_index]["lon"].to_numpy(dtype=float) / 5.0) * 5.0).astype(float).astype(str)
            part = pd.DataFrame({"source_month": source_period, "source_year": source_year, "h": rows_plain.iloc[start_index:stop_index]["h"].to_numpy(dtype=np.int16), "geo5": geo5, "error": error, "persistence_error": persistence_error})
            aggregate_parts.append(_aggregate_error_frame(part, ["source_month", "source_year", "h", "geo5"]))
            total_sse += float(np.sum(np.square(error)))
            total_persistence_sse += float(np.sum(np.square(persistence_error)))
            total_abs += float(np.sum(np.abs(error)))
            total_bias += float(np.sum(error))
            del x, part, pred_delta, truth_delta, error, persistence_error
            if start_index // int(args.batch_size) % 10 == 0:
                print(json.dumps({"stage": "full_fit", "rows_scored": stop_index, "total_rows": len(rows), "memory": _memory()}, sort_keys=True), flush=True)
        additive_columns = [
            "rows", "b3_sse", "b3_abs_error_sum", "b3_error_sum",
            "persistence_sse", "persistence_abs_error_sum", "persistence_error_sum",
        ]
        aggregates = pd.concat(aggregate_parts, ignore_index=True).groupby(
            ["source_month", "source_year", "h", "geo5"], as_index=False, sort=True
        )[additive_columns].sum()
        aggregates = _finalize_error_metrics(aggregates)
        source_table = aggregates.groupby(
            ["source_month", "source_year", "h"], as_index=False, sort=True
        )[additive_columns].sum()
        source_table = _finalize_error_metrics(source_table)
        source_table.to_csv(args.run_dir / "fit_by_source_month.csv", index=False)
        geo_table = aggregates.groupby(
            ["geo5", "h"], as_index=False, sort=True
        )[additive_columns].sum()
        geo_table = _finalize_error_metrics(geo_table)
        geo_table.to_csv(args.run_dir / "fit_by_geo5.csv", index=False)
        late_geo = aggregates.loc[aggregates["source_year"].eq(2015)].copy()
        late_geo.to_csv(args.run_dir / "fit_late_2015_by_geo5.csv", index=False)
        gain_table, tree_metadata = _tree_gain_groups(booster, feature_names)
        gain_table.to_csv(args.run_dir / "fit_tree_gain_groups.csv", index=False)
        expected_preflight = resolved.get("data_preflight", {}).get("sha256", {})
        current_hashes = {name: _sha256(args.data_dir / name) for name in ("Train.csv", "Test.csv", "SampleSubmission.csv")}
        hash_matches = {name: (expected_preflight.get(name) is None or expected_preflight.get(name) == value) for name, value in current_hashes.items()}
        result = {
            "model": {"path": str(model_path), "sha256": _sha256(model_path), "bytes": int(model_path.stat().st_size), "tree_metadata": tree_metadata},
            "feature_schema": {"count": int(len(feature_names)), "schema_sha256": _hash_values(feature_names), "expected_schema_sha256": schema.get("schema_sha256")},
            "training_reconstruction": {"rows": int(len(rows_plain)), "horizon_counts": {str(k): int(v) for k, v in rows_plain["h"].value_counts().sort_index().items()}, "hash_checks": training_hash_checks},
            "data_hashes": {"current": current_hashes, "expected_from_fit": expected_preflight, "matches": hash_matches},
            "in_sample": {"rows": int(len(rows_plain)), "b3_rmse": float(np.sqrt(total_sse / len(rows_plain))), "b3_mae": float(total_abs / len(rows_plain)), "b3_bias": float(total_bias / len(rows_plain)), "persistence_rmse": float(np.sqrt(total_persistence_sse / len(rows_plain))), "b3_minus_persistence_rmse": float(np.sqrt(total_sse / len(rows_plain)) - np.sqrt(total_persistence_sse / len(rows_plain)))},
            "by_source_month_csv": str(args.run_dir / "fit_by_source_month.csv"),
            "by_geo5_csv": str(args.run_dir / "fit_by_geo5.csv"),
            "late_geo5_csv": str(args.run_dir / "fit_late_2015_by_geo5.csv"),
            "tree_gain_csv": str(args.run_dir / "fit_tree_gain_groups.csv"),
        }
        _json(args.run_dir / "full_fit_details.json", result)
        del maps, train, train_source, train_structural, rows, aggregate_parts, aggregates
        gc.collect()
        _stage_finish(path, manifest, started, details_path=str(args.run_dir / "full_fit_details.json"), fit=result)
    except Exception as exc:
        _stage_fail(path, manifest, started, exc)
        raise


def stage_tree_inspection(args: argparse.Namespace) -> None:
    """Inspect saved trees without rereading the challenge data."""
    path, manifest, started = _stage_start(args.run_dir, "tree_inspection", args.expected_commit)
    try:
        import lightgbm as lgb

        model_path = args.model_dir.resolve() / "model.txt"
        schema_path = args.model_dir.resolve() / "feature_schema.json"
        if not model_path.is_file() or not schema_path.is_file():
            raise FileNotFoundError("saved model/schema is missing")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        names = list(schema.get("feature_names", []))
        booster = lgb.Booster(model_file=str(model_path))
        table, metadata = _tree_gain_groups(booster, names)
        table.to_csv(args.run_dir / "fit_tree_gain_groups.csv", index=False)
        result = {"model_path": str(model_path), "model_sha256": _sha256(model_path), "feature_count": len(names), "tree_metadata": metadata, "tree_gain_csv": str(args.run_dir / "fit_tree_gain_groups.csv")}
        _json(args.run_dir / "tree_inspection_details.json", result)
        _stage_finish(path, manifest, started, details_path=str(args.run_dir / "tree_inspection_details.json"), trees=result)
    except Exception as exc:
        _stage_fail(path, manifest, started, exc)
        raise


def _find_d0_replay_package(oof_root: Path, origin: str) -> Path:
    token = origin.replace("-", "_")
    candidates = sorted(
        path for path in oof_root.rglob(f"*_D0_{token}")
        if path.is_dir() and (path / "model.txt").is_file() and (path / "oof.csv.gz").is_file()
    )
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one D0 replay package for {origin}, found {candidates}")
    return candidates[0]


def _namespace_frame(frame: pd.DataFrame, prefix: str = "tr__") -> pd.DataFrame:
    out = frame.copy()
    if "sample_id" not in out.columns:
        raise ValueError("frame must contain sample_id before namespacing")
    out["sample_id"] = _namespace(prefix, out["sample_id"])
    return out


def _predict_saved_b3(
    booster: Any,
    ledger: pd.DataFrame,
    source: pd.DataFrame,
    structural: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    maps = build_b3_feature_maps(source)
    features = build_b3_matrix(ledger, source, structural, maps)
    if features.columns.tolist() != feature_names:
        raise AssertionError("saved model feature schema differs from rebuilt replay features")
    prediction = ledger["last_observed_TWS"].to_numpy(dtype=np.float64) + np.asarray(
        booster.predict(features), dtype=np.float64
    )
    return prediction, features


def stage_matched_comparison(args: argparse.Namespace) -> None:
    """Compare historical D0 and full B3 under one exact replay ledger."""
    path, manifest, started = _stage_start(args.run_dir, "matched_comparison", args.expected_commit)
    try:
        import lightgbm as lgb

        origin = str(args.matched_origin)
        if origin != "2014-12":
            raise ValueError("matched_comparison currently requires --matched-origin 2014-12")
        d0_dir = (
            args.d0_model_dir.resolve()
            if args.d0_model_dir is not None
            else _find_d0_replay_package(args.oof_root.resolve(), origin)
        )
        d0_model_path = d0_dir / "model.txt"
        d0_oof_path = args.d0_oof.resolve() if args.d0_oof is not None else d0_dir / "oof.csv.gz"
        d0_manifest_path = d0_dir / "manifest.json"
        for required in (d0_model_path, d0_oof_path, d0_manifest_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        d0_manifest = json.loads(d0_manifest_path.read_text(encoding="utf-8"))
        full_dir = args.model_dir.resolve()
        full_model_path = full_dir / "model.txt"
        full_schema_path = full_dir / "feature_schema.json"
        full_config_path = full_dir / "resolved_config.json"
        for required in (full_model_path, full_schema_path, full_config_path):
            if not required.is_file():
                raise FileNotFoundError(required)

        oof_columns = [
            "sample_id", "source_date", "last_observed_date", "last_observed_TWS", "h",
            "lat", "lon", "target", "prediction", "anchor_age_months",
        ]
        oof = pd.read_csv(d0_oof_path, usecols=oof_columns)
        oof["sample_id"] = oof["sample_id"].astype(str)
        if oof["sample_id"].duplicated().any():
            raise AssertionError("D0 replay OOF IDs are not unique")
        oof["h"] = oof["h"].astype(int)

        train = _load_train(args.data_dir, hydro=True)
        train["sample_id"] = train["sample_id"].astype(str)
        train_plain_structural = train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
        train_plain_source = train.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
        train_plain_source["sample_id"] = train_plain_source["sample_id"].astype(str)
        train_plain_labels = train.loc[:, ["sample_id", "target"]].copy()
        train_plain_labels["sample_id"] = train_plain_labels["sample_id"].astype(str)

        full_fold = build_mask_block_fold(
            train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t", "target"]],
            anchor_month=origin,
            end_month="2015-06",
            scenario_id="matched_leaderboard_2014_12",
            family="recent_mask_block",
        )
        full_ledger = full_fold.ledger.copy()
        full_ledger["sample_id"] = full_ledger["sample_id"].astype(str)
        if full_ledger["sample_id"].duplicated().any():
            raise AssertionError("reconstructed replay ledger IDs are not unique")
        missing_oof = sorted(set(oof["sample_id"]) - set(full_ledger["sample_id"]))
        if missing_oof:
            raise AssertionError(f"D0 OOF IDs absent from reconstructed legal ledger: {len(missing_oof)}")
        # Preserve the saved OOF row order while taking every other field from the
        # freshly reconstructed legal simulator ledger.
        replay = full_ledger.set_index("sample_id").loc[oof["sample_id"].tolist()].reset_index()
        if len(replay) != len(oof):
            raise AssertionError("reconstructed replay row count differs from OOF")

        identity = replay.loc[:, ["sample_id", "source_date", "last_observed_date", "last_observed_TWS", "h", "lat", "lon"]].merge(
            oof.loc[:, ["sample_id", "source_date", "last_observed_date", "last_observed_TWS", "h", "lat", "lon", "target"]],
            on="sample_id", how="outer", validate="one_to_one", sort=False, suffixes=("_replay", "_oof"),
        )
        if identity.isna().any(axis=None):
            raise AssertionError("replay/OOF identity join has missing fields")
        date_checks: dict[str, Any] = {}
        for column in ("source_date", "last_observed_date"):
            left = pd.to_datetime(identity[f"{column}_replay"]).dt.strftime("%Y-%m-%d")
            right = pd.to_datetime(identity[f"{column}_oof"]).dt.strftime("%Y-%m-%d")
            date_checks[column] = {"equal": bool(np.array_equal(left.to_numpy(), right.to_numpy())), "mismatched_rows": int((left != right).sum())}
        for column in ("h", "lat", "lon"):
            left = identity[f"{column}_replay"].to_numpy(dtype=np.float64)
            right = identity[f"{column}_oof"].to_numpy(dtype=np.float64)
            date_checks[column] = {"equal": bool(np.array_equal(left, right)), "max_abs_difference": float(np.max(np.abs(left - right)))}
        anchor_diff = identity["last_observed_TWS_replay"].to_numpy(dtype=np.float64) - identity["last_observed_TWS_oof"].to_numpy(dtype=np.float64)
        target_lookup = train_plain_labels.set_index("sample_id")["target"]
        train_target = target_lookup.loc[oof["sample_id"]].to_numpy(dtype=np.float64)
        target_diff = train_target - oof["target"].to_numpy(dtype=np.float64)
        identity_checks = {
            "rows": int(len(identity)),
            "oof_ids_equal_reconstructed_ledger_subset": bool(set(oof["sample_id"]) == set(replay["sample_id"])),
            "date_and_scalar_checks": date_checks,
            "anchor_value_max_abs_difference": float(np.max(np.abs(anchor_diff))) if len(anchor_diff) else 0.0,
            "target_train_vs_oof_max_abs_difference": float(np.max(np.abs(target_diff))) if len(target_diff) else 0.0,
        }
        if any(not value["equal"] for value in date_checks.values() if "equal" in value) or identity_checks["anchor_value_max_abs_difference"] > 1e-6 or identity_checks["target_train_vs_oof_max_abs_difference"] > 1e-6:
            raise AssertionError({"replay_oof_identity": identity_checks})

        sparse_view = build_replay_observation_view(
            train_plain_source,
            train_plain_structural,
            ledger=replay,
            first_source_month=full_fold.spec.first_source_month,
            last_source_month=full_fold.spec.last_source_month,
        )
        dense_source = train_plain_source.copy()
        source_support = _source_view_month_table(
            sparse_view.source,
            dense_source,
            first_source_month=full_fold.spec.first_source_month,
            last_source_month=full_fold.spec.last_source_month,
        )

        d0_booster = lgb.Booster(model_file=str(d0_model_path))
        d0_feature_names = list(d0_manifest.get("feature_names") or d0_booster.feature_name())
        if len(d0_feature_names) != int(d0_booster.num_feature()):
            raise AssertionError("D0 feature schema count differs from saved model")
        a_prediction, a_features = _predict_saved_b3(
            d0_booster,
            replay,
            sparse_view.source,
            sparse_view.structural,
            d0_feature_names,
        )
        a_difference = a_prediction - oof["prediction"].to_numpy(dtype=np.float64)
        a_reproduction = {
            "rows": int(len(a_prediction)),
            "max_abs_difference_to_saved_oof": float(np.max(np.abs(a_difference))) if len(a_difference) else 0.0,
            "within_1e-6": bool(not len(a_difference) or np.max(np.abs(a_difference)) <= 1e-6),
            "d0_model_sha256": _sha256(d0_model_path),
            "d0_oof_sha256": _sha256(d0_oof_path),
            "d0_feature_count": int(len(d0_feature_names)),
        }
        if not a_reproduction["within_1e-6"]:
            raise AssertionError({"A_d0_reproduction": a_reproduction})
        del a_features, d0_booster
        gc.collect()

        full_schema = json.loads(full_schema_path.read_text(encoding="utf-8"))
        full_feature_names = list(full_schema.get("feature_names", []))
        if len(full_feature_names) != 446:
            raise AssertionError(f"full B3 schema must contain 446 features, found {len(full_feature_names)}")
        full_booster = lgb.Booster(model_file=str(full_model_path))
        if int(full_booster.num_feature()) != len(full_feature_names):
            raise AssertionError("full B3 model/schema feature counts differ")
        # Use the same plain replay IDs for A, B, and C.  The full submission's
        # ``tr__`` namespace is only a join-safety convention; it is not a model
        # feature and must not become a comparison intervention.
        replay_input = replay.copy()
        sparse_source_input = sparse_view.source.copy()
        sparse_structural_input = sparse_view.structural.copy()
        dense_source_input = dense_source.copy()

        b_prediction, b_features = _predict_saved_b3(
            full_booster,
            replay_input,
            sparse_source_input,
            sparse_structural_input,
            full_feature_names,
        )
        b_base = b_features.loc[:, HYDRO_GAP_SAFE_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        gc.collect()

        c_prediction, c_features = _predict_saved_b3(
            full_booster,
            replay_input,
            dense_source_input,
            sparse_structural_input,
            full_feature_names,
        )
        c_base = c_features.loc[:, HYDRO_GAP_SAFE_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
        base_equal = bool(np.array_equal(b_base, c_base, equal_nan=True))
        base_difference = np.nan_to_num(b_base, nan=0.0) - np.nan_to_num(c_base, nan=0.0)
        base_max_abs_difference = float(np.max(np.abs(base_difference))) if base_difference.size else 0.0
        changed_features: list[dict[str, Any]] = []
        for column in full_feature_names:
            left = b_features[column].to_numpy(dtype=np.float32)
            right = c_features[column].to_numpy(dtype=np.float32)
            same = np.array_equal(left, right, equal_nan=True)
            if not same:
                delta = np.nan_to_num(left, nan=0.0) - np.nan_to_num(right, nan=0.0)
                changed_features.append({"feature": column, "changed_rows": int(np.sum(~np.isclose(left, right, equal_nan=True, atol=0.0, rtol=0.0))), "max_abs_difference": float(np.max(np.abs(delta)))})
        ledger_same = True
        for column in ("sample_id", "source_date", "last_observed_date", "last_observed_TWS", "h", "lat", "lon"):
            left = replay[column]
            right = replay_input[column]
            if column in ("sample_id", "source_date", "last_observed_date"):
                equal = bool(np.array_equal(left.astype(str).to_numpy(), right.astype(str).to_numpy()))
            else:
                equal = bool(np.array_equal(left.to_numpy(dtype=np.float64), right.to_numpy(dtype=np.float64)))
            ledger_same = ledger_same and equal
        structural_clone = sparse_structural_input.copy()
        structural_same = True
        for column in sparse_structural_input.columns:
            left = sparse_structural_input[column]
            right = structural_clone[column]
            if left.dtype.kind in "OUS" or right.dtype.kind in "OUS":
                equal = bool(np.array_equal(left.astype(str).to_numpy(), right.astype(str).to_numpy()))
            else:
                equal = bool(np.array_equal(left.to_numpy(), right.to_numpy(), equal_nan=True))
            structural_same = structural_same and equal
        feature_view_checks = {
            "B_sparse_source_rows": int(len(sparse_view.source)),
            "B_sparse_structural_rows": int(len(sparse_view.structural)),
            "C_dense_source_rows": int(len(dense_source)),
            "sparse_withheld_window_rows": int(sparse_view.withheld_window_rows),
            "B_C_same_replay_ledger": ledger_same,
            "B_C_same_legal_structural_panel": structural_same,
            "B_C_base_feature_equal": base_equal,
            "B_C_base_feature_max_abs_difference": base_max_abs_difference,
            "B_C_changed_feature_count": int(len(changed_features)),
            "B_C_changed_features": changed_features,
            "C_interpretation": "retrospective dense historical covariate view; not deployable when those rows are unavailable",
        }
        del full_booster, b_features, c_features, b_base, c_base, replay_input, sparse_source_input, sparse_structural_input, dense_source_input, structural_clone
        gc.collect()

        exposure_training = build_sampled_training_rows(
            train_plain_structural,
            train_plain_source.loc[:, SOURCE_CORE_COLUMNS],
            max_target_month=pd.to_datetime(train["time"]).dt.to_period("M").add(1).max(),
            seed=20260908,
        )
        y_delta = _attach_delta(exposure_training.rows, train_plain_labels).astype(np.float32)
        full_weights = np.asarray(horizon_rebalance_weights(exposure_training.rows["h"]), dtype=np.float32)
        resolved = json.loads(full_config_path.read_text(encoding="utf-8"))
        expected_training = resolved.get("training", {})
        training_hash_checks = {
            "row_id_hash": {"actual": _hash_values(exposure_training.rows["sample_id"]), "expected": expected_training.get("row_id_hash")},
            "label_hash": {"actual": _array_hash(y_delta), "expected": expected_training.get("label_hash")},
            "weight_hash": {"actual": _array_hash(full_weights), "expected": expected_training.get("weight_hash")},
            "rows": {"actual": int(len(exposure_training.rows)), "expected": expected_training.get("rows")},
            "dropped_missing_anchor": {"actual": int(exposure_training.dropped_missing_anchor), "expected": expected_training.get("dropped_missing_anchor")},
        }
        if any(item["expected"] is not None and item["actual"] != item["expected"] for item in training_hash_checks.values()):
            raise AssertionError({"full_training_hash_checks": training_hash_checks})
        exposure, exposure_counts = _tag_training_exposure(replay, exposure_training.rows)

        event = exposure.copy()
        event["target"] = oof["target"].to_numpy(dtype=np.float64)
        event["a_prediction"] = a_prediction
        event["b_prediction"] = b_prediction
        event["c_prediction"] = c_prediction
        event["persistence_prediction"] = replay["last_observed_TWS"].to_numpy(dtype=np.float64)
        event["source_month"] = pd.to_datetime(event["source_date"]).dt.to_period("M").astype(str)
        event["target_month"] = (pd.to_datetime(event["source_date"]) + pd.offsets.MonthBegin(1)).dt.to_period("M").astype(str)
        event["anchor_age_months"] = (
            pd.to_datetime(event["source_date"]).dt.to_period("M").astype("int64")
            - pd.to_datetime(event["last_observed_date"]).dt.to_period("M").astype("int64")
        ).astype(int)
        event["cell_lat5"] = np.floor(event["lat"].astype(float) / 5.0) * 5.0
        event["cell_lon5"] = np.floor(event["lon"].astype(float) / 5.0) * 5.0
        scopes = {
            "complete_replay": event,
            "exact_exposure": event.loc[event["exact_training_exposure"]].copy(),
            "different_anchor": event.loc[event["exposure_status"] == "different_anchor"].copy(),
        }
        overall = _comparison_metric_table(event, [], scopes)
        by_month_h = _comparison_metric_table(event, ["source_month", "target_month", "h"], scopes)
        by_age = _comparison_metric_table(event, ["anchor_age_months"], scopes)
        all_cells, top_cells, cell_checks = _cell_metric_table(event, scopes)
        support = event.groupby(["source_month", "target_month", "h"], sort=True).agg(
            rows=("sample_id", "size"),
            exact_exposure_rows=("exact_training_exposure", "sum"),
            source_id_in_full_training_rows=("source_id_in_full_training", "sum"),
        ).reset_index()
        support["different_anchor_rows"] = support["source_id_in_full_training_rows"] - support["exact_exposure_rows"]
        exposure_status_table = event["exposure_status"].value_counts().rename_axis("exposure_status").reset_index(name="rows")
        source_support.to_csv(args.run_dir / "matched_source_view_by_month.csv", index=False)
        overall.to_csv(args.run_dir / "matched_overall.csv", index=False)
        by_month_h.to_csv(args.run_dir / "matched_by_month_h.csv", index=False)
        by_age.to_csv(args.run_dir / "matched_by_anchor_age.csv", index=False)
        support.to_csv(args.run_dir / "matched_month_h_support.csv", index=False)
        exposure_status_table.to_csv(args.run_dir / "matched_exposure_status.csv", index=False)
        all_cells.to_csv(args.run_dir / "matched_cells_all.csv", index=False)
        top_cells.to_csv(args.run_dir / "matched_cells_top5.csv", index=False)
        pd.DataFrame(cell_checks).to_csv(args.run_dir / "matched_cell_sse_checks.csv", index=False)

        decomposition_columns = [column for column in overall.columns if column.endswith("_decomposition_error")]
        decomposition_max = max(
            (float(np.max(np.abs(overall[column].dropna().to_numpy(dtype=np.float64)))) for column in decomposition_columns if not overall[column].dropna().empty),
            default=0.0,
        )
        focus = by_month_h.loc[
            (by_month_h["scope"] == "complete_replay")
            & by_month_h["source_month"].isin(["2015-01", "2015-02", "2015-06"])
        ].copy()
        result = {
            "origin": origin,
            "replay_window": {"first_source_month": full_fold.spec.first_source_month, "last_source_month": full_fold.spec.last_source_month},
            "d0_package": str(d0_dir),
            "d0_model": {"path": str(d0_model_path), "sha256": _sha256(d0_model_path)},
            "d0_oof": {"path": str(d0_oof_path), "sha256": _sha256(d0_oof_path), "rows": int(len(oof))},
            "full_model": {"path": str(full_model_path), "sha256": _sha256(full_model_path), "feature_schema_sha256": full_schema.get("schema_sha256")},
            "replay_oof_identity": identity_checks,
            "A_d0_sparse_replay": a_reproduction,
            "B_full_model_sparse_replay_vs_C_dense": feature_view_checks,
            "full_training_reconstruction": {"rows": int(len(exposure_training.rows)), "horizon_counts": {str(k): int(v) for k, v in exposure_training.rows["h"].value_counts().sort_index().items()}, "hash_checks": training_hash_checks},
            "exposure_counts": exposure_counts,
            "metrics": {"overall": overall.to_dict(orient="records"), "focus_2015_months": focus.to_dict(orient="records")},
            "cell_sse_conservation": {"checks": cell_checks, "all_ok": bool(all(row["ok"] for row in cell_checks)), "max_abs_difference": max((float(row["abs_difference"]) for row in cell_checks), default=0.0)},
            "max_abs_rmse_decomposition_error": decomposition_max,
            "output_files": {
                "source_view_by_month": str(args.run_dir / "matched_source_view_by_month.csv"),
                "overall": str(args.run_dir / "matched_overall.csv"),
                "by_month_h": str(args.run_dir / "matched_by_month_h.csv"),
                "by_anchor_age": str(args.run_dir / "matched_by_anchor_age.csv"),
                "month_h_support": str(args.run_dir / "matched_month_h_support.csv"),
                "exposure_status": str(args.run_dir / "matched_exposure_status.csv"),
                "cells_all": str(args.run_dir / "matched_cells_all.csv"),
                "cells_top5": str(args.run_dir / "matched_cells_top5.csv"),
                "cell_sse_checks": str(args.run_dir / "matched_cell_sse_checks.csv"),
            },
            "no_row_level_prediction_file_written": True,
            "interpretation_limits": [
                "A_vs_B measures the joint effect of later training exposure and the resulting fitted model, not a pure causal recency effect.",
                "B_vs_C is a fixed-model retrospective availability intervention; C is not deployable if dense covariates are unavailable.",
                "Horizon-weighted values are validation proxies and h>7 rows remain a separate stress tail.",
            ],
        }
        _json(args.run_dir / "matched_comparison_details.json", result)
        del train, train_plain_structural, train_plain_source, train_plain_labels, replay, exposure, event, scopes, source_support
        gc.collect()
        _stage_finish(path, manifest, started, details_path=str(args.run_dir / "matched_comparison_details.json"), matched=result)
    except Exception as exc:
        _stage_fail(path, manifest, started, exc)
        raise


def _build_source_panel(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train_source = train.loc[:, ["sample_id", "time", "lat", "lon", *HYDRO_COLUMNS]].copy()
    train_source["_is_test"] = False
    test_source = test.loc[:, ["ID", "time", "lat", "lon", *HYDRO_COLUMNS]].rename(columns={"ID": "sample_id"}).copy()
    test_source["_is_test"] = True
    panel = pd.concat([train_source, test_source], ignore_index=True, sort=False)
    panel["sample_id"] = panel["sample_id"].astype(str)
    panel["_period"] = _period_number(panel["time"])
    if panel.duplicated(["lat", "lon", "_period"]).any():
        raise AssertionError("combined source has duplicate location/calendar rows")
    return panel


def _attach_history_counts(panel: pd.DataFrame, requests: pd.DataFrame, *, request_id: str = "ID") -> pd.DataFrame:
    """Attach local finite-observation counts in exact 3/6 calendar windows."""
    needed = [request_id, "lat", "lon", "source_period"]
    if set(needed).difference(requests.columns):
        raise ValueError(f"history requests missing {sorted(set(needed).difference(requests.columns))}")
    out = requests.loc[:, [request_id]].copy()
    out[request_id] = out[request_id].astype(str)
    out["_request_order"] = np.arange(len(out), dtype=np.int64)
    panel_sorted = panel.sort_values(["lat", "lon", "_period", "sample_id"], kind="mergesort").reset_index(drop=True)
    request_lookup = requests.loc[:, [request_id]].copy()
    request_lookup[request_id] = request_lookup[request_id].astype(str)
    request_lookup["_is_request"] = True
    request_lookup = request_lookup.rename(columns={request_id: "sample_id"})
    panel_sorted = panel_sorted.merge(request_lookup, on="sample_id", how="left", validate="one_to_one", sort=False)
    for variable in HYDRO_COLUMNS:
        values = panel_sorted[variable].to_numpy(dtype=np.float64, na_value=np.nan)
        for window in (3, 6):
            counts = np.zeros(len(panel_sorted), dtype=np.float32)
            for _, indices in panel_sorted.groupby(["lat", "lon"], sort=False).groups.items():
                positions = np.asarray(list(indices), dtype=np.int64)
                periods = panel_sorted.loc[positions, "_period"].to_numpy(dtype=np.int64)
                finite = np.isfinite(values[positions]).astype(np.int64)
                prefix = np.concatenate([[0], np.cumsum(finite)])
                starts = np.searchsorted(periods, periods - (window - 1), side="left")
                counts[positions] = (prefix[np.arange(len(positions)) + 1] - prefix[starts]).astype(np.float32)
            request_mask = panel_sorted["_is_request"].fillna(False).to_numpy(dtype=bool)
            picked = panel_sorted.loc[request_mask, ["sample_id"]].copy()
            picked[f"history_{variable}_{window}m_count"] = counts[request_mask]
            out = out.merge(picked.rename(columns={"sample_id": request_id}), on=request_id, how="left", validate="one_to_one", sort=False)
    out = out.sort_values("_request_order", kind="mergesort").drop(columns="_request_order").reset_index(drop=True)
    return out


def _support_metrics(frame: pd.DataFrame, group_columns: list[str], *, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {"population": label, **{column: value for column, value in zip(group_columns, key)}, "rows": int(len(group)), "anchor_age_mean": float(group["anchor_age_months"].mean()), "anchor_age_p95": float(group["anchor_age_months"].quantile(0.95)), "delta_vs_persistence_mean": float(group["delta_vs_persistence"].mean()), "delta_vs_persistence_abs_p95": float(group["delta_vs_persistence"].abs().quantile(0.95))}
        for variable in HYDRO_COLUMNS:
            for side in ("current", "anchor", "gap"):
                column = f"{side}_{variable}"
                row[f"{column}_finite_share"] = float(group[column].notna().mean())
                if side != "gap":
                    row[f"{column}_out_of_train_range_share"] = float(group[f"{side}_{variable}_oob"].mean())
            for window in (3, 6):
                count_column = f"history_{variable}_{window}m_count"
                if count_column in group:
                    row[f"{count_column}_mean"] = float(group[count_column].mean())
                    row[f"{count_column}_zero_share"] = float((group[count_column] == 0).mean())
        for width in (5, 15):
            for variable in HYDRO_COLUMNS:
                col = f"regional{width}_{variable}_count"
                if col in group:
                    row[f"{col}_mean"] = float(group[col].mean())
                    row[f"{col}_zero_share"] = float((group[col] == 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _prepare_support_rows(test: pd.DataFrame, ledger: pd.DataFrame, panel: pd.DataFrame, train_ranges: dict[str, tuple[float, float]], submission: pd.DataFrame | None = None) -> pd.DataFrame:
    meta = ledger.loc[:, ["ID", "source_date", "last_observed_date", "last_observed_TWS", "h"]].copy()
    meta["ID"] = meta["ID"].astype(str)
    meta["source_period"] = _period_number(meta["source_date"])
    meta["source_month"] = _period_labels(meta["source_period"]).to_numpy()
    meta["anchor_period"] = _period_number(meta["last_observed_date"])
    meta["anchor_age_months"] = (meta["source_period"] - meta["anchor_period"]).astype(float)
    current = test.loc[:, ["ID", *HYDRO_COLUMNS]].copy().rename(columns={c: f"current_{c}" for c in HYDRO_COLUMNS})
    current["ID"] = current["ID"].astype(str)
    meta = meta.merge(current, on="ID", how="left", validate="one_to_one", sort=False)
    lookup = panel.loc[:, ["lat", "lon", "_period", *HYDRO_COLUMNS]].rename(columns={"_period": "anchor_period", **{c: f"anchor_{c}" for c in HYDRO_COLUMNS}})
    # Coordinates are carried by the ledger; attach them before the exact key join.
    coords = ledger.loc[:, ["ID", "lat", "lon"]].copy(); coords["ID"] = coords["ID"].astype(str)
    meta = meta.merge(coords, on="ID", how="left", validate="one_to_one", sort=False)
    meta = meta.merge(lookup, on=["lat", "lon", "anchor_period"], how="left", validate="many_to_one", sort=False)
    for variable in HYDRO_COLUMNS:
        meta[f"gap_{variable}"] = meta[f"current_{variable}"] - meta[f"anchor_{variable}"]
        lo, hi = train_ranges[variable]
        for side in ("current", "anchor"):
            values = meta[f"{side}_{variable}"].to_numpy(dtype=np.float64, na_value=np.nan)
            meta[f"{side}_{variable}_oob"] = np.isfinite(values) & ((values < lo) | (values > hi))
    if submission is not None:
        sub = submission.loc[:, ["ID", "Target"]].copy(); sub["ID"] = sub["ID"].astype(str)
        meta = meta.merge(sub.rename(columns={"Target": "prediction"}), on="ID", how="left", validate="one_to_one", sort=False)
        meta["delta_vs_persistence"] = meta["prediction"] - meta["last_observed_TWS"]
    else:
        meta["prediction"] = np.nan; meta["delta_vs_persistence"] = np.nan
    return meta


def stage_test_support(args: argparse.Namespace) -> None:
    path, manifest, started = _stage_start(args.run_dir, "test_support", args.expected_commit)
    try:
        train = _load_train(args.data_dir, hydro=True)
        test = _load_test(args.data_dir, hydro=True)
        ledger_path = args.model_dir.resolve() / "test_visibility_ledger.csv.gz"
        if not ledger_path.is_file():
            raise FileNotFoundError(ledger_path)
        ledger = pd.read_csv(ledger_path, usecols=["ID", "source_date", "target_date", "lat", "lon", "last_observed_date", "last_observed_TWS", "h"])
        submission = pd.read_csv(args.submission.resolve(), usecols=["ID", "Target"]) if args.submission else None
        panel = _build_source_panel(train, test)
        train_ranges = {variable: (float(np.nanmin(train[variable].to_numpy(dtype=float))), float(np.nanmax(train[variable].to_numpy(dtype=float)))) for variable in HYDRO_COLUMNS}
        rows = _prepare_support_rows(test, ledger, panel, train_ranges, submission=submission)
        history = _attach_history_counts(panel, rows.loc[:, ["ID", "lat", "lon", "source_period"]].rename(columns={"source_period": "source_period"}), request_id="ID")
        rows = rows.merge(history.drop(columns=["lat", "lon", "source_period"], errors="ignore"), on="ID", how="left", validate="one_to_one", sort=False)
        # Contemporaneous regional support is calculated from actual source rows,
        # separately for each width and variable; no target or TWS is involved.
        panel["cell5_lat"] = np.floor(panel["lat"].astype(float) / 5.0) * 5.0; panel["cell5_lon"] = np.floor(panel["lon"].astype(float) / 5.0) * 5.0
        panel["cell15_lat"] = np.floor(panel["lat"].astype(float) / 15.0) * 15.0; panel["cell15_lon"] = np.floor(panel["lon"].astype(float) / 15.0) * 15.0
        regional_frames: list[pd.DataFrame] = []
        for width in (5, 15):
            keys = ["_period", f"cell{width}_lat", f"cell{width}_lon"]
            agg = panel.groupby(keys, sort=False)[list(HYDRO_COLUMNS)].count().reset_index()
            agg = agg.rename(columns={variable: f"regional{width}_{variable}_count" for variable in HYDRO_COLUMNS})
            regional_frames.append(agg)
        rows["_period"] = rows["source_period"]
        rows["cell5_lat"] = np.floor(rows["lat"].astype(float) / 5.0) * 5.0; rows["cell5_lon"] = np.floor(rows["lon"].astype(float) / 5.0) * 5.0
        rows["cell15_lat"] = np.floor(rows["lat"].astype(float) / 15.0) * 15.0; rows["cell15_lon"] = np.floor(rows["lon"].astype(float) / 15.0) * 15.0
        for width, agg in zip((5, 15), regional_frames, strict=True):
            keys = ["_period", f"cell{width}_lat", f"cell{width}_lon"]
            rows = rows.merge(agg, on=keys, how="left", validate="many_to_one", sort=False)
        test_table = _support_metrics(rows, ["source_month", "h"], label="Test")
        test_table.to_csv(args.run_dir / "test_support_by_month_h.csv", index=False)

        # The recent validation comparison uses the existing Dec-2014 OOF rows,
        # then applies exactly the same feature-support summaries to Train data.
        oof_files = _find_oof_files(args.oof_root.resolve())
        dec_files = [f for f in oof_files if f.parent.name.endswith("_2014_12")]
        validation_table = pd.DataFrame()
        if dec_files:
            oof = pd.read_csv(dec_files[0], usecols=["sample_id", "source_date", "last_observed_date", "last_observed_TWS", "h", "lat", "lon", "prediction"])
            oof = oof.rename(columns={"sample_id": "ID"}); oof["ID"] = oof["ID"].astype(str)
            oof["source_period"] = _period_number(oof["source_date"]); oof["source_month"] = _period_labels(oof["source_period"]).to_numpy(); oof["anchor_period"] = _period_number(oof["last_observed_date"]); oof["anchor_age_months"] = oof["source_period"] - oof["anchor_period"]
            current_lookup = train.loc[:, ["sample_id", *HYDRO_COLUMNS]].copy().rename(columns={"sample_id": "ID", **{c: f"current_{c}" for c in HYDRO_COLUMNS}}); current_lookup["ID"] = current_lookup["ID"].astype(str)
            oof = oof.merge(current_lookup, on="ID", how="left", validate="one_to_one", sort=False)
            oof = oof.merge(train.loc[:, ["lat", "lon", "time", *HYDRO_COLUMNS]].assign(_period=_period_number(train["time"])).rename(columns={"_period": "anchor_period", **{c: f"anchor_{c}" for c in HYDRO_COLUMNS}}), on=["lat", "lon", "anchor_period"], how="left", validate="many_to_one", sort=False)
            for variable in HYDRO_COLUMNS:
                oof[f"gap_{variable}"] = oof[f"current_{variable}"] - oof[f"anchor_{variable}"]
                lo, hi = train_ranges[variable]
                for side in ("current", "anchor"):
                    values = oof[f"{side}_{variable}"].to_numpy(dtype=np.float64, na_value=np.nan); oof[f"{side}_{variable}_oob"] = np.isfinite(values) & ((values < lo) | (values > hi))
            oof["delta_vs_persistence"] = oof["prediction"] - oof["last_observed_TWS"]
            validation_table = _support_metrics(oof, ["source_month", "h"], label="D0_2014_12")
        validation_table.to_csv(args.run_dir / "validation_support_recent.csv", index=False)
        result = {"test_rows": int(len(rows)), "ledger_rows": int(len(ledger)), "ledger_unique_ids": int(ledger["ID"].astype(str).nunique()), "train_ranges": train_ranges, "test_support_csv": str(args.run_dir / "test_support_by_month_h.csv"), "validation_support_csv": str(args.run_dir / "validation_support_recent.csv"), "submission_join": None if submission is None else {"rows": int(len(submission)), "finite_predictions": int(np.isfinite(submission["Target"].to_numpy(dtype=float)).sum()), "missing_join_predictions": int(rows["prediction"].isna().sum())}, "history_windows": [3, 6], "out_of_training_range_definition": "strictly below/above finite Train source covariate min/max; NaN reported separately"}
        _json(args.run_dir / "test_support_details.json", result)
        del panel, rows, train, test
        gc.collect()
        _stage_finish(path, manifest, started, details_path=str(args.run_dir / "test_support_details.json"), support=result)
    except Exception as exc:
        _stage_fail(path, manifest, started, exc)
        raise


def _prepare_competition_panels(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Rebuild the exact legal combined panels used by the frozen submission."""
    train_structural, train_source, _ = _build_internal_train(train)
    train_periods = _period_number(train["time"])
    max_training_target_month = int((train_periods + 1).max())
    ledger, test_structural, _, _visibility = _build_test_visibility(
        train_structural=train_structural,
        test=test,
        max_training_target_month=pd.Period(_period_label(max_training_target_month), freq="M"),
    )
    test_source = test.loc[:, ["ID", "time", "lat", "lon", "month_sin", "month_cos", *HYDRO_COLUMNS]].rename(columns={"ID": "sample_id"}).copy()
    test_source["sample_id"] = _namespace("te__", test_source["sample_id"])
    combined_source = pd.concat([train_source, test_source], ignore_index=True, sort=False)
    combined_structural = pd.concat([train_structural.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]], test_structural.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]]], ignore_index=True, sort=False)
    reverse_ids = dict(zip(_namespace("te__", test["ID"]), test["ID"].astype(str), strict=True))
    return ledger, test_structural, combined_source, combined_structural, train_structural, reverse_ids


def _predict_batches(booster: Any, ledger: pd.DataFrame, source: pd.DataFrame, structural: pd.DataFrame, maps: Any, *, batch_size: int, feature_names: list[str]) -> np.ndarray:
    prediction = np.empty(len(ledger), dtype=np.float64)
    for start_index in range(0, len(ledger), int(batch_size)):
        stop_index = min(start_index + int(batch_size), len(ledger))
        chunk = ledger.iloc[start_index:stop_index].reset_index(drop=True)
        features = build_b3_matrix(chunk, source, structural, maps)
        if features.columns.tolist() != feature_names:
            raise AssertionError("feature schema changed during sample inference")
        prediction[start_index:stop_index] = chunk["last_observed_TWS"].to_numpy(dtype=np.float64) + np.asarray(booster.predict(features), dtype=np.float64)
    return prediction


def _predict_feature_batches(booster: Any, ledger: pd.DataFrame, features: pd.DataFrame, *, batch_size: int) -> np.ndarray:
    """Predict an already-built matrix in chunks (no repeated keyed joins)."""
    if len(ledger) != len(features):
        raise ValueError("ledger and feature matrix lengths differ")
    prediction = np.empty(len(ledger), dtype=np.float64)
    for start_index in range(0, len(ledger), int(batch_size)):
        stop_index = min(start_index + int(batch_size), len(ledger))
        prediction[start_index:stop_index] = ledger.iloc[start_index:stop_index]["last_observed_TWS"].to_numpy(dtype=np.float64) + np.asarray(booster.predict(features.iloc[start_index:stop_index]), dtype=np.float64)
    return prediction


def _sample_test_ledger(ledger: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    x = ledger.copy()
    x["source_month"] = pd.to_datetime(x["source_date"]).dt.to_period("M").astype(str)
    x["_id_sort"] = x["sample_id"].astype(str)
    x = x.sort_values(["source_month", "h", "_id_sort"], kind="mergesort")
    groups = x.groupby(["source_month", "h"], sort=True, dropna=False)
    per_group = max(1, int(math.ceil(sample_size / max(1, groups.ngroups))))
    picked = groups.head(per_group)
    if len(picked) > sample_size:
        picked = picked.head(sample_size)
    return picked.drop(columns=["_id_sort"], errors="ignore").reset_index(drop=True)


def stage_sample_inference(args: argparse.Namespace) -> None:
    path, manifest, started = _stage_start(args.run_dir, "sample_inference", args.expected_commit)
    try:
        import lightgbm as lgb

        fit_dir = args.model_dir.resolve()
        model_path = fit_dir / "model.txt"
        schema_path = fit_dir / "feature_schema.json"
        if not model_path.is_file() or not schema_path.is_file():
            raise FileNotFoundError("saved B3 model/schema is missing")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        feature_names = list(schema.get("feature_names", []))
        train = _load_train(args.data_dir, hydro=True)
        test = _load_test(args.data_dir, hydro=True)
        ledger, test_structural, combined_source, combined_structural, train_structural, reverse_ids = _prepare_competition_panels(train, test)
        maps = build_b3_feature_maps(combined_source)
        booster = lgb.Booster(model_file=str(model_path))
        sample = _sample_test_ledger(ledger, int(args.sample_size))
        baseline_features = build_b3_matrix(sample, combined_source, combined_structural, maps)
        if baseline_features.columns.tolist() != feature_names:
            raise AssertionError("feature schema changed during baseline sample inference")
        baseline_prediction = _predict_feature_batches(booster, sample, baseline_features, batch_size=len(sample))
        submission = pd.read_csv(args.submission.resolve(), usecols=["ID", "Target"]) if args.submission else None
        compare: dict[str, Any] = {"sample_rows": int(len(sample)), "sample_id_unique": int(sample["sample_id"].astype(str).nunique()), "sample_size_requested": int(args.sample_size)}
        if submission is not None:
            submitted = submission.loc[:, ["ID", "Target"]].copy(); submitted["ID"] = submitted["ID"].astype(str)
            sample_ids = sample["sample_id"].astype(str).str.removeprefix("te__")
            joined = pd.DataFrame({"ID": sample_ids, "rebuilt": baseline_prediction}).merge(submitted.rename(columns={"Target": "submitted"}), on="ID", how="left", validate="one_to_one", sort=False)
            diff = joined["rebuilt"].to_numpy(dtype=float) - joined["submitted"].to_numpy(dtype=float)
            compare.update({"submission_join_rows": int(len(joined)), "submission_missing_sample_rows": int(joined["submitted"].isna().sum()), "max_abs_difference_to_submitted": None if joined["submitted"].isna().any() else float(np.max(np.abs(diff))), "within_1e-6": bool(not joined["submitted"].isna().any() and np.max(np.abs(diff)) <= 1e-6)})

        order_checks: list[dict[str, Any]] = []
        for batch_size in (1, 17, 257, min(4096, len(sample))):
            pred = _predict_feature_batches(booster, sample, baseline_features, batch_size=batch_size)
            order_checks.append({"batch_size": int(batch_size), "max_abs_difference": float(np.max(np.abs(pred - baseline_prediction)))})
        rng = np.random.default_rng(20260909)
        shuffled = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
        shuffled_prediction = _predict_batches(booster, shuffled, combined_source, combined_structural, maps, batch_size=257, feature_names=feature_names)
        by_id_original = pd.DataFrame({"sample_id": sample["sample_id"].astype(str), "prediction": baseline_prediction})
        by_id_shuffled = pd.DataFrame({"sample_id": shuffled["sample_id"].astype(str), "prediction": shuffled_prediction})
        shuffled_join = by_id_original.merge(by_id_shuffled, on="sample_id", how="left", validate="one_to_one", suffixes=("_original", "_shuffled"), sort=False)
        compare["batch_size_checks"] = order_checks
        compare["shuffled_request_max_abs_difference"] = float(np.max(np.abs(shuffled_join["prediction_original"] - shuffled_join["prediction_shuffled"])))

        # Re-running the legal visibility builder after changing hidden Test TWS
        # is the strong masked-input test: hidden values must be blanked before
        # they can influence anchors or maps.
        test_masked_perturbed = test.copy()
        hidden = test_masked_perturbed["TWS_t_masked"].astype(bool)
        test_masked_perturbed.loc[hidden, "TWS_t"] = test_masked_perturbed.loc[hidden, "TWS_t"].fillna(0.0) + 123456.0
        ledger_hidden, structural_hidden, _, _, _, _ = _prepare_competition_panels(train, test_masked_perturbed)
        hidden_sample = ledger_hidden.set_index("sample_id").loc[sample["sample_id"].astype(str)].reset_index()
        hidden_same = np.array_equal(hidden_sample["h"].to_numpy(), sample["h"].to_numpy()) and np.array_equal(hidden_sample["last_observed_date"].astype(str).to_numpy(), sample["last_observed_date"].astype(str).to_numpy()) and np.allclose(hidden_sample["last_observed_TWS"].to_numpy(dtype=float), sample["last_observed_TWS"].to_numpy(dtype=float), atol=0.0, rtol=0.0)
        # Source covariates do not contain TWS; structural_hidden is rebuilt and
        # its masked values are NaN, so maps remain exactly the original maps.
        pred_hidden = _predict_batches(booster, hidden_sample, combined_source, structural_hidden, maps, batch_size=257, feature_names=feature_names)
        compare["masked_tws_perturbation"] = {"perturbed_rows": int(hidden.sum()), "legal_ledger_unchanged": bool(hidden_same), "max_abs_prediction_difference": float(np.max(np.abs(pred_hidden - baseline_prediction)))}

        # Perturb only later covariate rows and compare a pre-cutoff subset.
        sample_period = _period_number(pd.to_datetime(sample["source_date"]))
        cutoff = int(sample_period.min()) + 6
        early_mask = sample_period <= cutoff
        covariate_perturbed = combined_source.copy()
        later = _period_number(covariate_perturbed["time"]) > cutoff
        for variable in HYDRO_COLUMNS:
            values = covariate_perturbed.loc[later, variable].to_numpy(dtype=float, na_value=np.nan)
            finite = np.isfinite(values)
            values[finite] += 98765.0
            covariate_perturbed.loc[later, variable] = values
        maps_later = build_b3_feature_maps(covariate_perturbed)
        early_sample = sample.loc[early_mask].reset_index(drop=True)
        early_original = baseline_prediction[early_mask]
        early_perturbed = _predict_batches(booster, early_sample, covariate_perturbed, combined_structural, maps_later, batch_size=257, feature_names=feature_names)
        compare["later_covariate_perturbation"] = {"cutoff_source_month": _period_label(cutoff), "perturbed_source_rows": int(later.sum()), "early_sample_rows": int(len(early_sample)), "max_abs_earlier_prediction_difference": None if not len(early_sample) else float(np.max(np.abs(early_perturbed - early_original)))}

        result = {"model_path": str(model_path), "model_sha256": _sha256(model_path), "feature_schema_sha256": _hash_values(feature_names), "test_ledger_rows": int(len(ledger)), "test_ledger_id_unique": int(ledger["sample_id"].astype(str).nunique()), "join_identity": {"test_rows": int(len(test)), "ledger_rows": int(len(ledger)), "sample_rows": int(len(sample)), "sample_ids_from_test": bool(set(sample["sample_id"].astype(str)) <= set(_namespace("te__", test["ID"]).astype(str)))}, "checks": compare, "no_prediction_file_written": True}
        _json(args.run_dir / "sample_inference_details.json", result)
        del maps, maps_later, combined_source, combined_structural, train, test, sample
        gc.collect()
        _stage_finish(path, manifest, started, details_path=str(args.run_dir / "sample_inference_details.json"), inference=result)
    except Exception as exc:
        _stage_fail(path, manifest, started, exc)
        raise


STAGES = {
    "integrity": stage_integrity,
    "target_alignment": stage_target_alignment,
    "target_behavior": stage_target_behavior,
    "oof_decomposition": stage_oof_decomposition,
    "full_fit": stage_full_fit,
    "tree_inspection": stage_tree_inspection,
    "matched_comparison": stage_matched_comparison,
    "test_support": stage_test_support,
    "sample_inference": stage_sample_inference,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one read-only B3 leaderboard root-cause diagnostic stage on Kaggle.")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=FIT_DIR_DEFAULT)
    parser.add_argument("--submission", type=Path, default=None)
    parser.add_argument("--oof-root", type=Path, default=Path("/kaggle/working/drought_runs"))
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--sample-size", type=int, default=2048)
    parser.add_argument("--matched-origin", default="2014-12")
    parser.add_argument("--d0-model-dir", type=Path, default=None)
    parser.add_argument("--d0-oof", type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.stage == "integrity" and args.submission is None:
        # Integrity remains useful without the submission, but an explicit path
        # is required for the leaderboard checksum verdict.
        print(json.dumps({"warning": "--submission omitted; submission checksum will be unavailable"}), flush=True)
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
