"""Fit the frozen dense-history B3/98 model and create one Test submission.

This is the single authorized competition-prediction entrypoint.  It is intended
to run on Kaggle, never on the Windows workstation.  The fit uses every supplied
Train label with the frozen sampled-horizon recipe; inference uses a namespaced
Train+Test covariate panel and a causal visibility ledger so masked Test TWS can
never become an anchor.  No pseudo-labels or Test targets are read.
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
from src.observation_simulator import ScenarioSpec, build_template_replay_fold, simulate_observations
from src.regional_context import HYDRO_COLUMNS
from src.validation import build_test_mask_template


SEED = 20260908
ROUNDS = 98
NUM_LEAVES = 63
MIN_DATA_IN_LEAF = 1000
EXPECTED_TEST_ROWS = 280_961
REPLAY_ORIGIN = "2007-09"
DEFAULT_REPLAY_MODEL = Path(
    "/kaggle/working/drought_runs/"
    "covariate_gap_outer_20260909T000000Z_D0_2007_09/model.txt"
)
TRAIN_COLUMNS = [
    "sample_id", "time", "lat", "lon", "TWS_t", "month_sin", "month_cos",
    *HYDRO_COLUMNS, "target",
]
TEST_COLUMNS = [
    "ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked",
    "month_sin", "month_cos", *HYDRO_COLUMNS,
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
    if isinstance(values, pd.Series):
        text = values.astype(str).str.cat(sep="\n")
    else:
        text = "\n".join(str(value) for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _memory() -> dict[str, float | None]:
    try:
        import psutil

        process = psutil.Process()
        vm = psutil.virtual_memory()
        return {
            "rss_gib": process.memory_info().rss / 1024**3,
            "available_gib": vm.available / 1024**3,
        }
    except Exception:
        return {"rss_gib": None, "available_gib": None}


def _namespace(prefix: str, values: pd.Series) -> pd.Series:
    return prefix + values.astype(str)


def _assert_float32_or_missing(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        raise AssertionError(f"{name} is empty")
    if any(dtype != np.dtype("float32") for dtype in frame.dtypes):
        bad = {column: str(dtype) for column, dtype in frame.dtypes.items() if dtype != np.dtype("float32")}
        raise AssertionError(f"{name} has non-float32 columns: {bad}")
    if np.isinf(frame.to_numpy(dtype=np.float32)).any():
        raise AssertionError(f"{name} contains infinite values")


def _attach_delta(rows: pd.DataFrame, labels: pd.DataFrame) -> np.ndarray:
    joined = rows.loc[:, ["sample_id", "last_observed_TWS"]].merge(
        labels.loc[:, ["sample_id", "target"]],
        how="left", on="sample_id", validate="one_to_one", sort=False,
    )
    if joined["target"].isna().any():
        raise AssertionError("A sampled training row has no supplied label")
    return (
        joined["target"].to_numpy(dtype=np.float32)
        - joined["last_observed_TWS"].to_numpy(dtype=np.float32)
    )


def _data_preflight(data_dir: Path) -> dict[str, Any]:
    paths = {name: data_dir / name for name in ("Train.csv", "Test.csv", "SampleSubmission.csv")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    headers = {name: pd.read_csv(path, nrows=2).columns.tolist() for name, path in paths.items()}
    if headers["SampleSubmission.csv"] != ["ID", "Target"]:
        raise AssertionError(f"SampleSubmission columns are not exactly ID,Target: {headers['SampleSubmission.csv']}")
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "headers": headers,
        "sha256": {name: _sha256(path) for name, path in paths.items()},
        "sizes": {name: int(path.stat().st_size) for name, path in paths.items()},
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory": _memory(),
        "packages": {
            name: getattr(__import__(name), "__version__", "unknown")
            for name in ("numpy", "pandas", "lightgbm")
        },
    }


def _build_internal_train(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    structural = train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
    source = train.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
    labels = train.loc[:, ["sample_id", "target"]].copy()
    structural["sample_id"] = _namespace("tr__", structural["sample_id"])
    source["sample_id"] = _namespace("tr__", source["sample_id"])
    labels["sample_id"] = _namespace("tr__", labels["sample_id"])
    return structural, source, labels


def _validate_replay_path(
    *,
    train_plain: pd.DataFrame,
    test_plain: pd.DataFrame,
    train_structural: pd.DataFrame,
    train_source: pd.DataFrame,
    train_maps,
    replay_model_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Rebuild one saved D0 replay and compare predictions without refitting."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM is required on Kaggle") from exc
    if not replay_model_path.is_file():
        raise FileNotFoundError(f"Saved D0 replay model is missing: {replay_model_path}")
    replay_dir = replay_model_path.parent
    saved_oof_path = replay_dir / "oof.csv.gz"
    saved_manifest_path = replay_dir / "manifest.json"
    for path in (saved_oof_path, saved_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    template = build_test_mask_template(
        test_plain.loc[:, ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]]
    )
    fold = build_template_replay_fold(train_plain, template, start_month=REPLAY_ORIGIN,
                                      scenario_id="submission_integration_replay_2007_09",
                                      family="submission_integration")
    ledger = fold.ledger.copy()
    ledger["sample_id"] = _namespace("tr__", ledger["sample_id"])
    valid_x = build_b3_matrix(ledger, train_source, train_structural, train_maps)
    _assert_float32_or_missing(valid_x, "integration validation features")

    saved_manifest = json.loads(saved_manifest_path.read_text(encoding="utf-8"))
    expected_names = saved_manifest.get("feature_names")
    if expected_names is not None and expected_names != valid_x.columns.tolist():
        raise AssertionError("Integration replay feature schema differs from saved D0 schema")
    booster = lgb.Booster(model_file=str(replay_model_path))
    prediction = ledger["last_observed_TWS"].to_numpy(dtype=np.float64) + booster.predict(valid_x)
    saved = pd.read_csv(saved_oof_path, usecols=["sample_id", "prediction"])
    if saved["sample_id"].astype(str).duplicated().any():
        raise AssertionError("Saved D0 OOF has duplicate IDs")
    rebuilt = pd.DataFrame({"sample_id": ledger["sample_id"].str.removeprefix("tr__"), "prediction": prediction})
    joined = saved.assign(sample_id=saved["sample_id"].astype(str)).merge(
        rebuilt, how="left", on="sample_id", suffixes=("_saved", "_rebuilt"),
        validate="one_to_one", sort=False,
    )
    if len(joined) != len(saved) or joined["prediction_rebuilt"].isna().any():
        raise AssertionError("Integration replay ID coverage differs from saved D0 OOF")
    max_abs = float(np.max(np.abs(
        joined["prediction_saved"].to_numpy(dtype=np.float64)
        - joined["prediction_rebuilt"].to_numpy(dtype=np.float64)
    )))
    tolerance = 1e-6
    result = {
        "status": "passed" if max_abs <= tolerance else "failed",
        "origin": REPLAY_ORIGIN,
        "saved_model": str(replay_model_path),
        "saved_model_sha256": _sha256(replay_model_path),
        "saved_oof": str(saved_oof_path),
        "saved_oof_sha256": _sha256(saved_oof_path),
        "rows": int(len(joined)),
        "feature_count": int(valid_x.shape[1]),
        "max_abs_prediction_difference": max_abs,
        "tolerance": tolerance,
        "passed": bool(max_abs <= tolerance),
        "refit": False,
    }
    _json(output_dir / "integration_validation.json", result)
    if not result["passed"]:
        raise AssertionError(result)
    return result


def _build_test_visibility(
    *,
    train_structural: pd.DataFrame,
    test: pd.DataFrame,
    max_training_target_month: pd.Period,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create namespaced legal Train+Test structural/source panels and Test ledger."""
    train_panel = train_structural.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
    train_panel["tws_visible"] = True
    test_structural = test.loc[:, ["ID", "time", "lat", "lon", "TWS_t"]].rename(columns={"ID": "sample_id"}).copy()
    test_structural["sample_id"] = _namespace("te__", test_structural["sample_id"])
    hidden = test["TWS_t_masked"].astype(bool).to_numpy()
    if ((~hidden) & test_structural["TWS_t"].isna().to_numpy()).any():
        raise AssertionError("Visible Test TWS contains a missing value")
    test_structural.loc[hidden, "TWS_t"] = np.nan
    test_structural["tws_visible"] = ~hidden
    panel = pd.concat([train_panel, test_structural], ignore_index=True, sort=False)
    if panel["sample_id"].duplicated().any():
        raise AssertionError("Namespaced Train+Test panel has duplicate IDs")

    scenario = ScenarioSpec(
        scenario_id="competition_test_full",
        family="competition_test",
        first_source_month=str(pd.to_datetime(test["time"]).dt.to_period("M").min()),
        last_source_month=str(pd.to_datetime(test["time"]).dt.to_period("M").max()),
        training_target_cutoff=str(max_training_target_month),
        visibility_rule="all supplied Train TWS visible; Test TWS visible only when TWS_t_masked is false",
        notes="Masked Test TWS values are blanked before simulation and never enter the legal anchor state.",
    )
    score_ids = test_structural["sample_id"]
    fold = simulate_observations(panel, score_ids=score_ids, labels=None, scenario=scenario)
    ledger = fold.ledger.copy()
    expected = set(score_ids.astype(str))
    actual = set(ledger["sample_id"].astype(str))
    if actual != expected or len(ledger) != len(test):
        raise AssertionError("Test visibility ledger does not cover exactly the Test IDs")
    masked_ids = set(score_ids.loc[hidden].astype(str))
    masked_rows = ledger["sample_id"].astype(str).isin(masked_ids)
    if (ledger.loc[masked_rows, "source_date"] == ledger.loc[masked_rows, "last_observed_date"]).any():
        raise AssertionError("A masked Test TWS value became its own anchor")
    visible_rows = ledger["sample_id"].astype(str).isin(set(score_ids.loc[~hidden].astype(str)))
    if (ledger.loc[visible_rows, "h"] != 1).any():
        raise AssertionError("A visible current Test TWS row does not have h=1")
    if ledger["last_observed_date"].isna().any() or (ledger["h"] < 1).any():
        raise AssertionError("Test visibility ledger has an invalid legal anchor/horizon")

    train_source_for_combined = train_structural  # caller supplies the matching source separately
    del train_source_for_combined
    return ledger, test_structural, panel, {
        "scenario": scenario.__dict__,
        "rows": int(len(ledger)),
        "masked_rows": int(hidden.sum()),
        "visible_rows": int((~hidden).sum()),
        "horizon_counts": {str(int(k)): int(v) for k, v in ledger["h"].value_counts().sort_index().items()},
        "sample_id_hash": _hash_text(ledger["sample_id"]),
        "masked_same_month_anchor_rows": 0,
        "exclusions": fold.exclusions,
    }


def _prediction_diagnostics(ledger: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    persistence = ledger["last_observed_TWS"].to_numpy(dtype=np.float64)
    h = ledger["h"].to_numpy(dtype=np.int64)
    difference = prediction - persistence
    by_h: list[dict[str, Any]] = []
    for horizon in sorted(np.unique(h)):
        mask = h == horizon
        values = prediction[mask]
        deltas = difference[mask]
        by_h.append({
            "h": int(horizon),
            "rows": int(mask.sum()),
            "prediction_mean": float(np.mean(values)),
            "prediction_std": float(np.std(values)),
            "prediction_min": float(np.min(values)),
            "prediction_max": float(np.max(values)),
            "delta_vs_persistence_mean": float(np.mean(deltas)),
            "delta_vs_persistence_std": float(np.std(deltas)),
            "delta_vs_persistence_abs_p95": float(np.quantile(np.abs(deltas), 0.95)),
            "delta_vs_persistence_abs_max": float(np.max(np.abs(deltas))),
        })
    return {
        "rows": int(len(prediction)),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction)),
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
        "persistence_mean": float(np.mean(persistence)),
        "persistence_min": float(np.min(persistence)),
        "persistence_max": float(np.max(persistence)),
        "delta_vs_persistence_mean": float(np.mean(difference)),
        "delta_vs_persistence_std": float(np.std(difference)),
        "delta_vs_persistence_abs_p95": float(np.quantile(np.abs(difference), 0.95)),
        "delta_vs_persistence_abs_max": float(np.max(np.abs(difference))),
        "large_abs_delta_gt_5_rows": int(np.sum(np.abs(difference) > 5.0)),
        "by_horizon": by_h,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit frozen dense-history B3/98 and generate one Test submission on Kaggle.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--replay-model", type=Path, default=DEFAULT_REPLAY_MODEL)
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args()
    if args.seed != SEED or args.rounds != ROUNDS:
        raise ValueError(f"Frozen B3/98 recipe requires seed={SEED} and rounds={ROUNDS}")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    import lightgbm as lgb

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit:
        raise RuntimeError({"expected_commit": args.expected_commit, "actual_commit": head})
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Refusing to run from a dirty repository checkout")

    manifest: dict[str, Any] = {
        "status": "running",
        "stage": "FULL_B3_DENSE_SUBMISSION",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit": head,
        "branch": branch,
        "seed": SEED,
        "rounds": ROUNDS,
        "num_leaves": NUM_LEAVES,
        "min_data_in_leaf": MIN_DATA_IN_LEAF,
        "batch_size": args.batch_size,
        "test_predictions_authorized": True,
        "no_test_labels_used": True,
        "fit_count": 0,
        "phases": [],
    }
    _json(output_dir / "manifest.json", manifest)

    def phase(name: str, **details: Any) -> None:
        payload = {"name": name, "elapsed_seconds": time.perf_counter() - started, "memory": _memory(), **details}
        manifest["phases"].append(payload)
        _json(output_dir / "manifest.json", manifest)
        print(json.dumps(payload, sort_keys=True), flush=True)

    try:
        preflight = _data_preflight(data_dir)
        _json(output_dir / "preflight.json", preflight)
        phase("preflight", train_csv=preflight["sha256"]["Train.csv"], test_csv=preflight["sha256"]["Test.csv"])

        train = pd.read_csv(data_dir / "Train.csv", usecols=TRAIN_COLUMNS)
        test = pd.read_csv(data_dir / "Test.csv", usecols=TEST_COLUMNS)
        sample = pd.read_csv(data_dir / "SampleSubmission.csv")
        if len(test) != EXPECTED_TEST_ROWS:
            raise AssertionError(f"Expected {EXPECTED_TEST_ROWS} Test rows, found {len(test)}")
        if len(sample) != len(test):
            raise AssertionError("SampleSubmission row count differs from Test")
        if train["sample_id"].duplicated().any() or test["ID"].duplicated().any() or sample["ID"].duplicated().any():
            raise AssertionError("Train/Test/SampleSubmission IDs must be unique")
        if set(test["ID"].astype(str)) != set(sample["ID"].astype(str)):
            raise AssertionError("Test and SampleSubmission ID sets differ")
        if train[["lat", "lon", "time"]].duplicated().any() or test[["lat", "lon", "time"]].duplicated().any():
            raise AssertionError("Train or Test contains duplicate location-month rows")
        if train["target"].isna().any():
            raise AssertionError("Full-training recipe requires every Train target to be supplied")
        if test["TWS_t_masked"].isna().any():
            raise AssertionError("Test TWS_t_masked contains missing flags")
        phase("data_loaded", train_rows=int(len(train)), test_rows=int(len(test)), locations=int(train[["lat", "lon"]].drop_duplicates().shape[0]))

        train_structural, train_source, train_labels = _build_internal_train(train)
        train_plain_structural = train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
        train_plain_source = train.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
        train_plain_labels = train.loc[:, ["sample_id", "target"]].copy()
        train_periods = pd.to_datetime(train["time"]).dt.to_period("M")
        max_training_target_month = (train_periods + 1).max()

        # Sampling is done before internal namespacing so the frozen sample-ID
        # hash and horizon assignment remain identical to the validated recipe.
        sampled_plain = build_sampled_training_rows(
            train_plain_structural,
            train_plain_source.loc[:, SOURCE_CORE_COLUMNS],
            max_target_month=max_training_target_month,
            seed=SEED,
        )
        rows_plain = sampled_plain.rows
        y_train = _attach_delta(rows_plain, train_plain_labels)
        train_weights = horizon_rebalance_weights(rows_plain["h"])
        rows = rows_plain.copy()
        rows["sample_id"] = _namespace("tr__", rows["sample_id"])
        training_row_hash = _hash_text(rows_plain["sample_id"])
        training_label_hash = _hash_array(y_train)
        training_weight_hash = _hash_array(np.asarray(train_weights, dtype=np.float32))
        training_info = {
            "rows": int(len(rows)),
            "horizon_counts": {str(int(k)): int(v) for k, v in rows_plain["h"].value_counts().sort_index().items()},
            "dropped_missing_anchor": int(sampled_plain.dropped_missing_anchor),
            "max_training_target_month": str(max_training_target_month),
            "sample_seed": SEED,
            "row_id_hash": training_row_hash,
            "label_hash": training_label_hash,
            "weight_hash": training_weight_hash,
        }
        phase("training_rows_ready", **training_info)

        # One Train-only map serves both the integration replay and the fit.  It
        # is released after fitting; inference gets a fresh combined map that
        # includes legally supplied Test covariates at Test source months.
        train_maps = build_b3_feature_maps(train_source)
        phase("train_feature_maps_ready")
        integration = _validate_replay_path(
            train_plain=train,
            test_plain=test,
            train_structural=train_structural,
            train_source=train_source,
            train_maps=train_maps,
            replay_model_path=args.replay_model,
            output_dir=output_dir,
        )
        phase("integration_replay_validated", max_abs_prediction_difference=integration["max_abs_prediction_difference"], rows=integration["rows"])

        x_train = build_b3_matrix(rows, train_source, train_structural, train_maps)
        _assert_float32_or_missing(x_train, "full-training B3 features")
        feature_names = x_train.columns.tolist()
        if len(feature_names) != 446:
            raise AssertionError(f"Frozen B3 schema must contain 446 features, found {len(feature_names)}")
        feature_schema = {
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "dtype": "float32",
            "missing_values_allowed": True,
            "schema_sha256": _hash_text(feature_names),
        }
        _json(output_dir / "feature_schema.json", feature_schema)
        params = dict(DEFAULT_PARAMS)
        params.update({
            "num_leaves": NUM_LEAVES,
            "min_data_in_leaf": MIN_DATA_IN_LEAF,
            "num_threads": min(4, os.cpu_count() or 1),
            "seed": SEED,
            "feature_fraction_seed": SEED,
            "bagging_seed": SEED,
            "data_random_seed": SEED,
        })
        resolved_config = {
            "model": "B3_dense_history_delta_lightgbm",
            "seed": SEED,
            "rounds": ROUNDS,
            "params": params,
            "feature_schema": feature_schema,
            "training": training_info,
            "data_preflight": preflight,
            "integration_validation": integration,
            "test_rows_expected": EXPECTED_TEST_ROWS,
            "test_ids_hash": _hash_text(test["ID"]),
            "test_predictions_use": "namespaced Train+Test source covariates plus causal TWS visibility ledger",
            "submission_columns": ["ID", "Target"],
        }
        _json(output_dir / "resolved_config.json", resolved_config)
        phase("resolved_config_written", feature_count=len(feature_names))

        fit_started = time.perf_counter()
        train_set = lgb.Dataset(
            x_train,
            label=y_train,
            weight=np.asarray(train_weights, dtype=np.float32),
            feature_name=feature_names,
            free_raw_data=True,
        )
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=ROUNDS,
            callbacks=[lgb.log_evaluation(25)],
        )
        manifest["fit_count"] = 1
        model_path = output_dir / "model.txt"
        booster.save_model(str(model_path))
        phase("full_fit_complete", seconds=time.perf_counter() - fit_started, model_bytes=int(model_path.stat().st_size))
        del train_set, x_train, y_train, train_weights, train_maps
        gc.collect()

        # Build the Test visibility stream after fitting.  Hidden Test TWS is
        # blanked in both the simulator panel and structural feature panel.
        ledger, test_structural, combined_panel, visibility_info = _build_test_visibility(
            train_structural=train_structural,
            test=test,
            max_training_target_month=max_training_target_month,
        )
        test_source = test.loc[:, ["ID", "time", "lat", "lon", "month_sin", "month_cos", *HYDRO_COLUMNS]].rename(columns={"ID": "sample_id"}).copy()
        test_source["sample_id"] = _namespace("te__", test_source["sample_id"])
        combined_source = pd.concat([train_source, test_source], ignore_index=True, sort=False)
        combined_structural = pd.concat([
            train_structural.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]],
            test_structural.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t"]],
        ], ignore_index=True, sort=False)
        if combined_source["sample_id"].duplicated().any() or combined_structural["sample_id"].duplicated().any():
            raise AssertionError("Combined source/structural IDs are not unique")
        if combined_source[["lat", "lon", "time"]].duplicated().any():
            raise AssertionError("Combined source has duplicate location-month rows")
        if combined_structural[["lat", "lon", "time"]].duplicated().any():
            raise AssertionError("Combined structural panel has duplicate location-month rows")
        test_ledger_for_disk = ledger.loc[:, ["sample_id", "source_date", "target_date", "lat", "lon", "tws_visible", "last_observed_date", "last_observed_TWS", "h"]].copy()
        reverse_ids = dict(zip(_namespace("te__", test["ID"]), test["ID"].astype(str), strict=True))
        test_ledger_for_disk.insert(0, "ID", test_ledger_for_disk.pop("sample_id").map(reverse_ids))
        test_ledger_for_disk.to_csv(output_dir / "test_visibility_ledger.csv.gz", index=False, compression="gzip")
        visibility_info["ledger_sha256"] = _sha256(output_dir / "test_visibility_ledger.csv.gz")
        _json(output_dir / "test_visibility_summary.json", visibility_info)
        phase("test_visibility_ready", **visibility_info)

        combined_maps = build_b3_feature_maps(combined_source)
        phase("combined_test_feature_maps_ready", source_rows=int(len(combined_source)))
        reloaded = lgb.Booster(model_file=str(model_path))
        prediction = np.empty(len(ledger), dtype=np.float64)
        first_reload_difference: float | None = None
        for start in range(0, len(ledger), args.batch_size):
            stop = min(start + args.batch_size, len(ledger))
            chunk = ledger.iloc[start:stop].reset_index(drop=True)
            x_chunk = build_b3_matrix(chunk, combined_source, combined_structural, combined_maps)
            _assert_float32_or_missing(x_chunk, f"Test B3 batch {start}:{stop}")
            if x_chunk.columns.tolist() != feature_names:
                raise AssertionError("Train/Test feature order differs")
            original_pred = booster.predict(x_chunk)
            reloaded_pred = reloaded.predict(x_chunk)
            if first_reload_difference is None:
                first_reload_difference = float(np.max(np.abs(original_pred - reloaded_pred)))
            prediction[start:stop] = chunk["last_observed_TWS"].to_numpy(dtype=np.float64) + reloaded_pred
            print(json.dumps({"phase": "test_batch", "start": start, "stop": stop, "rows": stop - start, "memory": _memory()}, sort_keys=True), flush=True)
        if first_reload_difference is None or first_reload_difference > 1e-6:
            raise AssertionError(f"Saved/reloaded model mismatch: {first_reload_difference}")
        if not np.isfinite(prediction).all():
            raise AssertionError("Test predictions contain NaN or infinity")
        diagnostics = _prediction_diagnostics(ledger, prediction)
        diagnostics.update({
            "saved_reloaded_model_max_abs_difference_first_batch": first_reload_difference,
            "saved_reloaded_model_tolerance": 1e-6,
            "no_clipping_or_calibration": True,
        })
        _json(output_dir / "prediction_diagnostics.json", diagnostics)
        phase("test_inference_complete", **{key: diagnostics[key] for key in ("rows", "prediction_min", "prediction_max", "delta_vs_persistence_abs_max", "large_abs_delta_gt_5_rows")})

        pred_by_id = pd.DataFrame({
            "ID": ledger["sample_id"].map(reverse_ids),
            "Target": prediction,
        })
        if pred_by_id["ID"].isna().any() or pred_by_id["ID"].duplicated().any():
            raise AssertionError("Internal Test prediction IDs failed to restore uniquely")
        submission = sample.loc[:, ["ID"]].copy()
        submission["_id_key"] = submission["ID"].astype(str)
        pred_by_id["_id_key"] = pred_by_id["ID"].astype(str)
        submission = submission.merge(pred_by_id.loc[:, ["_id_key", "Target"]], on="_id_key", how="left", validate="one_to_one", sort=False)
        if submission["Target"].isna().any():
            raise AssertionError("Submission has missing predictions")
        submission = submission.loc[:, ["ID", "Target"]]
        if submission.columns.tolist() != ["ID", "Target"]:
            raise AssertionError("Submission columns are not exactly ID,Target")
        if len(submission) != EXPECTED_TEST_ROWS or submission["ID"].duplicated().any() or not np.isfinite(submission["Target"]).all():
            raise AssertionError("Submission row/finite/duplicate checks failed")
        if submission["ID"].astype(str).tolist() != sample["ID"].astype(str).tolist():
            raise AssertionError("Submission IDs are not in SampleSubmission order")
        submission_path = output_dir / "submission_b3_dense_history_98_5bc9e52.csv"
        submission.to_csv(submission_path, index=False)
        output_info = {
            "path": str(submission_path),
            "rows": int(len(submission)),
            "columns": submission.columns.tolist(),
            "sha256": _sha256(submission_path),
            "bytes": int(submission_path.stat().st_size),
            "id_hash": _hash_text(submission["ID"]),
            "finite_predictions": int(np.isfinite(submission["Target"]).sum()),
            "duplicate_ids": int(submission["ID"].duplicated().sum()),
        }
        _json(output_dir / "submission_manifest.json", {
            "recipe": "dense-history B3/98",
            "source_commit": head,
            "seed": SEED,
            "rounds": ROUNDS,
            "feature_schema_sha256": feature_schema["schema_sha256"],
            "output": output_info,
            "no_test_labels_used": True,
            "no_pseudo_labels": True,
            "exactly_one_submission_file": True,
        })
        manifest.update({
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "model": {"path": str(model_path), "sha256": _sha256(model_path), "bytes": int(model_path.stat().st_size)},
            "submission": output_info,
            "submission_manifest": str(output_dir / "submission_manifest.json"),
            "prediction_diagnostics": str(output_dir / "prediction_diagnostics.json"),
            "test_visibility_summary": visibility_info,
            "output_checksums": {
                str(path.relative_to(output_dir)): _sha256(path)
                for path in output_dir.rglob("*") if path.is_file() and path.name != "manifest.json"
            },
        })
        _json(output_dir / "manifest.json", manifest)
        phase("submission_ready", submission_path=str(submission_path), submission_sha256=output_info["sha256"], rows=output_info["rows"])
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "memory": _memory(),
        })
        _json(output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
