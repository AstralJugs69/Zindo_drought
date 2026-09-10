"""Run the fixed B3-C versus GLDAS-augmented replay gates.

The four saved B3-C controls are loaded only after exact supervision, matrix,
schema, and model-tree fingerprints match.  The GLDAS candidate uses the same
rows, labels, anchors, horizons, weights, folds, and 444-column base matrices.
No Test labels, Test predictions, or submission artifacts are touched.
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_b3_neighbor_state_experiment import (  # noqa: E402
    ResourceTracker,
    _aggregate,
    _attach_training_labels,
    _run_candidate,
    _scope_metrics,
    _version,
)
from scripts.run_submission_compliant_b3_experiment import (  # noqa: E402
    GLDAS_METADATA_URL,
    GLDAS_README_URL,
    GLDAS_SAMPLE_URL,
    HISTORICAL_EXTERNAL_ACCESS_FAILURE,
    _external_access_record,
    _fold_for_origin,
    _prefit_contract_checks as _b3c_prefit_contract_checks,
)
from scripts.run_lgbm_core import DEFAULT_NUM_THREADS, DEFAULT_PARAMS, MAX_NUM_THREADS  # noqa: E402
from src.availability import build_replay_observation_view  # noqa: E402
from src.compliant_b3 import (  # noqa: E402
    COORDINATE_FREE_B3_FEATURE_COLUMNS,
    ORIGINAL_B3_FEATURE_COLUMNS,
    assert_coordinate_free_b3_schema,
    build_coordinate_free_b3_matrix,
    coordinate_free_b3_feature_provenance,
)
from src.gldas_external import (  # noqa: E402
    GLDAS_EXTERNAL_FEATURE_COLUMNS,
    GLDAS_PRODUCT,
    GLDAS_PROHIBITED_MODEL_IDENTIFIERS,
    GLDASRawStore,
    build_external_request_plan,
    build_gldas_external_features,
    external_coverage_report,
    run_gldas_external_prefit_checks,
)
from src.ml_features import (  # noqa: E402
    SOURCE_CORE_COLUMNS,
    SOURCE_HYDRO_HISTORY_COLUMNS,
    build_sampled_training_rows,
    horizon_rebalance_weights,
)
from src.regional_context import HYDRO_COLUMNS  # noqa: E402
from src.validation import build_test_mask_template  # noqa: E402


SEED = 20260908
ROUNDS = 98
NUM_LEAVES = 63
MIN_DATA_IN_LEAF = 1000
INNER_ORIGINS = ("2003-04", "2004-04")
RECENT_ORIGINS = ("2014-04", "2014-12")
OLDER_ORIGIN = "2009-01"
ALL_ORIGINS = (*INNER_ORIGINS, *RECENT_ORIGINS, OLDER_ORIGIN)
BASE_CANDIDATE = "B3C"
EXTERNAL_CANDIDATE = "B3C_GLDAS"
TEST_GEOMETRY_COLUMNS = ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]
TRAIN_COLUMNS = [
    "sample_id", "time", "lat", "lon", "TWS_t", "month_sin", "month_cos",
    *HYDRO_COLUMNS, "target",
]
EXPECTED_CONTROL_COMMIT = "958060ffca757c50c936d678bdaaca2533bae780"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def _assert_no_secret_like_fields(value: object) -> None:
    allowed = {"credential_values_recorded", "cookies_recorded"}
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered not in allowed and any(
                token in lowered for token in ("password", "passwd", "authorization", "cookie", "token")
            ):
                raise ValueError("artifact contains prohibited secret-like fields")
            _assert_no_secret_like_fields(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_secret_like_fields(child)


def _safe_external_access(sample_verification: dict[str, object]) -> dict[str, object]:
    """Create the current measured record from sanitized retained verification."""
    measured = {
        "status": "success_netcdf_verified",
        "http_status": 200,
        "final_host": "hydro1.gesdisc.eosdis.nasa.gov",
        "content_type": "application/x-netcdf",
        "netrc_present": True,
        "netrc_permissions_ok": True,
        "netrc_machine_present": True,
        "sample_file_retained": True,
        "netcdf_verification": sample_verification,
        "credentials": {"netrc_present": True, "netrc_permissions_ok": True, "netrc_machine_present": True},
    }
    return _external_access_record(measured)


def _load_store(store_dir: Path) -> tuple[GLDASRawStore, dict[str, object], dict[str, object]]:
    manifest_path = store_dir / "manifest.json"
    metadata_path = store_dir / "store_metadata.json"
    plan_path = store_dir / "request_plan.json"
    for path in (manifest_path, metadata_path, plan_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _assert_no_secret_like_fields(manifest)
    _assert_no_secret_like_fields(metadata)
    if manifest.get("status") != "completed" or manifest.get("product") != GLDAS_PRODUCT:
        raise ValueError("GLDAS acquisition manifest is not a completed selected-product run")
    raw_path = store_dir / str(manifest.get("raw_store", "raw_store.npz"))
    store = GLDASRawStore.load(raw_path, metadata_path)
    if int(manifest.get("raw_store_rows", -1)) != store.row_count:
        raise ValueError("GLDAS store row-count fingerprint mismatch")
    if manifest.get("raw_store_key_sha256") != store.store_key_sha256:
        raise ValueError("GLDAS store key fingerprint mismatch")
    if metadata.get("coordinate_sha256") != plan.get("coordinate_sha256"):
        raise ValueError("GLDAS store/request coordinate fingerprint mismatch")
    return store, manifest, plan


def _supervision(rows: pd.DataFrame, y_target: np.ndarray, y_delta: np.ndarray, weights: np.ndarray, fold) -> dict[str, str]:
    return {
        "training_ids_hash": _hash_values(rows["sample_id"]),
        "training_target_hash": _hash_values(y_target),
        "training_delta_hash": _hash_values(y_delta),
        "training_anchor_hash": _hash_values(rows["last_observed_TWS"].to_numpy(dtype=np.float32)),
        "training_h_hash": _hash_values(rows["h"].to_numpy(dtype=np.int16)),
        "training_weights_hash": _hash_values(weights),
        "validation_ids_hash": _hash_values(fold.ledger["sample_id"]),
        "validation_target_hash": _hash_values(fold.labels["target"].to_numpy(dtype=np.float32)),
        "validation_anchor_hash": _hash_values(fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float32)),
        "validation_h_hash": _hash_values(fold.ledger["h"].to_numpy(dtype=np.int16)),
    }


def _reuse_b3c_control(
    *,
    control_dir: Path,
    origin: str,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    supervision: dict[str, str],
):
    """Load a saved B3-C booster only when every persisted fingerprint agrees."""
    import lightgbm as lgb

    info: dict[str, object] = {"status": "checking", "control_dir": str(control_dir)}
    manifest_path = control_dir / "manifest.json"
    details_path = control_dir / f"origin_{origin}_details.json"
    model_path = control_dir / f"model_{origin}_B3C.txt"
    for path in (manifest_path, details_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"saved B3-C control artifact missing: {path}")
    control_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    details = json.loads(details_path.read_text(encoding="utf-8"))
    if control_manifest.get("status") != "completed":
        raise ValueError(f"saved B3-C control manifest is not completed for {origin}")
    control_commit = control_manifest.get("commit")
    if control_commit != EXPECTED_CONTROL_COMMIT:
        raise ValueError({"origin": origin, "saved_control_commit": control_commit, "expected": EXPECTED_CONTROL_COMMIT})
    training = details.get("training", {})
    validation = details.get("validation", {})
    contract = details.get("feature_contract", {})
    expected_hashes = dict(supervision)
    actual_hashes = {}
    for key in expected_hashes:
        actual_hashes[key] = training.get(key, validation.get(key))
    mismatches = {
        key: {"expected": value, "actual": actual_hashes.get(key)}
        for key, value in expected_hashes.items()
        if actual_hashes.get(key) != value
    }
    if contract.get("b3c_count") != len(COORDINATE_FREE_B3_FEATURE_COLUMNS):
        mismatches["b3c_count"] = {"expected": len(COORDINATE_FREE_B3_FEATURE_COLUMNS), "actual": contract.get("b3c_count")}
    if contract.get("ordered_allowlist") != list(COORDINATE_FREE_B3_FEATURE_COLUMNS):
        mismatches["ordered_allowlist"] = {"expected": list(COORDINATE_FREE_B3_FEATURE_COLUMNS), "actual": contract.get("ordered_allowlist")}
    matrix_expected = {
        "training_matrix_sha256": _hash_frame(x_train),
        "validation_matrix_sha256": _hash_frame(x_valid),
    }
    for key, value in matrix_expected.items():
        if contract.get(key) != value:
            mismatches[key] = {"expected": value, "actual": contract.get(key)}
    if mismatches:
        raise ValueError({"origin": origin, "saved_control_fingerprint_mismatch": mismatches})
    booster = lgb.Booster(model_file=str(model_path))
    feature_names = list(booster.feature_name())
    if feature_names != x_train.columns.tolist() or feature_names != x_valid.columns.tolist():
        booster.free_dataset()
        raise ValueError(f"saved B3-C feature schema mismatch for {origin}")
    if int(booster.num_trees()) != ROUNDS:
        booster.free_dataset()
        raise ValueError(f"saved B3-C tree count mismatch for {origin}")
    info.update({
        "status": "reused",
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "num_trees": int(booster.num_trees()),
        "feature_count": len(feature_names),
        "source_commit": control_commit,
        "fingerprints": {"supervision": expected_hashes, **matrix_expected},
    })
    return booster, info


def _assert_paired_features(base: pd.DataFrame, candidate: pd.DataFrame) -> None:
    if base.columns.tolist() != list(COORDINATE_FREE_B3_FEATURE_COLUMNS):
        raise AssertionError("B3-C base columns differ from the frozen 444-column allowlist")
    expected = list(COORDINATE_FREE_B3_FEATURE_COLUMNS) + list(GLDAS_EXTERNAL_FEATURE_COLUMNS)
    if candidate.columns.tolist() != expected:
        raise AssertionError("GLDAS candidate schema is not exactly B3-C plus ten allowlisted columns")
    prohibited = GLDAS_PROHIBITED_MODEL_IDENTIFIERS.intersection(column.lower() for column in candidate.columns)
    if prohibited:
        raise AssertionError(f"prohibited identifier reached GLDAS model schema: {sorted(prohibited)}")
    if not np.array_equal(
        base.to_numpy(dtype=np.float32), candidate.iloc[:, : len(base.columns)].to_numpy(dtype=np.float32), equal_nan=True
    ):
        raise AssertionError("GLDAS candidate changed a B3-C base feature value")


def _prepare_origin(
    *,
    train: pd.DataFrame,
    labels: pd.DataFrame,
    origin: str,
    template: pd.DataFrame,
    store: GLDASRawStore,
) -> dict[str, object]:
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
    supervision = _supervision(rows, y_target, y_delta, weights, fold)
    maps = __import__("src.neural_sequence", fromlist=["build_b3_feature_maps"]).build_b3_feature_maps(view.source)
    x_train_base = build_coordinate_free_b3_matrix(rows, view.source, view.structural, maps)
    x_valid_base = build_coordinate_free_b3_matrix(fold.ledger, view.source, view.structural, maps)
    assert_coordinate_free_b3_schema(x_train_base.columns)
    assert_coordinate_free_b3_schema(x_valid_base.columns)
    if x_train_base.columns.tolist() != x_valid_base.columns.tolist():
        raise AssertionError("B3-C train/validation schema differs")
    external_train = build_gldas_external_features(rows, store)
    external_valid = build_gldas_external_features(fold.ledger, store)
    x_train_external = pd.concat([x_train_base.reset_index(drop=True), external_train], axis=1)
    x_valid_external = pd.concat([x_valid_base.reset_index(drop=True), external_valid], axis=1)
    _assert_paired_features(x_train_base, x_train_external)
    _assert_paired_features(x_valid_base, x_valid_external)
    if not external_train.notna().to_numpy().any():
        raise ValueError(f"GLDAS features have no finite training values for {origin}")
    return {
        "origin": origin,
        "role": role,
        "fold": fold,
        "view": view,
        "sampled": sampled,
        "rows": rows,
        "y_delta": y_delta,
        "y_target": y_target,
        "weights": weights,
        "supervision": supervision,
        "x_train_base": x_train_base,
        "x_valid_base": x_valid_base,
        "x_train_external": x_train_external,
        "x_valid_external": x_valid_external,
        "external_train": external_train,
        "external_valid": external_valid,
        "coverage_training": external_coverage_report(rows, external_train),
        "coverage_validation": external_coverage_report(fold.ledger, external_valid),
        "cutoff": str(cutoff),
    }


def _run_prepared_origin(
    *,
    context: dict[str, object],
    output_dir: Path,
    params: dict[str, object],
    resource: ResourceTracker,
    log,
    control_dir: Path,
    fit_external: bool,
    fit_base_if_no_control: bool = False,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    import lightgbm as lgb

    origin = str(context["origin"])
    role = str(context["role"])
    fold = context["fold"]
    supervision = context["supervision"]
    rows = context["rows"]
    y_delta = context["y_delta"]
    y_target = context["y_target"]
    weights = context["weights"]
    x_train_base = context["x_train_base"]
    x_valid_base = context["x_valid_base"]
    x_train_external = context["x_train_external"]
    x_valid_external = context["x_valid_external"]

    def emit(payload: dict[str, object]) -> None:
        record = {**payload, "resource": resource.snapshot()}
        line = json.dumps(record, sort_keys=True, default=str)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    emit({
        "phase": "origin_start",
        "origin": origin,
        "role": role,
        "training_rows": int(len(rows)),
        "validation_rows": int(len(fold.ledger)),
        "cutoff": context["cutoff"],
        "fit_external": bool(fit_external),
    })
    try:
        reused_booster, reuse_info = _reuse_b3c_control(
            control_dir=control_dir,
            origin=origin,
            x_train=x_train_base,
            x_valid=x_valid_base,
            supervision=supervision,
        )
    except FileNotFoundError:
        if not fit_base_if_no_control:
            raise
        reused_booster = None
        reuse_info = {
            "status": "not_available_transfer_fit",
            "control_dir": str(control_dir),
            "reason": "no saved B3-C control exists for the newly declared 2009-01 transfer origin",
        }
    base_metric, base_oof, base_detail = _run_candidate(
        lgb=lgb,
        origin=origin,
        role=role,
        candidate=BASE_CANDIDATE,
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
        reused_booster=reused_booster,
        reuse_info=reuse_info,
    )
    emit({"phase": "candidate_complete", "origin": origin, "candidate": BASE_CANDIDATE, "h1_7_raw_rmse": base_metric["h1_7_raw_rmse"], "weighted_proxy": base_metric["test_horizon_weighted_validation_proxy"], "model_status": base_metric["model"]["status"]})
    results = [base_metric]
    oofs = [base_oof]
    fit_details: list[dict[str, object]] = [{**base_detail, "reuse_check": reuse_info}]
    if fit_external:
        external_metric, external_oof, external_detail = _run_candidate(
            lgb=lgb,
            origin=origin,
            role=role,
            candidate=EXTERNAL_CANDIDATE,
            x_train=x_train_external,
            x_valid=x_valid_external,
            y_delta=y_delta,
            y_target=y_target,
            weights=weights,
            fold=fold,
            output_dir=output_dir,
            params=params,
            supervision=supervision,
            resource=resource,
        )
        emit({"phase": "candidate_complete", "origin": origin, "candidate": EXTERNAL_CANDIDATE, "h1_7_raw_rmse": external_metric["h1_7_raw_rmse"], "weighted_proxy": external_metric["test_horizon_weighted_validation_proxy"], "model_status": external_metric["model"]["status"]})
        results.append(external_metric)
        oofs.append(external_oof)
        fit_details.append(external_detail)
    detail = {
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
            "source_rows": int(len(context["view"].source)),
            "dropped_missing_anchor": int(context["sampled"].dropped_missing_anchor),
            "horizon_counts": {str(int(k)): int(v) for k, v in rows["h"].value_counts().sort_index().items()},
            "target_cutoff": context["cutoff"],
        },
        "validation": {
            "rows": int(len(fold.ledger)),
            "horizons_present": sorted(int(value) for value in fold.ledger["h"].unique()),
            "exclusions": fold.exclusions,
        },
        "feature_contract": {
            "base_candidate": BASE_CANDIDATE,
            "external_candidate": EXTERNAL_CANDIDATE,
            "historical_b3_count": len(ORIGINAL_B3_FEATURE_COLUMNS),
            "b3c_count": len(COORDINATE_FREE_B3_FEATURE_COLUMNS),
            "external_count": len(GLDAS_EXTERNAL_FEATURE_COLUMNS),
            "external_total_count": len(COORDINATE_FREE_B3_FEATURE_COLUMNS) + len(GLDAS_EXTERNAL_FEATURE_COLUMNS),
            "b3c_allowlist": list(COORDINATE_FREE_B3_FEATURE_COLUMNS),
            "external_allowlist": list(GLDAS_EXTERNAL_FEATURE_COLUMNS),
            "b3c_provenance": coordinate_free_b3_feature_provenance(),
            "base_training_matrix_sha256": _hash_frame(x_train_base),
            "base_validation_matrix_sha256": _hash_frame(x_valid_base),
            "external_training_matrix_sha256": _hash_frame(x_train_external),
            "external_validation_matrix_sha256": _hash_frame(x_valid_external),
            "paired_base_values_identical": True,
            "prohibited_identifiers_absent": True,
        },
        "external_data": {
            "product": GLDAS_PRODUCT,
            "feature_names": list(GLDAS_EXTERNAL_FEATURE_COLUMNS),
            "causal_cutoff": "current source month t; anchor <= t; previous exact calendar month t-1",
            "units": "kg m-2",
            "missingness": "missing inputs propagate NaN; no partial sums or zero fill",
            "training_coverage": context["coverage_training"],
            "validation_coverage": context["coverage_validation"],
        },
        "fits": fit_details,
        "results": results,
        "no_test_predictions": True,
        "submission_written": False,
    }
    _atomic_json(output_dir / f"origin_{origin}_details.json", detail)
    return detail, pd.concat(oofs, ignore_index=True), pd.DataFrame(results)


def _inner_gate(results: list[dict[str, object]]) -> dict[str, object]:
    by_key = {(row["origin"], row["candidate"]): row for row in results}
    rows: list[dict[str, object]] = []
    for origin in INNER_ORIGINS:
        base = by_key[(origin, BASE_CANDIDATE)]
        ext = by_key[(origin, EXTERNAL_CANDIDATE)]
        gain = float(base["h1_7_raw_rmse"] - ext["h1_7_raw_rmse"])
        rows.append({
            "origin": origin,
            "base_h1_7_raw_rmse": base["h1_7_raw_rmse"],
            "external_h1_7_raw_rmse": ext["h1_7_raw_rmse"],
            "raw_rmse_gain_base_minus_external": gain,
            "strict_improvement": bool(gain > 0.0),
            "base_weighted_proxy": base["test_horizon_weighted_validation_proxy"],
            "external_weighted_proxy": ext["test_horizon_weighted_validation_proxy"],
        })
    return {
        "status": "evaluated",
        "origins": rows,
        "pass": bool(all(row["strict_improvement"] for row in rows)),
        "rule": "GLDAS candidate must strictly improve raw h1-7 RMSE on both 2003-04 and 2004-04; weighted metrics cannot rescue a failure.",
    }


def _outer_gate(results: list[dict[str, object]], validation_by_month: pd.DataFrame) -> dict[str, object]:
    by_key = {(row["origin"], row["candidate"]): row for row in results}
    recent: list[dict[str, object]] = []
    for origin in RECENT_ORIGINS:
        base = by_key[(origin, BASE_CANDIDATE)]
        ext = by_key[(origin, EXTERNAL_CANDIDATE)]
        gain = float(base["h1_7_raw_rmse"] - ext["h1_7_raw_rmse"])
        recent.append({
            "origin": origin,
            "base_h1_7_raw_rmse": base["h1_7_raw_rmse"],
            "external_h1_7_raw_rmse": ext["h1_7_raw_rmse"],
            "raw_rmse_gain_base_minus_external": gain,
            "strict_raw_improvement": bool(gain > 0.0),
            "horizons_present_base": base["h1_7_horizons_present"],
            "horizons_present_external": ext["h1_7_horizons_present"],
            "december_weighted_nonregression": (
                origin != "2014-12"
                or (
                    base["test_horizon_weighted_validation_proxy"] is not None
                    and ext["test_horizon_weighted_validation_proxy"] is not None
                    and ext["test_horizon_weighted_validation_proxy"] <= base["test_horizon_weighted_validation_proxy"]
                )
            ),
        })
    month = validation_by_month.loc[
        validation_by_month["scope"].eq("h1_7") & validation_by_month["rows"].ge(1000)
    ].copy()
    pivot = month.pivot_table(index=["origin", "source_month"], columns="candidate", values="raw_rmse", aggfunc="first").reset_index()
    regressions: list[dict[str, object]] = []
    if BASE_CANDIDATE in pivot.columns and EXTERNAL_CANDIDATE in pivot.columns:
        pivot["external_minus_base_raw_rmse"] = pivot[EXTERNAL_CANDIDATE] - pivot[BASE_CANDIDATE]
        for row in pivot.to_dict(orient="records"):
            if float(row["external_minus_base_raw_rmse"]) > 0.02:
                regressions.append({**row, "regression_gate_pass": False})
    passed = bool(
        all(row["strict_raw_improvement"] for row in recent)
        and all(row["december_weighted_nonregression"] for row in recent)
        and not regressions
    )
    return {
        "status": "evaluated",
        "recent_origins": recent,
        "monthly_regressions_over_0.02": regressions,
        "pass": passed,
        "thresholds": {
            "recent_raw_h1_7_improvement": 0.0,
            "december_weighted_validation_proxy_regression": 0.0,
            "monthly_raw_rmse_regression_rows_at_least_1000": 0.02,
        },
        "note": "2014-04 lacks h=4; its present-support raw h1-7 metric is reported without inventing a complete weighted proxy.",
    }


def _transfer_gate(results: list[dict[str, object]]) -> dict[str, object]:
    by_key = {(row["origin"], row["candidate"]): row for row in results}
    base = by_key[(OLDER_ORIGIN, BASE_CANDIDATE)]
    ext = by_key[(OLDER_ORIGIN, EXTERNAL_CANDIDATE)]
    gain = float(base["h1_7_raw_rmse"] - ext["h1_7_raw_rmse"])
    return {
        "status": "evaluated",
        "origin": OLDER_ORIGIN,
        "base_h1_7_raw_rmse": base["h1_7_raw_rmse"],
        "external_h1_7_raw_rmse": ext["h1_7_raw_rmse"],
        "raw_rmse_gain_base_minus_external": gain,
        "maximum_allowed_regression": 0.003,
        "pass": bool(gain >= -0.003),
        "rule": "A regression greater than 0.003 raw h1-7 RMSE blocks promotion; transfer cannot rescue a failed recent gate.",
    }


def _load_product_gate(path: Path) -> dict[str, object]:
    gate = json.loads(path.read_text(encoding="utf-8"))
    _assert_no_secret_like_fields(gate)
    if gate.get("status") != "pass_conditional_historical_proxy":
        raise ValueError("GLDAS product gate is not the documented conditional historical proxy pass")
    if gate.get("selected_stream") != "main_production":
        raise ValueError("GLDAS selected stream is not main production")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--external-store-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--product-gate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
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
    if args.num_threads < 1 or args.num_threads > MAX_NUM_THREADS:
        raise ValueError(f"--num-threads must be in [1, {MAX_NUM_THREADS}]")
    num_threads = min(int(args.num_threads), os.cpu_count() or 1)
    train_path = args.data_dir / "Train.csv"
    test_path = args.data_dir / "Test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Train.csv/Test.csv missing")
    store, store_manifest, request_plan = _load_store(args.external_store_dir)
    sample_verification_path = args.sample_dir / "verification.json"
    if not sample_verification_path.is_file():
        raise FileNotFoundError(sample_verification_path)
    sample_verification = json.loads(sample_verification_path.read_text(encoding="utf-8"))
    if sample_verification.get("status") != "verified_netcdf" or sample_verification.get("product") != GLDAS_PRODUCT:
        raise ValueError("retained GLDAS sample verification is not a successful selected-product verification")
    product_gate_path = args.product_gate or (args.sample_dir / "product_gate.json")
    product_gate = _load_product_gate(product_gate_path)
    external_access = _safe_external_access(sample_verification)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest: dict[str, object] = {
        "status": "running",
        "stage": "GLDAS_EXTERNAL_B3C_GATED_EXPERIMENT",
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
        "base_candidate": BASE_CANDIDATE,
        "external_candidate": EXTERNAL_CANDIDATE,
        "no_test_predictions": True,
        "test_labels_read": False,
        "test_predictions_written": False,
        "submission_written": False,
        "external_access": external_access,
        "product_gate": product_gate,
        "external_store": {
            "directory": str(args.external_store_dir),
            "manifest": "manifest.json",
            "manifest_sha256": _sha256(args.external_store_dir / "manifest.json"),
            "raw_store": store_manifest.get("raw_store"),
            "raw_store_key_sha256": store.store_key_sha256,
            "request_plan_coordinate_sha256": request_plan.get("coordinate_sha256"),
        },
        "data": {
            "train_path": str(train_path),
            "train_sha256": _sha256(train_path),
            "test_geometry_path": str(test_path),
            "test_geometry_sha256": _sha256(test_path),
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count() or 1,
            "packages": {name: _version(name) for name in ("numpy", "pandas", "lightgbm", "netCDF4")},
        },
    }
    _atomic_json(args.output_dir / "manifest.json", manifest)
    _atomic_json(args.output_dir / "external_access.json", external_access)
    resource = ResourceTracker()
    resource.start()
    try:
        prefit = {
            "b3c": _b3c_prefit_contract_checks(),
            "gldas": run_gldas_external_prefit_checks(),
        }
        _atomic_json(args.output_dir / "prefit_checks.json", prefit)
        with (args.output_dir / "events.jsonl").open("a", encoding="utf-8") as log:
            def emit(payload: dict[str, object]) -> None:
                record = {**payload, "resource": resource.snapshot()}
                line = json.dumps(record, sort_keys=True, default=str)
                print(line, flush=True)
                log.write(line + "\n")
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
            recomputed_plan = build_external_request_plan(train, template)
            if recomputed_plan.get("coordinate_sha256") != request_plan.get("coordinate_sha256") or recomputed_plan.get("periods") != request_plan.get("periods"):
                raise AssertionError("external acquisition request plan does not match current replay geometry")
            manifest["request_plan"] = {
                "period_count": recomputed_plan["period_count"],
                "coordinate_count": recomputed_plan["coordinate_count"],
                "coordinate_sha256": recomputed_plan["coordinate_sha256"],
                "periods": recomputed_plan["periods"],
            }
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
                "base_candidate": BASE_CANDIDATE,
                "external_candidate": EXTERNAL_CANDIDATE,
                "base_feature_count": len(COORDINATE_FREE_B3_FEATURE_COLUMNS),
                "external_feature_count": len(GLDAS_EXTERNAL_FEATURE_COLUMNS),
                "external_feature_names": list(GLDAS_EXTERNAL_FEATURE_COLUMNS),
                "product": GLDAS_PRODUCT,
                "no_test_predictions": True,
            })
            control_dir = args.control_dir.resolve()
            all_results: list[dict[str, object]] = []
            all_oofs: list[pd.DataFrame] = []
            all_fit_metrics: list[pd.DataFrame] = []
            origin_details: list[dict[str, object]] = []
            for origin in INNER_ORIGINS:
                context = _prepare_origin(train=train, labels=labels, origin=origin, template=template, store=store)
                detail, oof, metrics = _run_prepared_origin(
                    context=context,
                    output_dir=args.output_dir,
                    params=params,
                    resource=resource,
                    log=log,
                    control_dir=control_dir,
                    fit_external=True,
                )
                origin_details.append(detail)
                all_results.extend(detail["results"])
                all_oofs.append(oof)
                all_fit_metrics.append(metrics)
            inner_gate = _inner_gate(all_results)
            _atomic_json(args.output_dir / "inner_gate.json", inner_gate)
            emit({"phase": "inner_gate", "gate": inner_gate})
            outer_gate: dict[str, object] = {"status": "not_run", "reason": "inner_gate_failed"}
            transfer_gate: dict[str, object] = {"status": "not_run", "reason": "outer_gate_not_passed"}
            if inner_gate["pass"]:
                for origin in RECENT_ORIGINS:
                    context = _prepare_origin(train=train, labels=labels, origin=origin, template=template, store=store)
                    detail, oof, metrics = _run_prepared_origin(
                        context=context,
                        output_dir=args.output_dir,
                        params=params,
                        resource=resource,
                        log=log,
                        control_dir=control_dir,
                        fit_external=True,
                    )
                    origin_details.append(detail)
                    all_results.extend(detail["results"])
                    all_oofs.append(oof)
                    all_fit_metrics.append(metrics)
                validation_so_far = pd.concat(all_oofs, ignore_index=True)
                outer_gate = _outer_gate(all_results, _aggregate(validation_so_far, ["source_month"]))
                _atomic_json(args.output_dir / "outer_gate.json", outer_gate)
                emit({"phase": "outer_gate", "gate": outer_gate})
                if outer_gate["pass"]:
                    context = _prepare_origin(train=train, labels=labels, origin=OLDER_ORIGIN, template=template, store=store)
                    detail, oof, metrics = _run_prepared_origin(
                        context=context,
                        output_dir=args.output_dir,
                        params=params,
                        resource=resource,
                        log=log,
                        control_dir=control_dir,
                        fit_external=True,
                        fit_base_if_no_control=True,
                    )
                    origin_details.append(detail)
                    all_results.extend(detail["results"])
                    all_oofs.append(oof)
                    all_fit_metrics.append(metrics)
                    transfer_gate = _transfer_gate(all_results)
                    _atomic_json(args.output_dir / "transfer_gate.json", transfer_gate)
                    emit({"phase": "transfer_gate", "gate": transfer_gate})
            else:
                for origin in RECENT_ORIGINS:
                    context = _prepare_origin(train=train, labels=labels, origin=origin, template=template, store=store)
                    detail, oof, metrics = _run_prepared_origin(
                        context=context,
                        output_dir=args.output_dir,
                        params=params,
                        resource=resource,
                        log=log,
                        control_dir=control_dir,
                        fit_external=False,
                    )
                    origin_details.append(detail)
                    all_results.extend(detail["results"])
                    all_oofs.append(oof)
                    all_fit_metrics.append(metrics)
                _atomic_json(args.output_dir / "outer_gate.json", outer_gate)
                _atomic_json(args.output_dir / "transfer_gate.json", transfer_gate)

            # The inner gate can pass while the recent outer gate fails.  Keep
            # the declared transfer artifact durable in that branch too, so a
            # completed run never advertises a file that was not written.
            transfer_path = args.output_dir / "transfer_gate.json"
            if not transfer_path.is_file():
                _atomic_json(transfer_path, transfer_gate)

            validation_long = pd.concat(all_oofs, ignore_index=True)
            fit_metrics = pd.concat(all_fit_metrics, ignore_index=True)
            validation_by_month = _aggregate(validation_long, ["source_month"])
            validation_by_horizon = _aggregate(validation_long, ["h"])
            validation_overall = _aggregate(validation_long, [])
            validation_long.to_csv(args.output_dir / "validation_long.csv.gz", index=False, compression="gzip")
            fit_metrics.to_csv(args.output_dir / "fit_metrics.csv", index=False)
            validation_by_month.to_csv(args.output_dir / "validation_by_source_month.csv", index=False)
            validation_by_horizon.to_csv(args.output_dir / "validation_by_horizon.csv", index=False)
            validation_overall.to_csv(args.output_dir / "validation_overall.csv", index=False)
            all_origin_names = [detail["origin"] for detail in origin_details]
            by_key = {(row["origin"], row["candidate"]): row for row in all_results}
            comparisons = []
            for origin in all_origin_names:
                base = by_key[(origin, BASE_CANDIDATE)]
                ext = by_key.get((origin, EXTERNAL_CANDIDATE))
                comparisons.append({
                    "origin": origin,
                    "base_h1_7_raw_rmse": base["h1_7_raw_rmse"],
                    "external_h1_7_raw_rmse": None if ext is None else ext["h1_7_raw_rmse"],
                    "raw_rmse_gain_base_minus_external": None if ext is None else float(base["h1_7_raw_rmse"] - ext["h1_7_raw_rmse"]),
                    "base_weighted_proxy": base["test_horizon_weighted_validation_proxy"],
                    "external_weighted_proxy": None if ext is None else ext["test_horizon_weighted_validation_proxy"],
                })
            external_fit_ran = any(row["candidate"] == EXTERNAL_CANDIDATE for row in all_results)
            promotion = bool(inner_gate.get("pass") and outer_gate.get("pass") and transfer_gate.get("pass"))
            decision = {
                "status": "promote_b3c_gldas" if promotion else "retain_b3c",
                "external_fit_ran": external_fit_ran,
                "inner_gate": inner_gate,
                "outer_gate": outer_gate,
                "transfer_gate": transfer_gate,
                "comparisons": comparisons,
                "historical_failure_preserved": HISTORICAL_EXTERNAL_ACCESS_FAILURE,
                "no_test_predictions": True,
                "submission_written": False,
            }
            _atomic_json(args.output_dir / "decision.json", decision)
            manifest.update({
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "resource_peak": resource.snapshot(),
                "origins_completed": all_origin_names,
                "fit_count": int(sum(1 for row in all_results if row["model"]["status"] == "fitted")),
                "model_fit_count": int(sum(1 for row in all_results if row["model"]["status"] == "fitted")),
                "decision": decision,
                "artifacts": {
                    "manifest": "manifest.json",
                    "resolved_config": "resolved_config.json",
                    "prefit_checks": "prefit_checks.json",
                    "external_access": "external_access.json",
                    "inner_gate": "inner_gate.json",
                    "outer_gate": "outer_gate.json",
                    "transfer_gate": "transfer_gate.json",
                    "decision": "decision.json",
                    "validation_long": "validation_long.csv.gz",
                    "validation_by_source_month": "validation_by_source_month.csv",
                    "validation_by_horizon": "validation_by_horizon.csv",
                    "validation_overall": "validation_overall.csv",
                    "events": "events.jsonl",
                },
            })
            _atomic_json(args.output_dir / "external_access.json", external_access)
            _atomic_json(args.output_dir / "manifest.json", manifest)
            emit({"phase": "completed", "decision": decision, "resource": resource.snapshot()})
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
        gc.collect()


if __name__ == "__main__":
    main()
