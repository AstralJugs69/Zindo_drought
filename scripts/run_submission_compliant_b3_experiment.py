"""Build the submission-compliant B3-C reference on the remote worker.

This runner deliberately fits only the coordinate-free B3-C reference.  The
single GLDAS-2.1 Noah augmentation is gated by a separately measured,
sanitized sample-GET record; no stale access constant, substitute dataset, or
partial external fit is silently introduced.  Test geometry is used only for
label-free replay construction.  No Test predictions, submissions, uploads,
calibration, ensembles, or post-decision experiments are created.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
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
    _fold_for_origin,
    _hash_frame,
    _hash_values,
    _run_candidate,
    _version,
)
from scripts.run_lgbm_core import DEFAULT_NUM_THREADS, DEFAULT_PARAMS, MAX_NUM_THREADS  # noqa: E402
from src.availability import build_replay_observation_view  # noqa: E402
from src.compliant_b3 import (  # noqa: E402
    COORDINATE_FREE_B3_FEATURE_COLUMNS,
    ORIGINAL_B3_FEATURE_COLUMNS,
    assert_coordinate_free_b3_schema,
    build_coordinate_free_b3_matrix,
    coordinate_free_b3_feature_provenance,
    write_coordinate_free_b3_audit,
)
from src.ml_features import (  # noqa: E402
    SOURCE_CORE_COLUMNS,
    SOURCE_HYDRO_HISTORY_COLUMNS,
    build_sampled_training_rows,
    horizon_rebalance_weights,
)
from src.neural_sequence import build_b3_feature_maps  # noqa: E402
from src.regional_context import HYDRO_COLUMNS  # noqa: E402
from src.validation import build_test_mask_template  # noqa: E402


SEED = 20260908
ROUNDS = 98
NUM_LEAVES = 63
MIN_DATA_IN_LEAF = 1000
INNER_ORIGINS = ("2003-04", "2004-04")
RECENT_ORIGINS = ("2014-04", "2014-12")
OLDER_ORIGIN = "2009-01"
END_MONTHS = {"2014-04": "2014-10", "2014-12": "2015-06"}
REFERENCE_ORIGINS = (*INNER_ORIGINS, *RECENT_ORIGINS)
TRAIN_COLUMNS = [
    "sample_id", "time", "lat", "lon", "TWS_t", "month_sin", "month_cos",
    *HYDRO_COLUMNS, "target",
]
TEST_GEOMETRY_COLUMNS = ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]
GLDAS_SAMPLE_URL = (
    "https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_M.2.1/2014/"
    "GLDAS_NOAH025_M.A201404.021.nc4"
)
GLDAS_METADATA_URL = "https://disc.gsfc.nasa.gov/datasets/GLDAS_NOAH025_M_2.1/summary"
GLDAS_README_URL = (
    "https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_M.2.1/doc/"
    "README_GLDAS2.pdf"
)
GLDAS_VARIABLES = {
    "soil_moisture_0_10cm": {"source": "SoilMoi0_10cm_inst", "units": "kg m-2"},
    "soil_moisture_10_40cm": {"source": "SoilMoi10_40cm_inst", "units": "kg m-2"},
    "soil_moisture_40_100cm": {"source": "SoilMoi40_100cm_inst", "units": "kg m-2"},
    "soil_moisture_100_200cm": {"source": "SoilMoi100_200cm_inst", "units": "kg m-2"},
    "snow_water_equivalent": {"source": "SWE_inst", "units": "kg m-2"},
    "canopy_storage": {"source": "CanopInt_inst", "units": "kg m-2"},
}
GLDAS_EXTERNAL_FEATURES = (
    "gldas_soil_moisture_0_10cm_t",
    "gldas_soil_moisture_10_40cm_t",
    "gldas_soil_moisture_40_100cm_t",
    "gldas_soil_moisture_100_200cm_t",
    "gldas_swe_t",
    "gldas_canopy_storage_t",
    "gldas_storage_sum_t",
    "gldas_storage_sum_anchor",
    "gldas_storage_sum_gap",
    "gldas_storage_sum_prev_month_delta",
)


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


def _prefit_contract_checks() -> dict[str, object]:
    """Exercise schema, target-blindness, row-order, and future-cutoff contracts."""
    rows: list[dict[str, object]] = []
    months = pd.date_range("2018-01-01", periods=24, freq="MS")
    for month_index, month in enumerate(months):
        for location_index, (lat, lon) in enumerate(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))):
            row: dict[str, object] = {
                "sample_id": f"{location_index}_{month:%Y%m}",
                "time": month,
                "lat": lat,
                "lon": lon,
                "month_sin": np.sin(2.0 * np.pi * month.month / 12.0),
                "month_cos": np.cos(2.0 * np.pi * month.month / 12.0),
            }
            for variable_index, variable in enumerate(HYDRO_COLUMNS):
                row[variable] = float(month_index + location_index + variable_index)
            rows.append(row)
    source = pd.DataFrame(rows)
    structural = source.loc[:, ["sample_id", "time", "lat", "lon"]].copy()
    structural["TWS_t"] = np.arange(len(structural), dtype=np.float32)
    ledger = pd.DataFrame({
        "sample_id": ["0_201906", "1_201906"],
        "source_date": pd.to_datetime(["2019-06-01", "2019-06-01"]),
        "last_observed_date": pd.to_datetime(["2019-04-01", "2019-05-01"]),
        "last_observed_TWS": [10.0, 11.0],
        "h": [3, 2],
        "lat": [0.0, 1.0],
        "lon": [0.0, 0.0],
    })

    maps = build_b3_feature_maps(source)
    baseline = build_coordinate_free_b3_matrix(ledger, source, structural, maps)
    assert_coordinate_free_b3_schema(baseline.columns)
    if baseline.shape != (2, len(COORDINATE_FREE_B3_FEATURE_COLUMNS)):
        raise AssertionError("B3-C prefit width/row contract failed")
    if set(baseline.columns).intersection({"lat", "lon", "cell_id", "location_id"}):
        raise AssertionError("B3-C prefit contains a prohibited model input")

    shuffled_source = source.sample(frac=1.0, random_state=17).reset_index(drop=True)
    shuffled_structural = structural.sample(frac=1.0, random_state=31).reset_index(drop=True)
    reordered = build_coordinate_free_b3_matrix(
        ledger, shuffled_source, shuffled_structural, build_b3_feature_maps(shuffled_source)
    )
    if not np.array_equal(baseline.to_numpy(), reordered.to_numpy(), equal_nan=True):
        raise AssertionError("B3-C output is not stable under source row reordering")

    future_changed = source.copy()
    future_changed.loc[
        future_changed["time"].ge(pd.Timestamp("2019-07-01")), HYDRO_COLUMNS[0]
    ] += 1_000_000.0
    future = build_coordinate_free_b3_matrix(
        ledger, future_changed, structural, build_b3_feature_maps(future_changed)
    )
    if not np.array_equal(baseline.to_numpy(), future.to_numpy(), equal_nan=True):
        raise AssertionError("future source values influenced a pre-cutoff B3-C row")

    target_blind_rejected = False
    try:
        build_b3_feature_maps(source.assign(target=0.0))
    except AssertionError:
        target_blind_rejected = True
    if not target_blind_rejected:
        raise AssertionError("B3-C source map accepted a target column")

    if len(GLDAS_EXTERNAL_FEATURES) != 10:
        raise AssertionError("GLDAS candidate must define exactly ten external features")
    return {
        "status": "passed",
        "historical_b3_width": len(ORIGINAL_B3_FEATURE_COLUMNS),
        "coordinate_free_b3_width": len(COORDINATE_FREE_B3_FEATURE_COLUMNS),
        "coordinate_free_schema": list(COORDINATE_FREE_B3_FEATURE_COLUMNS),
        "provenance_groups": sorted(set(coordinate_free_b3_feature_provenance().values())),
        "row_order_invariant": True,
        "future_cutoff_invariant": True,
        "target_blind_source_rejection": True,
        "external_feature_count": len(GLDAS_EXTERNAL_FEATURES),
        "external_feature_names": list(GLDAS_EXTERNAL_FEATURES),
    }


HISTORICAL_EXTERNAL_ACCESS_FAILURE = {
    "status": "blocked_http_401_no_noninteractive_earthdata_credentials",
    "source": "gcp_submission_compliant_b3c_20260910_v1/external_access.json",
    "sample_request": "HEAD returned 200; authenticated GET returned HTTP 401",
    "credentials_checked_without_exposure": {
        "netrc": False,
        "earthdata_token": False,
        "earthdata_username": False,
        "earthdata_config": False,
    },
    "bulk_acquisition_started": False,
    "substitute_dataset_used": False,
    "external_fit_started": False,
    "decision": "stop_external_acquisition_and_finish_b3c_stage1",
}


def _safe_measurement_text(value: object, *, default: str = "") -> str:
    text = str(value if value is not None else default).replace("\r", " ").replace("\n", " ").strip()
    lowered = text.lower()
    if "://" in text or "=" in text or any(
        token in lowered for token in ("password", "passwd", "token", "cookie", "authorization")
    ):
        return "redacted_error"
    return text[:160]


def _external_access_record(measured: Mapping[str, object]) -> dict[str, object]:
    """Record measured sample access while retaining the prior result as history.

    Only a strict allowlist of probe fields is copied.  This prevents a future
    probe implementation from accidentally persisting credentials, cookies,
    response bodies, or redirect query parameters in the experiment record.
    """
    if not isinstance(measured, Mapping):
        raise TypeError("measured GLDAS access result must be a mapping")
    status = _safe_measurement_text(measured.get("status"), default="measurement_missing")
    if not status or status == "measurement_missing":
        raise ValueError("measured GLDAS access result has no status")
    credentials = measured.get("credentials", {})
    if not isinstance(credentials, Mapping):
        credentials = {}
    verification = measured.get("netcdf_verification", {})
    if not isinstance(verification, Mapping):
        verification = {"status": _safe_measurement_text(verification)}
    safe_verification: dict[str, object] = {
        "status": _safe_measurement_text(verification.get("status"), default="not_run"),
    }
    for key in (
        "file_format", "magic", "product", "sample_month", "bytes", "dimensions",
        "time_months", "variable_units", "variable_shapes", "product_marker_attribute_names",
    ):
        if key in verification:
            value = verification[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe_verification[key] = value
            elif isinstance(value, (list, tuple)):
                safe_verification[key] = list(value)
            elif isinstance(value, Mapping):
                safe_verification[key] = dict(value)
    return {
        "status": status,
        "dataset": "GLDAS_NOAH025_M.2.1",
        "product": "GLDAS-2.1 Noah monthly 0.25-degree main production",
        "sample_url": GLDAS_SAMPLE_URL,
        "metadata_url": GLDAS_METADATA_URL,
        "readme_url": GLDAS_README_URL,
        "sample_month": "2014-04",
        "sample_request": {
            "method": "GET",
            "redirect_policy": "HTTPS-only",
            "tls_verification": "default_secure_verification",
            "http_status": measured.get("http_status"),
            "final_host": _safe_measurement_text(measured.get("final_host")),
            "content_type": _safe_measurement_text(measured.get("content_type")),
            "sanitized_error": _safe_measurement_text(measured.get("sanitized_error")),
        },
        "credentials_checked_without_exposure": {
            "netrc_present": bool(measured.get("netrc_present", credentials.get("netrc_present", False))),
            "netrc_permissions_ok": bool(measured.get("netrc_permissions_ok", credentials.get("netrc_permissions_ok", False))),
            "netrc_machine_present": bool(measured.get("netrc_machine_present", credentials.get("netrc_machine_present", False))),
            "credential_values_recorded": False,
            "cookies_recorded": False,
        },
        "sample_file_retained": bool(measured.get("sample_file_retained", False)),
        "netcdf_verification": safe_verification,
        "bulk_acquisition_started": False,
        "substitute_dataset_used": False,
        "external_fit_started": False,
        "product_gate": {
            "status": "documented_historical_proxy_eligible",
            "basis": "Official GLDAS-2.1 documentation covers the archive from 2000 onward; challenge periods are historical, so archived monthly data are a documented historical proxy. Main-production latency does not invalidate already archived months.",
            "grace_assimilation": "not_used; GLDAS-2.2/GRACE-derived products remain excluded",
            "sample_verification_required_before_bulk": True,
            "official_sources": [GLDAS_README_URL, GLDAS_METADATA_URL],
        },
        "historical_failure": HISTORICAL_EXTERNAL_ACCESS_FAILURE,
        "variables": GLDAS_VARIABLES,
        "feature_contract": list(GLDAS_EXTERNAL_FEATURES),
        "causal_rules": {
            "current": "source month t only",
            "anchor": "focal legal last_observed_date month <= t",
            "previous": "exact previous calendar month t-1 only",
            "missingness": "missing inputs propagate NaN; no partial sums or zero fill",
            "units": "retain original kg m-2 values; no anomaly or scale transformation",
            "coordinates": "sampling/indexing metadata only; never model inputs",
        },
    }


def _load_external_access_measurement(path: Path) -> dict[str, object]:
    """Load only a JSON probe result; reject files that look like secret dumps."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GLDAS access measurement must be a JSON object")
    serialized = json.dumps(payload, sort_keys=True, default=str).lower()
    if any(token in serialized for token in ("password", "passwd", "authorization", "cookie", "token")):
        raise ValueError("GLDAS access measurement contains prohibited secret-like fields")
    return payload


def _run_origin(
    *,
    train: pd.DataFrame,
    labels: pd.DataFrame,
    origin: str,
    template: pd.DataFrame,
    output_dir: Path,
    params: dict[str, object],
    resource: ResourceTracker,
    log,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    import lightgbm as lgb

    def emit(payload: dict[str, object]) -> None:
        record = {**payload, "resource": resource.snapshot()}
        line = json.dumps(record, sort_keys=True, default=str)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    if origin in INNER_ORIGINS:
        fold, role = _fold_for_origin(train, origin, template)
    else:
        fold, role = _fold_for_origin(train, origin, None)
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
    supervision = {
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
    emit({
        "phase": "origin_start",
        "origin": origin,
        "role": role,
        "training_rows": len(rows),
        "validation_rows": len(fold.ledger),
        "cutoff": str(cutoff),
    })

    maps = build_b3_feature_maps(view.source)
    x_train = build_coordinate_free_b3_matrix(rows, view.source, view.structural, maps)
    x_valid = build_coordinate_free_b3_matrix(fold.ledger, view.source, view.structural, maps)
    assert_coordinate_free_b3_schema(x_train.columns)
    assert_coordinate_free_b3_schema(x_valid.columns)
    if x_train.columns.tolist() != x_valid.columns.tolist():
        raise AssertionError("B3-C train/validation schema differs")
    if x_train.shape[1] != 444:
        raise AssertionError("B3-C must have exactly 444 model inputs")
    emit({
        "phase": "prefit_features_ready",
        "origin": origin,
        "feature_count": x_train.shape[1],
        "feature_schema_sha256": _hash_values(x_train.columns.tolist()),
        "training_matrix_sha256": _hash_frame(x_train),
        "validation_matrix_sha256": _hash_frame(x_valid),
        "supervision": supervision,
        "source_rows": len(view.source),
        "view_withheld_window_rows": int(view.withheld_window_rows),
    })

    metric, oof, fit_detail = _run_candidate(
        lgb=lgb,
        origin=origin,
        role=role,
        candidate="B3C",
        x_train=x_train,
        x_valid=x_valid,
        y_delta=y_delta,
        y_target=y_target,
        weights=weights,
        fold=fold,
        output_dir=output_dir,
        params=params,
        supervision=supervision,
        resource=resource,
    )
    emit({
        "phase": "candidate_complete",
        "origin": origin,
        "candidate": "B3C",
        "h1_7_raw_rmse": metric["h1_7_raw_rmse"],
        "weighted_proxy": metric["test_horizon_weighted_validation_proxy"],
        "model_status": metric["model"]["status"],
    })
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
            "horizons_present": sorted(int(value) for value in fold.ledger["h"].unique()),
            "exclusions": fold.exclusions,
        },
        "feature_contract": {
            "historical_b3_count": len(ORIGINAL_B3_FEATURE_COLUMNS),
            "b3c_count": len(COORDINATE_FREE_B3_FEATURE_COLUMNS),
            "removed_model_inputs": ["lat", "lon"],
            "ordered_allowlist": list(COORDINATE_FREE_B3_FEATURE_COLUMNS),
            "provenance": coordinate_free_b3_feature_provenance(),
            "training_matrix_sha256": _hash_frame(x_train),
            "validation_matrix_sha256": _hash_frame(x_valid),
        },
        "fit": fit_detail,
        "result": metric,
        "no_test_predictions": True,
        "submission_written": False,
    }
    _atomic_json(output_dir / f"origin_{origin}_details.json", details)
    return details, oof, pd.DataFrame([metric])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--external-access-measurement",
        type=Path,
        required=True,
        help="sanitized JSON emitted by probe_gldas_access.py; no secret fields permitted",
    )
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
    if not args.external_access_measurement.is_file():
        raise FileNotFoundError(args.external_access_measurement)
    external_access = _external_access_record(
        _load_external_access_measurement(args.external_access_measurement)
    )
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
        "stage": "SUBMISSION_COMPLIANT_B3C_REFERENCE",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit": head,
        "branch": branch,
        "seed": SEED,
        "rounds": ROUNDS,
        "num_leaves": NUM_LEAVES,
        "min_data_in_leaf": MIN_DATA_IN_LEAF,
        "requested_num_threads": int(args.num_threads),
        "num_threads": num_threads,
        "reference_origins": list(REFERENCE_ORIGINS),
        "optional_transfer_origin": OLDER_ORIGIN,
        "coordinate_free_b3_schema_width": len(COORDINATE_FREE_B3_FEATURE_COLUMNS),
        "historical_coordinate_bearing_b3_width": len(ORIGINAL_B3_FEATURE_COLUMNS),
        "coordinate_policy": "coordinates are metadata-only for causal indexing/aggregation and are not fitted inputs",
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
        write_coordinate_free_b3_audit(args.output_dir / "b3c_feature_audit.json")
        _atomic_json(args.output_dir / "external_access.json", external_access)
        manifest.update({
            "prefit_checks": prefit,
            "external_access": external_access,
            "external_candidate": {
                "status": "blocked",
                "feature_count": len(GLDAS_EXTERNAL_FEATURES),
                "feature_names": list(GLDAS_EXTERNAL_FEATURES),
                "fit_started": False,
            },
        })
        _atomic_json(args.output_dir / "manifest.json", manifest)
        with (args.output_dir / "events.jsonl").open("a", encoding="utf-8") as log:
            line = json.dumps({"phase": "start", "manifest": manifest, "prefit_checks": prefit}, sort_keys=True, default=str)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
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
                "candidate": "B3C",
                "reference_origins": list(REFERENCE_ORIGINS),
                "optional_transfer_origin": OLDER_ORIGIN,
                "coordinate_free_schema_width": len(COORDINATE_FREE_B3_FEATURE_COLUMNS),
                "external_access_status": external_access["status"],
                "no_test_predictions": True,
            })
            all_results: list[dict[str, object]] = []
            all_oofs: list[pd.DataFrame] = []
            all_train_metrics: list[pd.DataFrame] = []
            origin_details: list[dict[str, object]] = []
            for origin in REFERENCE_ORIGINS:
                details, oof, metrics = _run_origin(
                    train=train,
                    labels=labels,
                    origin=origin,
                    template=template,
                    output_dir=args.output_dir,
                    params=params,
                    resource=resource,
                    log=log,
                )
                origin_details.append(details)
                all_results.append(details["result"])
                all_oofs.append(oof)
                all_train_metrics.append(metrics)

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
            b3c_refs = [
                {
                    "origin": result["origin"],
                    "h1_7_raw_rmse": result["h1_7_raw_rmse"],
                    "h1_7_rows": result["h1_7_rows"],
                    "h1_7_horizons_present": result["h1_7_horizons_present"],
                    "weighted_proxy": result["test_horizon_weighted_validation_proxy"],
                }
                for result in all_results
            ]
            decision = {
                "status": "b3c_reference_established_external_blocked",
                "b3c_reference_origins": b3c_refs,
                "external_candidate": {
                    "status": "blocked_before_bulk_acquisition",
                    "reason": external_access["status"],
                    "fit_started": False,
                },
                "inner_paired_gate": {
                    "status": "not_evaluable_external_access_blocked",
                    "rule": "paired B3-C versus the one GLDAS candidate requires an authenticated official sample and identical ten-feature joins",
                },
                "recent_external_gate": {"status": "not_run", "reason": "inner paired gate not evaluable"},
                "optional_transfer_2009_01": {"status": "not_run", "reason": "external candidate unavailable and no post-decision experiment permitted"},
                "historical_coordinate_bearing_b3": "not_submission_compliant",
                "no_test_predictions": True,
                "submission_written": False,
            }
            _atomic_json(args.output_dir / "decision.json", decision)
            manifest.update({
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "resource_peak": resource.snapshot(),
                "origins_completed": [detail["origin"] for detail in origin_details],
                "fit_count": len(origin_details),
                "model_fit_count": len(origin_details),
                "decision": decision,
                "artifacts": {
                    "manifest": "manifest.json",
                    "resolved_config": "resolved_config.json",
                    "prefit_checks": "prefit_checks.json",
                    "b3c_feature_audit": "b3c_feature_audit.json",
                    "external_access": "external_access.json",
                    "decision": "decision.json",
                    "validation_long": "validation_long.csv.gz",
                    "validation_by_source_month": "validation_by_source_month.csv",
                    "validation_by_horizon": "validation_by_horizon.csv",
                    "validation_overall": "validation_overall.csv",
                    "events": "events.jsonl",
                },
            })
            _atomic_json(args.output_dir / "manifest.json", manifest)
            line = json.dumps({"phase": "completed", "decision": decision, "resource": resource.snapshot()}, sort_keys=True, default=str)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
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
