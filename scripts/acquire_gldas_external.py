"""Acquire only the GLDAS cells/months required by the frozen replay.

The downloader uses the same Earthdata ``.netrc``/HTTPS-only client as the
authenticated sample probe.  Credentials and cookies stay in process memory;
only sanitized HTTP metadata and extracted numeric arrays are persisted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import netrc as netrc_module
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import (
    HTTPBasicAuthHandler,
    HTTPPasswordMgrWithDefaultRealm,
    HTTPCookieProcessor,
    Request,
    build_opener,
)
from http.cookiejar import CookieJar

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_gldas_access import (  # noqa: E402
    DATA_HOST,
    EXPECTED_PRODUCT,
    EXPECTED_VARIABLES,
    HTTPSOnlyRedirectHandler,
    URS_HOST,
    _units_ok,
    _verify_netcdf,
)
from src.gldas_external import (  # noqa: E402
    GLDAS_GRID_TOLERANCE_DEGREES,
    GLDAS_PRODUCT,
    GLDAS_RAW_FIELDS,
    GLDAS_RAW_VARIABLES,
    GLDASRawStore,
    build_external_request_plan,
)
from src.validation import build_test_mask_template  # noqa: E402
from src.regional_context import HYDRO_COLUMNS  # noqa: E402


SAMPLE_DIR = Path("/home/milli/gldas_sample/auth_client_check")
SAMPLE_FILE_NAME = "GLDAS_NOAH025_M.A201404.021.nc4"
TEST_GEOMETRY_COLUMNS = ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]
TRAIN_COLUMNS = [
    "sample_id", "time", "lat", "lon", "TWS_t", "month_sin", "month_cos",
    *HYDRO_COLUMNS, "target",
]


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


def _safe_error(reason: object) -> str:
    text = str(reason).replace("\r", " ").replace("\n", " ").strip()
    lowered = text.lower()
    if "https://" in lowered or "http://" in lowered or any(
        token in lowered for token in ("password", "passwd", "token", "cookie", "authorization")
    ):
        return "redacted_error"
    return text[:160] or "unknown_error"


def _build_earthdata_opener(netrc_path: Path):
    """Build the known-good Basic-auth client without exposing its values."""
    try:
        parsed = netrc_module.netrc(str(netrc_path))
        auth = parsed.authenticators(URS_HOST)
    except (OSError, netrc_module.NetrcParseError) as exc:
        raise RuntimeError("earthdata_netrc_unavailable") from exc
    if auth is None or auth[0] is None or auth[2] is None:
        raise RuntimeError("earthdata_netrc_entry_unavailable")
    if (netrc_path.stat().st_mode & 0o077) != 0:
        raise RuntimeError("earthdata_netrc_permissions_too_open")
    login, _account, password = auth
    manager = HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, f"https://{URS_HOST}/", login, password)
    return build_opener(
        HTTPSOnlyRedirectHandler,
        HTTPBasicAuthHandler(manager),
        HTTPCookieProcessor(CookieJar()),
    )


def _load_sample_grid(sample_file: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    verification = _verify_netcdf(sample_file)
    if verification.get("product") != EXPECTED_PRODUCT or verification.get("sample_month") != "2014-04":
        raise RuntimeError("sample_product_or_month_mismatch")
    try:
        import netCDF4  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("netcdf4_unavailable") from exc
    with netCDF4.Dataset(sample_file, mode="r") as dataset:
        lat = np.asarray(dataset.variables["lat"][:], dtype=np.float32)
        lon = np.asarray(dataset.variables["lon"][:], dtype=np.float32)
    if lat.shape != (600,) or lon.shape != (1440,):
        raise RuntimeError("sample_grid_shape_mismatch")
    if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)):
        raise RuntimeError("sample_grid_not_ascending")
    return lat, lon, verification


def _month_url(month: str) -> str:
    year, month_number = month.split("-")
    return (
        f"https://{DATA_HOST}/data/GLDAS/{GLDAS_PRODUCT}/{year}/"
        f"GLDAS_NOAH025_M.A{year}{month_number}.021.nc4"
    )


def _extract_month(
    path: Path,
    *,
    expected_month: str,
    grid_i: np.ndarray,
    grid_j: np.ndarray,
    sample_lat: np.ndarray,
    sample_lon: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Verify one NetCDF and extract only the requested grid cells."""
    try:
        import netCDF4  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("netcdf4_unavailable") from exc
    with netCDF4.Dataset(path, mode="r") as dataset:
        dimensions = {name: int(len(value)) for name, value in dataset.dimensions.items()}
        if dimensions.get("lat") != len(sample_lat) or dimensions.get("lon") != len(sample_lon):
            raise ValueError("monthly_grid_shape_mismatch")
        lat = np.asarray(dataset.variables["lat"][:], dtype=np.float32)
        lon = np.asarray(dataset.variables["lon"][:], dtype=np.float32)
        if not (np.array_equal(lat, sample_lat) and np.array_equal(lon, sample_lon)):
            raise ValueError("monthly_grid_coordinates_changed")
        time_variable = dataset.variables.get("time")
        if time_variable is None:
            raise ValueError("monthly_time_variable_missing")
        calendar = str(getattr(time_variable, "calendar", "standard"))
        import netCDF4  # type: ignore

        dates = netCDF4.num2date(
            time_variable[:],
            units=str(getattr(time_variable, "units")),
            calendar=calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=False,
        )
        date_months = [f"{int(value.year):04d}-{int(value.month):02d}" for value in (dates.flat if hasattr(dates, "flat") else dates)]
        if expected_month not in date_months:
            raise ValueError("monthly_date_mismatch")
        extracted: dict[str, np.ndarray] = {}
        missing_counts: dict[str, int] = {}
        for field, variable_name in GLDAS_RAW_VARIABLES.items():
            if variable_name not in dataset.variables:
                raise ValueError(f"monthly_variable_missing_{field}")
            variable = dataset.variables[variable_name]
            units = str(getattr(variable, "units", ""))
            if not _units_ok(units):
                raise ValueError(f"monthly_units_mismatch_{field}")
            values = np.ma.filled(variable[0, :, :], np.nan).astype(np.float32, copy=False)
            values = values[grid_i, grid_j].astype(np.float32, copy=False)
            values[~np.isfinite(values)] = np.nan
            extracted[field] = values
            missing_counts[field] = int(np.isnan(values).sum())
    return extracted, {
        "month": expected_month,
        "date_months": date_months,
        "variable_units": dict(EXPECTED_VARIABLES),
        "missing_counts": missing_counts,
        "rows": int(len(grid_i)),
    }


def _write_month(path: Path, *, month: str, grid_i: np.ndarray, grid_j: np.ndarray, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    np.savez_compressed(
        temporary,
        period_key=np.full(len(grid_i), int(month.replace("-", "")), dtype=np.int32),
        grid_i=grid_i.astype(np.int16),
        grid_j=grid_j.astype(np.int16),
        **{field: values[field].astype(np.float32) for field in GLDAS_RAW_FIELDS},
    )
    Path(str(temporary) + ".npz").replace(path)


def _load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "Train.csv", usecols=TRAIN_COLUMNS)
    geometry = pd.read_csv(data_dir / "Test.csv", usecols=TEST_GEOMETRY_COLUMNS)
    if train["sample_id"].duplicated().any() or train[["lat", "lon", "time"]].duplicated().any():
        raise ValueError("Train IDs and location-month keys must be unique")
    return train, geometry


def acquire(
    *,
    data_dir: Path,
    output_dir: Path,
    sample_dir: Path,
    netrc_path: Path,
    max_workers: int = 1,
) -> dict[str, object]:
    if max_workers != 1:
        raise ValueError("only one bounded downloader worker is permitted for this memory-safe acquisition")
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir = output_dir / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    sample_file = sample_dir / SAMPLE_FILE_NAME
    if not sample_file.is_file():
        raise FileNotFoundError(sample_file)
    sample_lat, sample_lon, sample_verification = _load_sample_grid(sample_file)
    product_gate_path = sample_dir / "product_gate.json"
    if not product_gate_path.is_file():
        raise FileNotFoundError(product_gate_path)
    product_gate = json.loads(product_gate_path.read_text(encoding="utf-8"))
    if product_gate.get("status") != "pass_conditional_historical_proxy":
        raise RuntimeError("product_gate_not_passed")

    train, geometry = _load_inputs(data_dir)
    template = build_test_mask_template(geometry)
    plan = build_external_request_plan(train, template)
    _atomic_json(output_dir / "request_plan.json", plan)
    coordinates = np.asarray(plan["coordinates"], dtype=np.float64)
    grid_i, lat_distance = _coordinate_indices(coordinates[:, 0], sample_lat)
    grid_j, lon_distance = _coordinate_indices(coordinates[:, 1], sample_lon)
    max_distance = np.maximum(lat_distance, lon_distance)
    if np.any(max_distance > GLDAS_GRID_TOLERANCE_DEGREES):
        raise RuntimeError("planned_coordinate_outside_grid_tolerance")
    # The request plan is deterministic and sorted; preserve that order in every
    # monthly file so the final store is reproducible.
    grid_i = grid_i.astype(np.int16)
    grid_j = grid_j.astype(np.int16)
    periods = list(plan["periods"])
    opener = _build_earthdata_opener(netrc_path)
    prior_months: dict[str, object] = {}
    prior_manifest_path = output_dir / "manifest.json"
    if prior_manifest_path.is_file():
        try:
            prior_payload = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
            if isinstance(prior_payload.get("months"), dict):
                prior_months = dict(prior_payload["months"])
        except (OSError, json.JSONDecodeError):
            prior_months = {}
    manifest: dict[str, object] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "product": GLDAS_PRODUCT,
        "product_gate": product_gate,
        "sample_verification": sample_verification,
        "data_dir": str(data_dir),
        "request_plan": "request_plan.json",
        "period_count": int(len(periods)),
        "coordinate_count": int(len(coordinates)),
        "coordinate_sha256": plan["coordinate_sha256"],
        "sampling_method": plan["sampling_method"],
        "grid_shape": [int(len(sample_lat)), int(len(sample_lon))],
        "grid_tie_count": int(np.sum(np.isclose(lat_distance, 0.125, atol=1e-6) | np.isclose(lon_distance, 0.125, atol=1e-6))),
        "bounded_max_workers": int(max_workers),
        "credential_values_recorded": False,
        "cookies_recorded": False,
        "substitute_dataset_used": False,
        "test_labels_read": False,
        "test_predictions_written": False,
        "submission_written": False,
        "months": prior_months,
        "python": sys.version,
        "platform": platform.platform(),
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    try:
        for month in periods:
            month_file = monthly_dir / f"gldas_{month.replace('-', '')}.npz"
            month_record = manifest["months"].get(month, {}) if isinstance(manifest["months"], dict) else {}
            if month_file.is_file() and month_record.get("status") == "success":
                continue
            url = _month_url(month)
            temporary_path: Path | None = None
            try:
                request = Request(url, method="GET", headers={"Accept": "application/x-netcdf, application/octet-stream"})
                with opener.open(request, timeout=240) as response:
                    with tempfile.NamedTemporaryFile(
                        mode="wb", prefix=".gldas_month_", suffix=".nc4", dir=output_dir, delete=False
                    ) as handle:
                        temporary_path = Path(handle.name)
                        while True:
                            block = response.read(1 << 20)
                            if not block:
                                break
                            handle.write(block)
                    values, extraction = _extract_month(
                        temporary_path,
                        expected_month=month,
                        grid_i=grid_i,
                        grid_j=grid_j,
                        sample_lat=sample_lat,
                        sample_lon=sample_lon,
                    )
                    _write_month(month_file, month=month, grid_i=grid_i, grid_j=grid_j, values=values)
                    record = {
                        "status": "success",
                        "http_status": int(getattr(response, "status", 200)),
                        "final_host": urlparse(response.geturl()).hostname,
                        "content_type": str(response.headers.get("Content-Type", ""))[:120],
                        "bytes": int(temporary_path.stat().st_size),
                        "file": month_file.name,
                        "file_sha256": _sha256(month_file),
                        "extraction": extraction,
                    }
            except HTTPError as exc:
                if int(exc.code) == 404:
                    record = {
                        "status": "missing_http_404",
                        "http_status": 404,
                        "final_host": urlparse(str(getattr(exc, "url", ""))).hostname,
                        "content_type": str(exc.headers.get("Content-Type", ""))[:120] if exc.headers else "",
                        "sanitized_error": "requested_month_not_found",
                    }
                else:
                    raise RuntimeError(f"gldas_month_http_{int(exc.code)}") from exc
            except (URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
                raise RuntimeError(f"gldas_month_failed_{_safe_error(exc)}") from exc
            finally:
                if temporary_path is not None:
                    try:
                        temporary_path.unlink()
                    except FileNotFoundError:
                        pass
            manifest["months"][month] = record
            _atomic_json(output_dir / "manifest.json", manifest)

        successful = [
            month for month in periods
            if isinstance(manifest["months"].get(month), dict)
            and manifest["months"][month].get("status") == "success"
        ]
        if not successful:
            raise RuntimeError("no_gldas_months_acquired")
        keys: list[np.ndarray] = []
        values_by_field: dict[str, list[np.ndarray]] = {field: [] for field in GLDAS_RAW_FIELDS}
        for month in successful:
            with np.load(monthly_dir / f"gldas_{month.replace('-', '')}.npz", allow_pickle=False) as payload:
                keys.append(np.asarray(payload["period_key"], dtype=np.int32))
                for field in GLDAS_RAW_FIELDS:
                    values_by_field[field].append(np.asarray(payload[field], dtype=np.float32))
        all_keys = np.concatenate(keys)
        all_i = np.tile(grid_i, len(successful))
        all_j = np.tile(grid_j, len(successful))
        all_values = {field: np.concatenate(chunks).astype(np.float32) for field, chunks in values_by_field.items()}
        store = GLDASRawStore(
            period_key=all_keys,
            grid_i=all_i,
            grid_j=all_j,
            values=all_values,
            grid_lat=sample_lat,
            grid_lon=sample_lon,
            metadata={
                "product": GLDAS_PRODUCT,
                "sampling_method": plan["sampling_method"],
                "requested_periods": periods,
                "successful_periods": successful,
                "coordinate_sha256": plan["coordinate_sha256"],
                "source_variables": dict(GLDAS_RAW_VARIABLES),
                "units": "kg m-2",
            },
        )
        store.save(output_dir / "raw_store.npz", output_dir / "store_metadata.json")
        manifest.update({
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "successful_periods": successful,
            "missing_periods": [month for month in periods if month not in successful],
            "raw_store": "raw_store.npz",
            "raw_store_metadata": "store_metadata.json",
            "raw_store_rows": store.row_count,
            "raw_store_key_sha256": store.store_key_sha256,
            "raw_fields": list(GLDAS_RAW_FIELDS),
        })
        _atomic_json(output_dir / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "sanitized_error": _safe_error(exc),
        })
        _atomic_json(output_dir / "manifest.json", manifest)
        raise


def _coordinate_indices(values: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    right = np.searchsorted(axis, values, side="left")
    lo = np.clip(right - 1, 0, len(axis) - 1)
    hi = np.clip(right, 0, len(axis) - 1)
    lo_distance = np.abs(values - axis[lo])
    hi_distance = np.abs(axis[hi] - values)
    choose_lo = lo_distance <= hi_distance
    chosen = np.where(choose_lo, lo, hi).astype(np.int32)
    return chosen, np.minimum(lo_distance, hi_distance).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, default=SAMPLE_DIR)
    parser.add_argument("--netrc-path", type=Path, default=Path.home() / ".netrc")
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()
    result = acquire(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_dir=args.sample_dir,
        netrc_path=args.netrc_path,
        max_workers=args.max_workers,
    )
    summary = {
        "status": result["status"],
        "product": result["product"],
        "period_count": result["period_count"],
        "successful_period_count": len(result.get("successful_periods", [])),
        "missing_period_count": len(result.get("missing_periods", [])),
        "raw_store_rows": result.get("raw_store_rows"),
        "credential_values_recorded": False,
        "cookies_recorded": False,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
