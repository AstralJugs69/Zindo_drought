"""Probe the official GLDAS sample with Earthdata credentials, safely.

The probe performs a real GET (not HEAD), follows only HTTPS redirects, and
keeps the cookie jar in memory.  Its JSON output is deliberately limited to
sanitized metadata; it never writes credential values, cookies, response
bodies, or authorization headers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import netrc as netrc_module
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import (
    HTTPBasicAuthHandler,
    HTTPPasswordMgrWithDefaultRealm,
    HTTPCookieProcessor,
    HTTPRedirectHandler,
    Request,
    build_opener,
)
from http.cookiejar import CookieJar


SAMPLE_URL = (
    "https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_M.2.1/2014/"
    "GLDAS_NOAH025_M.A201404.021.nc4"
)
DATA_HOST = "hydro1.gesdisc.eosdis.nasa.gov"
URS_HOST = "urs.earthdata.nasa.gov"
EXPECTED_PRODUCT = "GLDAS_NOAH025_M.2.1"
EXPECTED_DATE = "2014-04"
EXPECTED_VARIABLES = {
    "SoilMoi0_10cm_inst": "kg m-2",
    "SoilMoi10_40cm_inst": "kg m-2",
    "SoilMoi40_100cm_inst": "kg m-2",
    "SoilMoi100_200cm_inst": "kg m-2",
    "SWE_inst": "kg m-2",
    "CanopInt_inst": "kg m-2",
}


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_error(reason: object) -> str:
    """Return a short reason that cannot accidentally contain a URL/secret."""
    text = str(reason).replace("\r", " ").replace("\n", " ").strip()
    lowered = text.lower()
    if (
        "://" in text
        or "=" in text
        or any(token in lowered for token in ("password", "passwd", "token", "cookie", "authorization"))
    ):
        return "redacted_error"
    return text[:160] or "unknown_error"


def _https_only_redirect(req: Request, newurl: str) -> None:
    old_scheme = urlparse(req.full_url).scheme.lower()
    new_scheme = urlparse(newurl).scheme.lower()
    if old_scheme != "https" or new_scheme != "https":
        raise URLError("insecure_redirect_rejected")


class HTTPSOnlyRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _https_only_redirect(req, newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _units_ok(value: object) -> bool:
    normalized = "".join(str(value).lower().split()).replace("**", "^")
    return normalized in {"kgm-2", "kgm^-2", "kg/m-2", "kg/m^2"}


def _verify_netcdf(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        magic = handle.read(8)
    if magic != b"\x89HDF\r\n\x1a\n":
        raise ValueError("sample_not_hdf5_netcdf")

    try:
        import netCDF4  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("netcdf4_unavailable") from exc

    try:
        dataset = netCDF4.Dataset(path, mode="r")
    except Exception as exc:
        raise ValueError("sample_not_readable_netcdf") from exc
    try:
        variable_units: dict[str, str] = {}
        variable_shapes: dict[str, list[int]] = {}
        for name, expected_units in EXPECTED_VARIABLES.items():
            if name not in dataset.variables:
                raise ValueError(f"missing_variable_{name}")
            variable = dataset.variables[name]
            actual_units = str(getattr(variable, "units", ""))
            if not _units_ok(actual_units):
                raise ValueError(f"unexpected_units_{name}")
            variable_units[name] = actual_units
            variable_shapes[name] = [int(size) for size in variable.shape]

        dimensions = {name: int(len(dimension)) for name, dimension in dataset.dimensions.items()}
        if not {"time", "lat", "lon"}.issubset(dimensions):
            raise ValueError("missing_expected_dimensions")
        if dimensions["lat"] != 600 or dimensions["lon"] != 1440:
            raise ValueError("unexpected_grid_shape")

        time_variable = dataset.variables.get("time")
        if time_variable is None:
            raise ValueError("missing_time_variable")
        calendar = str(getattr(time_variable, "calendar", "standard"))
        dates = netCDF4.num2date(
            time_variable[:],
            units=str(getattr(time_variable, "units")),
            calendar=calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=False,
        )
        date_values = list(dates.flat) if hasattr(dates, "flat") else list(dates)
        date_months = [f"{int(value.year):04d}-{int(value.month):02d}" for value in date_values]
        if EXPECTED_DATE not in date_months:
            raise ValueError("sample_date_not_april_2014")

        marker_names: list[str] = []
        for attribute in dataset.ncattrs():
            value = str(getattr(dataset, attribute, ""))
            marker = value.lower()
            if "gldas" in marker or "2.1" in marker:
                marker_names.append(attribute)
        if not marker_names:
            raise ValueError("product_marker_missing")

        return {
            "status": "verified_netcdf",
            "file_format": str(getattr(dataset, "file_format", "unknown")),
            "bytes": int(path.stat().st_size),
            "magic": "HDF5_NETCDF4",
            "product": EXPECTED_PRODUCT,
            "sample_month": EXPECTED_DATE,
            "dimensions": dimensions,
            "time_months": date_months,
            "variable_units": variable_units,
            "variable_shapes": variable_shapes,
            "product_marker_attribute_names": sorted(set(marker_names)),
        }
    finally:
        dataset.close()


def measure_sample(*, output_dir: Path, sample_url: str = SAMPLE_URL) -> dict[str, Any]:
    """GET and verify one official sample, returning only sanitized metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(sample_url)
    if parsed.scheme.lower() != "https":
        raise ValueError("sample_url_must_use_https")

    netrc_path = Path.home() / ".netrc"
    netrc_present = netrc_path.is_file()
    netrc_mode_ok = False
    machine_present = False
    auth = None
    if netrc_present:
        try:
            netrc_mode_ok = (netrc_path.stat().st_mode & 0o077) == 0
            parsed_netrc = netrc_module.netrc(str(netrc_path))
            auth = parsed_netrc.authenticators(URS_HOST)
            machine_present = auth is not None and all(value is not None for value in auth)
        except (OSError, netrc_module.NetrcParseError):
            auth = None

    base: dict[str, Any] = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "method": "GET",
        "sample_url": sample_url,
        "sample_host": parsed.hostname,
        "redirect_policy": "HTTPS-only; TLS certificate verification left at urllib defaults",
        "credential_source": "~/.netrc",
        "netrc_present": bool(netrc_present),
        "netrc_permissions_ok": bool(netrc_mode_ok),
        "netrc_machine_present": bool(machine_present),
        "credential_values_recorded": False,
        "cookies_recorded": False,
        "product_expected": EXPECTED_PRODUCT,
        "sample_month_expected": EXPECTED_DATE,
        "variables_expected": EXPECTED_VARIABLES,
        "bulk_acquisition_started": False,
        "substitute_dataset_used": False,
        "external_fit_started": False,
    }
    if not machine_present:
        base.update({
            "status": "blocked_missing_or_unusable_netrc",
            "http_status": None,
            "sanitized_error": "earthdata_netrc_entry_unavailable",
            "sample_file_retained": False,
            "netcdf_verification": {"status": "not_run"},
        })
        _atomic_json(output_dir / "access_measurement.json", base)
        return base

    login, _account, password = auth  # values stay in local memory only
    password_manager = HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, f"https://{URS_HOST}/", login, password)
    opener = build_opener(
        HTTPSOnlyRedirectHandler,
        HTTPBasicAuthHandler(password_manager),
        HTTPCookieProcessor(CookieJar()),
    )
    request = Request(sample_url, method="GET", headers={"Accept": "application/x-netcdf, application/octet-stream"})
    temporary_path: Path | None = None
    try:
        with opener.open(request, timeout=180) as response:
            final_host = urlparse(response.geturl()).hostname
            content_type = str(response.headers.get("Content-Type", ""))[:120]
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".gldas_sample_", suffix=".nc4", dir=output_dir, delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    handle.write(block)
            verification = _verify_netcdf(temporary_path)
            final_path = output_dir / "GLDAS_NOAH025_M.A201404.021.auth.nc4"
            os.replace(temporary_path, final_path)
            temporary_path = None
            base.update({
                "status": "success_netcdf_verified",
                "http_status": int(getattr(response, "status", 200)),
                "final_host": final_host,
                "content_type": content_type,
                "sample_file_retained": True,
                "sample_file": final_path.name,
                "netcdf_verification": verification,
            })
    except HTTPError as exc:
        base.update({
            "status": f"blocked_http_{int(exc.code)}_netrc_get",
            "http_status": int(exc.code),
            "final_host": urlparse(str(getattr(exc, "url", ""))).hostname,
            "content_type": str(exc.headers.get("Content-Type", ""))[:120] if exc.headers else "",
            "sanitized_error": _safe_error(getattr(exc, "reason", "http_error")),
            "sample_file_retained": False,
            "netcdf_verification": {"status": "not_run_http_error"},
        })
    except (URLError, TimeoutError, OSError) as exc:
        base.update({
            "status": "blocked_transport_or_redirect",
            "http_status": None,
            "sanitized_error": _safe_error(getattr(exc, "reason", exc)),
            "sample_file_retained": False,
            "netcdf_verification": {"status": "not_run_transport_error"},
        })
    except (RuntimeError, ValueError) as exc:
        base.update({
            "status": "blocked_sample_verification",
            "http_status": 200,
            "sanitized_error": _safe_error(exc),
            "sample_file_retained": False,
            "netcdf_verification": {"status": "failed", "error": _safe_error(exc)},
        })
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    _atomic_json(output_dir / "access_measurement.json", base)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-url", default=SAMPLE_URL)
    args = parser.parse_args()
    result = measure_sample(output_dir=args.output_dir, sample_url=args.sample_url)
    # This is the complete console report; it contains no response body or
    # credential/cookie material.
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
