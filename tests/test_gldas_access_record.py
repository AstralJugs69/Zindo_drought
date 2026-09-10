from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_submission_compliant_b3_experiment import (
    HISTORICAL_EXTERNAL_ACCESS_FAILURE,
    _external_access_record,
    _load_external_access_measurement,
)


def test_external_access_record_uses_measurement_and_keeps_history() -> None:
    measured = {
        "status": "blocked_http_401_netrc_get",
        "http_status": 401,
        "final_host": "urs.earthdata.nasa.gov",
        "content_type": "text/html; charset=utf-8",
        "sanitized_error": "Unauthorized",
        "netrc_present": True,
        "netrc_permissions_ok": True,
        "netrc_machine_present": True,
        "sample_file_retained": False,
        "netcdf_verification": {"status": "not_run_http_error"},
    }

    record = _external_access_record(measured)

    assert record["status"] == "blocked_http_401_netrc_get"
    assert record["sample_request"]["method"] == "GET"
    assert record["sample_request"]["http_status"] == 401
    assert record["sample_request"]["content_type"] == "text/html; charset=utf-8"
    assert record["credentials_checked_without_exposure"]["netrc_present"] is True
    assert record["netcdf_verification"]["status"] == "not_run_http_error"
    assert record["historical_failure"] == HISTORICAL_EXTERNAL_ACCESS_FAILURE
    assert record["bulk_acquisition_started"] is False
    assert record["external_fit_started"] is False


def test_measurement_loader_rejects_secret_like_fields(tmp_path) -> None:
    path = tmp_path / "measurement.json"
    path.write_text(json.dumps({"status": "ok", "cookie": "must-not-persist"}), encoding="utf-8")
    with pytest.raises(ValueError, match="prohibited secret-like fields"):
        _load_external_access_measurement(path)
