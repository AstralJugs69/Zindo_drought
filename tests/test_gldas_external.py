import numpy as np
import pandas as pd
import pytest

from src.gldas_external import (
    GLDAS_EXTERNAL_FEATURE_COLUMNS,
    GLDAS_PRODUCT,
    GLDAS_RAW_FIELDS,
    GLDASRawStore,
    build_gldas_external_features,
    run_gldas_external_prefit_checks,
)


def _store() -> GLDASRawStore:
    period_key = np.array([202001, 202002, 202003], dtype=np.int32)
    values = {
        field: np.array([1.0 + index, 11.0 + index, 21.0 + index], dtype=np.float32)
        for index, field in enumerate(GLDAS_RAW_FIELDS)
    }
    return GLDASRawStore(
        period_key=period_key,
        grid_i=np.zeros(3, dtype=np.int16),
        grid_j=np.zeros(3, dtype=np.int16),
        values=values,
        grid_lat=np.array([0.125], dtype=np.float32),
        grid_lon=np.array([10.125], dtype=np.float32),
        metadata={"product": GLDAS_PRODUCT},
    )


def test_prefit_contract_checks_pass():
    result = run_gldas_external_prefit_checks()
    assert result["status"] == "passed"
    assert result["feature_count"] == 10


def test_exact_schema_and_calendar_deltas():
    ledger = pd.DataFrame(
        {
            "sample_id": ["a"],
            "source_date": pd.to_datetime(["2020-03-17"]),
            "last_observed_date": pd.to_datetime(["2020-02-01"]),
            "lat": [0.125],
            "lon": [10.125],
        }
    )
    features = build_gldas_external_features(ledger, _store())
    assert features.columns.tolist() == list(GLDAS_EXTERNAL_FEATURE_COLUMNS)
    assert features.dtypes.astype(str).eq("float32").all()
    assert features.loc[0, "gldas_storage_sum_t"] == pytest.approx(sum(21.0 + i for i in range(6)))
    assert features.loc[0, "gldas_storage_sum_anchor"] == pytest.approx(sum(11.0 + i for i in range(6)))
    assert features.loc[0, "gldas_storage_sum_gap"] == pytest.approx(60.0)
    assert features.loc[0, "gldas_storage_sum_prev_month_delta"] == pytest.approx(60.0)


def test_target_is_rejected_and_future_anchor_is_rejected():
    ledger = pd.DataFrame(
        {
            "source_date": pd.to_datetime(["2020-02-01"]),
            "last_observed_date": pd.to_datetime(["2020-03-01"]),
            "lat": [0.125],
            "lon": [10.125],
        }
    )
    with pytest.raises(AssertionError, match="anchor month"):
        build_gldas_external_features(ledger, _store())
    with pytest.raises(AssertionError, match="target-blind"):
        build_gldas_external_features(ledger.assign(target=0.0), _store())


def test_missing_components_do_not_form_partial_sums():
    store = _store()
    values = {field: array.copy() for field, array in store.values.items()}
    values[GLDAS_RAW_FIELDS[0]][2] = np.nan
    missing_store = GLDASRawStore(
        period_key=store.period_key,
        grid_i=store.grid_i,
        grid_j=store.grid_j,
        values=values,
        grid_lat=store.grid_lat,
        grid_lon=store.grid_lon,
        metadata={"product": GLDAS_PRODUCT},
    )
    ledger = pd.DataFrame(
        {
            "source_date": pd.to_datetime(["2020-03-01"]),
            "last_observed_date": pd.to_datetime(["2020-02-01"]),
            "lat": [0.125],
            "lon": [10.125],
        }
    )
    features = build_gldas_external_features(ledger, missing_store)
    assert np.isnan(features.loc[0, "gldas_soil_moisture_0_10cm_t"])
    assert np.isnan(features.loc[0, "gldas_storage_sum_t"])
    assert np.isnan(features.loc[0, "gldas_storage_sum_gap"])
