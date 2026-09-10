from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.compliant_b3 import (
    COORDINATE_FREE_B3_FEATURE_COLUMNS,
    ORIGINAL_B3_FEATURE_COLUMNS,
    assert_coordinate_free_b3_schema,
    build_coordinate_free_b3_matrix,
)
from src.neural_sequence import build_b3_feature_maps, build_b3_matrix
from src.regional_context import HYDRO_COLUMNS


def _panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for month in pd.date_range("2018-01-01", periods=18, freq="MS"):
        for lat in (0.0, 1.0):
            row: dict[str, object] = {
                "sample_id": f"{lat}_{month:%Y%m}",
                "time": month,
                "lat": lat,
                "lon": 0.0,
                "month_sin": np.sin(month.month),
                "month_cos": np.cos(month.month),
            }
            for index, column in enumerate(HYDRO_COLUMNS):
                row[column] = float(month.month + lat + index)
            rows.append(row)
    source = pd.DataFrame(rows)
    structural = source[["sample_id", "time", "lat", "lon"]].copy()
    structural["TWS_t"] = np.arange(len(structural), dtype=float)
    ledger = pd.DataFrame({
        "sample_id": ["0.0_201906", "1.0_201906"],
        "source_date": pd.to_datetime(["2019-06-01", "2019-06-01"]),
        "last_observed_date": pd.to_datetime(["2019-04-01", "2019-05-01"]),
        "last_observed_TWS": [30.0, 31.0],
        "h": [3, 2],
        "lat": [0.0, 1.0],
        "lon": [0.0, 0.0],
    })
    return source, structural, ledger, source


def test_coordinate_free_schema_is_exactly_historical_b3_minus_coordinates():
    assert len(ORIGINAL_B3_FEATURE_COLUMNS) == 446
    assert len(COORDINATE_FREE_B3_FEATURE_COLUMNS) == 444
    assert "lat" not in COORDINATE_FREE_B3_FEATURE_COLUMNS
    assert "lon" not in COORDINATE_FREE_B3_FEATURE_COLUMNS
    assert list(COORDINATE_FREE_B3_FEATURE_COLUMNS) == [
        column for column in ORIGINAL_B3_FEATURE_COLUMNS if column not in {"lat", "lon"}
    ]


def test_coordinate_free_builder_preserves_values_and_drops_only_raw_coordinates():
    source, structural, ledger, _ = _panels()
    maps = build_b3_feature_maps(source)
    historical = build_b3_matrix(ledger, source, structural, maps)
    compliant = build_coordinate_free_b3_matrix(ledger, source, structural, maps)
    assert compliant.columns.tolist() == list(COORDINATE_FREE_B3_FEATURE_COLUMNS)
    np.testing.assert_allclose(
        compliant.to_numpy(),
        historical.drop(columns=["lat", "lon"]).to_numpy(),
        equal_nan=True,
    )
    assert compliant.shape == (2, 444)


def test_coordinate_free_schema_rejects_order_drift_or_location_names():
    with pytest.raises(AssertionError, match="schema/order mismatch"):
        assert_coordinate_free_b3_schema(list(COORDINATE_FREE_B3_FEATURE_COLUMNS[1:]))
    with pytest.raises(AssertionError, match="schema/order mismatch"):
        assert_coordinate_free_b3_schema(
            [*COORDINATE_FREE_B3_FEATURE_COLUMNS[:-1], "location_id"]
        )
