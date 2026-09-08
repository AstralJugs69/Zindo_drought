from __future__ import annotations

import numpy as np
import pandas as pd

from src.covariate_gap_augmentation import (
    build_augmented_b3_matrix,
    derive_test_schedule_patterns,
    event_visibility,
)
from src.neural_sequence import build_b3_feature_maps, build_b3_matrix
from src.regional_context import HYDRO_COLUMNS


def _panels():
    rows = []
    for month in pd.date_range("2019-01-01", periods=18, freq="MS"):
        for lat in (0.0, 1.0):
            row = {"sample_id": f"{lat}_{month:%Y%m}", "time": month, "lat": lat, "lon": 0.0,
                   "month_sin": np.sin(month.month), "month_cos": np.cos(month.month)}
            for index, col in enumerate(HYDRO_COLUMNS):
                row[col] = float(month.month + lat + index)
            rows.append(row)
    source = pd.DataFrame(rows)
    structural = source[["sample_id", "time", "lat", "lon"]].copy()
    structural["TWS_t"] = np.arange(len(structural), dtype=float)
    ledger = pd.DataFrame({
        "sample_id": ["0.0_202006", "1.0_202006"],
        "source_date": pd.to_datetime(["2020-06-01", "2020-06-01"]),
        "last_observed_date": pd.to_datetime(["2020-04-01", "2020-05-01"]),
        "last_observed_TWS": [30.0, 31.0], "h": [3, 2], "lat": [0.0, 1.0], "lon": [0.0, 0.0],
    })
    test = pd.DataFrame({"time": pd.to_datetime(["2022-01-01", "2022-03-01", "2022-06-01"])})
    return source, structural, ledger, test


def test_visibility_is_reproducible_target_independent_and_keeps_anchor():
    _, _, ledger, test = _panels()
    patterns = derive_test_schedule_patterns(test)
    one = event_visibility(ledger, patterns, recipe="D2_mixed", seed=20260909)
    changed = ledger.assign(target=[999.0, -999.0])
    two = event_visibility(changed, patterns, recipe="D2_mixed", seed=20260909)
    for left, right in zip(one, two, strict=True):
        np.testing.assert_array_equal(left, right)
    masks, _, _ = event_visibility(ledger, patterns, recipe="D1_sparse", seed=20260909)
    assert masks[:, 0].all()
    assert masks[0, 2] and masks[1, 1]


def test_augmented_features_recompute_history_without_future_or_removed_cache():
    source, structural, ledger, test = _panels()
    maps = build_b3_feature_maps(source)
    patterns = derive_test_schedule_patterns(test)
    augmented, metadata = build_augmented_b3_matrix(ledger, source, structural, maps, patterns, recipe="D1_sparse", seed=20260909)
    assert metadata["sparse_rows"] == len(ledger)
    assert augmented.columns.tolist() == build_b3_matrix(ledger, source, structural, maps).columns.tolist()
    # A future covariate cannot influence a June event.
    changed = source.copy()
    changed.loc[changed.time == pd.Timestamp("2020-07-01"), HYDRO_COLUMNS[0]] = 1_000_000.0
    later, _ = build_augmented_b3_matrix(ledger, changed, structural, build_b3_feature_maps(changed), patterns, recipe="D1_sparse", seed=20260909)
    np.testing.assert_allclose(augmented.to_numpy(), later.to_numpy(), equal_nan=True)
    # At least one masked historical slot changes an actual trajectory value;
    # this rejects a post-hoc feature blanking/no-op implementation.
    dense = build_b3_matrix(ledger, source, structural, maps)
    assert not np.array_equal(augmented.filter(like="trail").to_numpy(), dense.filter(like="trail").to_numpy(), equal_nan=True)
