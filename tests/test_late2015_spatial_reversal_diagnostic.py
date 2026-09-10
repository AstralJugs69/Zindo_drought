import numpy as np
import pandas as pd

from scripts.run_late2015_spatial_reversal_diagnostic import (
    build_exact_calendar_reversal,
    build_legal_anchor_state,
    build_neighbor_index,
    cell_sse_decomposition,
    neighbor_state_for_rows,
)


def test_reversal_join_does_not_bridge_calendar_gap():
    frame = pd.DataFrame(
        {
            "location_id": [0, 0, 0],
            "source_ord": [100, 102, 103],
            "d": [1.0, 2.0, -3.0],
        }
    )
    pairs = build_exact_calendar_reversal(frame)
    assert pairs[["source_ord", "target_ord"]].to_records(index=False).tolist() == [(102, 103)]
    assert pairs["d"].tolist() == [2.0]


def test_neighbor_index_excludes_self_and_wraps_longitude():
    lat = np.array([0.0, 0.0, 1.0, 1.0])
    lon = np.array([179.9, -179.9, 0.0, 1.0])
    indices, distances = build_neighbor_index(lat, lon, k=1, radius_km=500.0)
    assert indices[0, 0] == 1
    assert distances[0, 0] < 30.0
    assert not np.any(indices == np.arange(len(lat), dtype=np.int32)[:, None])


def test_unavailable_neighbors_are_masked_and_minimum_support_is_enforced():
    indices = np.array([[1, 2, -1], [0, 2, -1], [0, 1, -1]], dtype=np.int32)
    values = np.array([10.0, np.nan, 30.0])
    means, counts, _ = neighbor_state_for_rows(
        np.array([0, 1, 2]), values, n_locations=3,
        neighbor_index=indices, min_neighbors=2,
    )
    assert counts.tolist() == [1, 2, 1]
    assert np.isnan(means[[0, 2]]).all()
    assert means[1] == 20.0


def test_cell_sse_conserves_unequal_cell_counts():
    result = cell_sse_decomposition(
        np.array([1.0, 1.0, 1.0, 3.0]), np.array([0, 0, 0, 1])
    )
    assert result["rows"] == 4
    assert result["sse_total"] == 12.0
    assert result["sse_between_cell_mean"] == 12.0
    assert result["sse_within_cell"] == 0.0
    assert result["sse_conservation_abs_error"] == 0.0


def test_legal_anchor_state_ignores_future_tws_changes():
    train = pd.DataFrame(
        {
            "location_id": [0, 0, 0, 1, 1],
            "source_ord": [10, 11, 12, 10, 13],
            "TWS_t": [1.0, 2.0, 3.0, 4.0, 99.0],
        }
    )
    base_tws, base_ord = build_legal_anchor_state(train, origin_ord=11)
    changed = train.copy()
    changed.loc[changed["source_ord"] > 11, "TWS_t"] += 1000.0
    new_tws, new_ord = build_legal_anchor_state(changed, origin_ord=11)
    np.testing.assert_array_equal(base_tws, new_tws)
    np.testing.assert_array_equal(base_ord, new_ord)
