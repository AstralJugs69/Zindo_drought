import numpy as np
import pandas as pd

from src.neighbor_state import NEIGHBOR_FEATURE_COLUMNS, build_neighbor_state_index


def _panel() -> pd.DataFrame:
    rows = []
    locations = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0), (0.0, 1.5), (0.0, 2.0),
                 (0.0, 2.5), (0.0, 3.0), (0.0, 3.5), (0.0, 4.0)]
    for month, values in [("2020-01-01", np.arange(9, dtype=float)),
                          ("2020-02-01", np.arange(9, dtype=float) + 10),
                          ("2020-03-01", np.arange(9, dtype=float) + 20),
                          ("2020-04-01", np.arange(9, dtype=float) + 30)]:
        for (lat, lon), value in zip(locations, values):
            rows.append({"time": month, "lat": lat, "lon": lon, "TWS_t": value})
    return pd.DataFrame(rows)


def _event(source: str = "2020-04-01", anchor: str = "2020-02-01") -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": ["e"], "source_date": [source], "last_observed_date": [anchor],
        "lat": [0.0], "lon": [0.0], "last_observed_TWS": [10.0],
    })


def test_cutoff_is_focal_anchor_and_future_changes_are_invariant():
    panel = _panel()
    index = build_neighbor_state_index(panel)
    baseline = index.build_features(_event()).features
    mutated = panel.copy()
    future = pd.to_datetime(mutated["time"]) >= pd.Timestamp("2020-03-01")
    mutated.loc[future, "TWS_t"] += 10000.0
    changed = build_neighbor_state_index(mutated).build_features(_event()).features
    assert np.allclose(baseline.to_numpy(), changed.to_numpy(), equal_nan=True)
    assert baseline.loc[0, "neighbor_tws_mean"] == 14.5
    assert baseline.loc[0, "neighbor_tws_median_age"] == 2.0


def test_masked_neighbor_before_cutoff_is_not_recovered():
    panel = _panel()
    panel.loc[(panel.lat == 0.0) & (panel.lon == 1.0) & (panel.time == "2020-02-01"), "TWS_t"] = np.nan
    result = build_neighbor_state_index(panel).build_features(_event()).features
    # The missing February value falls back to January (age three at the April event).
    assert result.loc[0, "neighbor_tws_count"] == 8.0
    assert result.loc[0, "neighbor_tws_median_age"] == 2.0


def test_self_exclusion_and_longitude_wrap_geometry():
    rows = []
    locations = [(0.0, 179.0), (0.0, -179.0), (0.0, 170.0), (0.0, -170.0),
                 (1.0, 179.0), (-1.0, -179.0), (1.0, 170.0), (-1.0, -170.0), (2.0, 0.0)]
    for lat, lon in locations:
        rows.append({"time": "2020-01-01", "lat": lat, "lon": lon, "TWS_t": 1.0})
    index = build_neighbor_state_index(pd.DataFrame(rows))
    zero_id = int(index.locations.index[(index.locations.lat == 0.0) & (index.locations.lon == 179.0)][0])
    assert zero_id not in index.neighbor_ids[zero_id]
    assert np.isfinite(index.neighbor_distances_km[zero_id, 0])
    assert index.neighbor_distances_km[zero_id, 0] < 300.0


def test_unsupported_state_keeps_count_but_missing_summary_features():
    panel = _panel()
    panel.loc[panel.lon.isin([0.5, 1.0, 1.5, 2.0, 2.5]), "TWS_t"] = np.nan
    result = build_neighbor_state_index(panel).build_features(_event()).features
    assert result.columns.tolist() == NEIGHBOR_FEATURE_COLUMNS
    assert result.loc[0, "neighbor_tws_count"] == 3.0
    assert result.loc[0, ["neighbor_tws_mean", "neighbor_tws_minus_focal", "neighbor_tws_median_age"]].isna().all()


def test_identical_event_and_panel_replay_are_deterministic():
    panel = _panel()
    events = _event().copy()
    left = build_neighbor_state_index(panel).build_features(events).features
    right = build_neighbor_state_index(panel.copy()).build_features(events.copy()).features
    pd.testing.assert_frame_equal(left, right)
