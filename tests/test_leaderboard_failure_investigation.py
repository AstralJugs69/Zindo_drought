"""Unit guards for the read-only leaderboard root-cause diagnostics."""

import numpy as np
import pandas as pd

from scripts.run_leaderboard_failure_investigation import (
    _aggregate_error_frame,
    _finalize_error_metrics,
    _join_identity,
    _metric_arrays,
    _next_calendar_alignment,
    _tag_training_exposure,
    _weighted_rmse,
)
from src.availability import build_replay_observation_view


def test_sse_decomposition_is_explicit():
    result = _metric_arrays(np.array([0.0, 2.0, 4.0]), np.array([1.0, 1.0, 5.0]))
    assert result["rows"] == 3
    assert result["decomposition_error"] is not None
    assert abs(float(result["decomposition_error"])) < 1e-12


def test_next_calendar_alignment_does_not_row_shift():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "time": ["2015-01-01", "2015-03-01", "2015-02-01"],
            "lat": [1.0, 1.0, 1.0],
            "lon": [2.0, 2.0, 2.0],
            "TWS_t": [10.0, 30.0, 20.0],
            "target": [20.0, 31.0, 30.0],
        }
    )
    joined, evidence = _next_calendar_alignment(frame)
    # Jan's target is the February row and March has no April row; the input
    # order is intentionally non-chronological.
    jan = joined.loc[joined["sample_id"] == "a"].iloc[0]
    assert jan["next_calendar_TWS"] == 20.0
    assert evidence["matched_rows"] == 2
    assert evidence["unmatched_rows"] == 1


def test_join_identity_reports_cardinality_and_unmatched_keys():
    left = pd.DataFrame({"id": ["a", "b"], "value": [1, 2]})
    right = pd.DataFrame({"id": ["a", "c"], "other": [3, 4]})
    output = left.merge(right, on="id", how="left", validate="one_to_one", sort=False)
    evidence = _join_identity(left, right, ["id"], output)
    assert evidence["left_rows"] == 2
    assert evidence["output_rows"] == 2
    assert evidence["unmatched_left_keys"] == 1


def test_missing_horizon_is_reported_instead_of_silently_weighted():
    frame = pd.DataFrame({"h": [1, 2, 3, 4, 5, 6], "target": [0.0] * 6, "prediction": [0.0] * 6})
    assert _weighted_rmse(frame, "prediction") is None


def test_two_stage_error_aggregation_preserves_unequal_group_sizes():
    # The first group has one row and the second has three.  Averaging their
    # means and then averaging again would give the small group four times too
    # much influence; additive sums must reproduce direct row-level metrics.
    frame = pd.DataFrame(
        {
            "bucket": ["all"] * 4,
            "part": ["small", "large", "large", "large"],
            "error": [10.0, 1.0, -2.0, 3.0],
            "persistence_error": [8.0, 2.0, -1.0, 4.0],
        }
    )
    first = _aggregate_error_frame(frame, ["bucket", "part"])
    additive = [
        "rows", "b3_sse", "b3_abs_error_sum", "b3_error_sum",
        "persistence_sse", "persistence_abs_error_sum", "persistence_error_sum",
    ]
    second = first.groupby(["bucket"], as_index=False, sort=True)[additive].sum()
    actual = _finalize_error_metrics(second).iloc[0]
    direct_b3 = _metric_arrays(np.zeros(len(frame)), frame["error"].to_numpy())
    direct_persistence = _metric_arrays(np.zeros(len(frame)), frame["persistence_error"].to_numpy())
    assert np.isclose(actual["b3_rmse"], direct_b3["rmse"])
    assert np.isclose(actual["b3_mae"], direct_b3["mae"])
    assert np.isclose(actual["b3_bias"], direct_b3["bias"])
    assert np.isclose(actual["persistence_rmse"], direct_persistence["rmse"])
    assert np.isclose(actual["persistence_mae"], direct_persistence["mae"])
    assert np.isclose(actual["persistence_bias"], direct_persistence["bias"])


def test_training_exposure_join_is_anchor_and_horizon_sensitive():
    replay = pd.DataFrame(
        {
            "sample_id": ["same", "different"],
            "last_observed_date": pd.to_datetime(["2014-12-01", "2014-12-01"]),
            "h": [2, 2],
        }
    )
    training = pd.DataFrame(
        {
            "sample_id": ["same", "different"],
            "last_observed_date": pd.to_datetime(["2014-12-01", "2014-11-01"]),
            "h": [2, 3],
        }
    )
    tagged, counts = _tag_training_exposure(replay, training)
    status = tagged.set_index("sample_id")["exposure_status"]
    assert status["same"] == "exact_exposure"
    assert status["different"] == "different_anchor"
    assert counts == {
        "replay_rows": 2,
        "source_id_in_full_training_rows": 2,
        "exact_exposure_rows": 1,
        "different_anchor_rows": 1,
        "source_id_not_in_full_training_rows": 0,
    }


def test_matched_views_intentionally_withhold_unscheduled_window_rows():
    source = pd.DataFrame(
        {
            "sample_id": ["prefix", "scheduled", "withheld"],
            "time": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-02-01"]),
            "lat": [0.0, 0.0, 1.0],
            "lon": [0.0, 0.0, 0.0],
            "SPEI_01_t": [1.0, 2.0, 9.0],
            "SPEI_03_t": [1.0, 2.0, 9.0],
            "SPEI_06_t": [1.0, 2.0, 9.0],
            "SPEI_12_t": [1.0, 2.0, 9.0],
            "SOIL_MOISTURE_t": [1.0, 2.0, 9.0],
            "month_sin": [0.0, 1.0, 1.0],
            "month_cos": [1.0, 0.0, 0.0],
        }
    )
    structural = source.loc[:, ["sample_id", "time", "lat", "lon"]].copy()
    structural["TWS_t"] = [1.0, 2.0, 9.0]
    ledger = pd.DataFrame(
        {
            "sample_id": ["scheduled"],
            "source_date": pd.to_datetime(["2020-02-01"]),
            "tws_visible": [True],
        }
    )
    view = build_replay_observation_view(
        source,
        structural,
        ledger=ledger,
        first_source_month="2020-02",
        last_source_month="2020-02",
    )
    assert set(view.source["sample_id"]) == {"prefix", "scheduled"}
    assert view.withheld_window_rows == 1
    assert set(view.structural["sample_id"]) == {"prefix", "scheduled"}
