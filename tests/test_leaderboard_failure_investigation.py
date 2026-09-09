"""Unit guards for the read-only leaderboard root-cause diagnostics."""

import numpy as np
import pandas as pd

from scripts.run_leaderboard_failure_investigation import (
    _join_identity,
    _metric_arrays,
    _next_calendar_alignment,
    _weighted_rmse,
)


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

