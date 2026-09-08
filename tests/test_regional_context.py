import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.run_regional_capacity import _select_capacity  # noqa: E402
from src.regional_context import (  # noqa: E402
    HYDRO_COLUMNS,
    attach_regional_features,
    build_regional_context,
    regional_feature_names,
)


def _source():
    rows = []
    for t in ("2001-01", "2001-02"):
        for i, (lat, lon) in enumerate(((0.0, 0.0), (1.0, 1.0), (9.0, 9.0), (20.0, 20.0))):
            row = {"sample_id": f"{t}_{i}", "time": t, "lat": lat, "lon": lon}
            for j, c in enumerate(HYDRO_COLUMNS):
                row[c] = 10 * j + i + (0 if t == "2001-01" else 100)
            rows.append(row)
    return pd.DataFrame(rows)


def _keyed(frame):
    return frame.set_index("sample_id").sort_index()


def test_regional_builder_is_row_permutation_invariant():
    x = _source()
    a = _keyed(build_regional_context(x))
    b = _keyed(build_regional_context(x.sample(frac=1.0, random_state=3).reset_index(drop=True)))
    assert a.columns.tolist() == regional_feature_names()
    assert a.index.tolist() == b.index.tolist()
    assert np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)


def test_unavailable_row_is_not_secretly_used_in_aggregate():
    x = _source()
    full = _keyed(build_regional_context(x, widths=(5.0,)))
    removed_id = "2001-01_1"
    reduced_x = x.loc[x.sample_id != removed_id].reset_index(drop=True)
    reduced = _keyed(build_regional_context(reduced_x, widths=(5.0,)))
    # Cell (0,0)/(1,1) shares the 5-degree bin. Removing one row must change
    # the other row's count and mean rather than reconstructing the absent row.
    sid = "2001-01_0"
    assert full.loc[sid, "count5_SPEI_01_t"] == 2
    assert reduced.loc[sid, "count5_SPEI_01_t"] == 1
    assert full.loc[sid, "reg5_SPEI_01_t"] != reduced.loc[sid, "reg5_SPEI_01_t"]


def test_future_and_label_like_columns_cannot_affect_features():
    x = _source()
    base = build_regional_context(x)
    mutated = x.copy()
    future = mutated.time == "2001-02"
    for c in HYDRO_COLUMNS:
        mutated.loc[future, c] += 1e6
    # January features must not change when a later source date changes.
    changed = build_regional_context(mutated)
    jan = x.time == "2001-01"
    assert np.allclose(base.loc[jan, regional_feature_names()].to_numpy(float),
                       changed.loc[jan, regional_feature_names()].to_numpy(float), equal_nan=True)
    with_target = x.assign(target=999.0)
    try:
        build_regional_context(with_target)
    except AssertionError:
        pass
    else:
        raise AssertionError("target-bearing frame should be rejected")


def test_missing_values_remain_missing_with_counts_and_coverage():
    x = _source()
    x.loc[(x.time == "2001-01") & (x.lat <= 1.0), "SPEI_01_t"] = np.nan
    out = _keyed(build_regional_context(x, widths=(5.0,)))
    sid = "2001-01_0"
    assert np.isnan(out.loc[sid, "reg5_SPEI_01_t"])
    assert np.isnan(out.loc[sid, "dev5_SPEI_01_t"])
    assert out.loc[sid, "count5_SPEI_01_t"] == 0
    assert out.loc[sid, "coverage5_SPEI_01_t"] == 0


def test_training_inference_attachment_uses_identical_keyed_builder():
    x = _source()
    regional = build_regional_context(x)
    ids = x.sample(frac=1.0, random_state=9).sample_id.reset_index(drop=True)
    a = attach_regional_features(ids, regional)
    b = regional.set_index("sample_id").loc[ids].reset_index(drop=True)
    assert np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)

def test_capacity_selection_uses_mean_score_and_upper_median_iteration():
    records = [
        {"candidate": "C0_safe", "config_name": "current_63_1000", "weighted_rmse": 0.60, "best_iteration": 100},
        {"candidate": "C0_safe", "config_name": "current_63_1000", "weighted_rmse": 0.62, "best_iteration": 300},
        {"candidate": "C0_safe", "config_name": "larger_127_200", "weighted_rmse": 0.59, "best_iteration": 200},
        {"candidate": "C0_safe", "config_name": "larger_127_200", "weighted_rmse": 0.60, "best_iteration": 400},
    ]
    selected = _select_capacity(records, "C0_safe")
    assert selected["config_name"] == "larger_127_200"
    assert selected["selected_rounds"] == 400
    assert selected["iteration_rule"] == "upper_median_inner_best_iteration"
