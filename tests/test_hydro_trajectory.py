from __future__ import annotations

import numpy as np
import pandas as pd

from src.hydro_trajectory import (
    attach_trajectory_features,
    build_hydro_trajectory_map,
    build_regional_trajectory_map,
)
from src.regional_context import HYDRO_COLUMNS, build_regional_context


def _source() -> pd.DataFrame:
    rows = []
    values = {
        (0.0, 0.0): [("2020-01-01", 1.0), ("2020-03-01", 5.0), ("2020-04-01", 7.0)],
        (1.0, 1.0): [("2020-01-01", 3.0), ("2020-03-01", 7.0), ("2020-04-01", 9.0)],
    }
    for (lat, lon), series in values.items():
        for time, value in series:
            row = {"sample_id": f"{lat}_{lon}_{time[:7]}", "time": time, "lat": lat, "lon": lon}
            for number, column in enumerate(HYDRO_COLUMNS):
                row[column] = value + number
            rows.append(row)
    return pd.DataFrame(rows)


def test_calendar_windows_preserve_real_gaps_and_elapsed_slope():
    result = build_hydro_trajectory_map(_source()).set_index("sample_id")
    row = result.loc["0.0_0.0_2020-03"]
    assert row["local_SPEI_01_t_trail3_count"] == 2.0
    assert np.isclose(row["local_SPEI_01_t_trail3_coverage"], 2.0 / 3.0)
    assert np.isclose(row["local_SPEI_01_t_trail3_mean"], 3.0)
    assert np.isclose(row["local_SPEI_01_t_trail3_slope"], 2.0)
    assert np.isclose(row["local_SPEI_01_t_recent_elapsed_months"], 2.0)
    assert row["local_SPEI_01_t_observed_run_length"] == 1.0


def test_future_changes_labels_and_permutations_cannot_change_prior_features():
    source = _source()
    base = build_hydro_trajectory_map(source).set_index("sample_id")
    changed = source.copy()
    changed.loc[changed.time == "2020-04-01", "SPEI_01_t"] = 1_000_000.0
    changed_map = build_hydro_trajectory_map(changed.sample(frac=1.0, random_state=3)).set_index("sample_id")
    january = "0.0_0.0_2020-01"
    march = "0.0_0.0_2020-03"
    np.testing.assert_allclose(base.loc[[january, march]].to_numpy(float), changed_map.loc[[january, march]].to_numpy(float), equal_nan=True)
    try:
        build_hydro_trajectory_map(source.assign(target=0.0))
    except AssertionError:
        pass
    else:
        raise AssertionError("target-bearing source frame should be rejected")


def test_attachment_is_keyed_and_regional_trends_use_only_regional_source_values():
    source = _source()
    local = build_hydro_trajectory_map(source)
    attached = attach_trajectory_features(source.sample_id.iloc[::-1].reset_index(drop=True), local)
    assert len(attached) == len(source)
    regional = build_regional_context(source)
    regional_map = build_regional_trajectory_map(source, regional, local, widths=(5.0,))
    row = regional_map.set_index("sample_id").loc["0.0_0.0_2020-03"]
    assert np.isclose(row["regional5_SPEI_01_t_trail3_mean"], 4.0)
    assert np.isclose(row["local_minus_reg5_SPEI_01_t_trail3_mean"], -1.0)
    changed = source.copy()
    changed.loc[changed.time == "2020-04-01", "SPEI_01_t"] = -1_000_000.0
    changed_local = build_hydro_trajectory_map(changed)
    changed_regional = build_regional_context(changed)
    changed_map = build_regional_trajectory_map(changed, changed_regional, changed_local, widths=(5.0,)).set_index("sample_id")
    assert np.isclose(
        row["regional5_SPEI_01_t_trail3_mean"],
        changed_map.loc["0.0_0.0_2020-03", "regional5_SPEI_01_t_trail3_mean"],
    )
