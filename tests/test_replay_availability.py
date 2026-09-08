from __future__ import annotations

import numpy as np
import pandas as pd

from src.availability import build_replay_observation_view
from src.hydro_sequence import build_hydro_sequence_features
from src.neural_sequence import ArrayNormalizer, build_temporal_tensor, temporal_channel_names
from src.regional_context import HYDRO_COLUMNS


def _panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for month, value in [("2020-01-01", 1.0), ("2020-02-01", 999.0), ("2020-03-01", 3.0)]:
        for lat in (0.0, 1.0):
            row = {
                "sample_id": f"{lat}_{month[:7]}", "time": month, "lat": lat, "lon": 0.0,
            }
            for index, column in enumerate(HYDRO_COLUMNS):
                row[column] = value + index + lat
            rows.append(row)
    source = pd.DataFrame(rows)
    structural = source.loc[:, ["sample_id", "time", "lat", "lon"]].copy()
    structural["TWS_t"] = np.arange(len(structural), dtype=float)
    return source, structural


def test_replay_view_keeps_prefix_and_only_transplanted_window_rows():
    source, structural = _panels()
    ledger = pd.DataFrame({
        "sample_id": ["0.0_2020-03", "1.0_2020-03"],
        "source_date": pd.to_datetime(["2020-03-01", "2020-03-01"]),
        "tws_visible": [False, True],
    })
    view = build_replay_observation_view(
        source, structural, ledger=ledger, first_source_month="2020-02", last_source_month="2020-03",
    )
    assert set(view.source.sample_id) == {"0.0_2020-01", "1.0_2020-01", "0.0_2020-03", "1.0_2020-03"}
    assert view.withheld_window_rows == 2
    hidden = view.structural.set_index("sample_id")
    assert np.isnan(hidden.loc["0.0_2020-03", "TWS_t"])
    assert np.isfinite(hidden.loc["1.0_2020-03", "TWS_t"])


def test_withheld_window_covariates_cannot_enter_sequence_features():
    source, structural = _panels()
    ledger = pd.DataFrame({
        "sample_id": ["0.0_2020-03"],
        "source_date": pd.to_datetime(["2020-03-01"]),
        "tws_visible": [True],
    })
    view = build_replay_observation_view(
        source, structural, ledger=ledger, first_source_month="2020-02", last_source_month="2020-03",
    )
    sequence = build_hydro_sequence_features(view.source, ledger.sample_id, span=2)
    variable = HYDRO_COLUMNS[0]
    assert np.isnan(sequence.loc[0, f"seq_tminus1_{variable}"])
    assert sequence.loc[0, f"seq_tminus1_{variable}_observed"] == 0.0


def test_temporal_tensor_has_declared_channels_and_is_future_blind():
    source, _ = _panels()
    request = pd.Series(["0.0_2020-03"])
    base = build_temporal_tensor(source, request, span=3)
    assert base.values.shape == (1, 3, len(temporal_channel_names()))
    assert base.channels == temporal_channel_names()
    assert base.values[0, -1, base.channels.index("local_SPEI_01_t_observed")] == 1.0
    changed = source.copy()
    future = changed.loc[[0]].copy()
    future["sample_id"] = "0.0_2020-04"
    future["time"] = "2020-04-01"
    future[HYDRO_COLUMNS[0]] = 1_000_000.0
    changed = pd.concat([changed, future], ignore_index=True)
    later = build_temporal_tensor(changed, request, span=3)
    np.testing.assert_allclose(base.values, later.values, equal_nan=True)


def test_array_normalizer_round_trip(tmp_path):
    values = np.array([[[1.0, np.nan], [3.0, 10.0]]], dtype=np.float32)
    normalizer = ArrayNormalizer.fit(values)
    path = tmp_path / "normalizer.npz"
    normalizer.save(path, columns=("a", "b"))
    restored, columns = ArrayNormalizer.load(path)
    assert columns == ("a", "b")
    np.testing.assert_allclose(normalizer.transform(values), restored.transform(values))
