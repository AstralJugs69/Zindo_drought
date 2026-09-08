import json

import pandas as pd
import pytest

from scripts.analyze_neural_followups import analyze


def _metrics(path, rows):
    path.mkdir()
    pd.DataFrame(rows).to_csv(path / "neural_epoch_metrics.csv", index=False)


def test_followup_analysis_uses_selected_epoch_and_pairs_origin_seed(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"chosen": {"architecture": "gru", "span": 12, "epoch": 1}}))
    rows = [
        {"architecture": "gru", "span": 12, "epoch": 1, "origin": "a", "seed": 1, "valid_raw_rmse": .5, "train_weighted_mse": .1},
        {"architecture": "gru", "span": 12, "epoch": 1, "origin": "a", "seed": 2, "valid_raw_rmse": .7, "train_weighted_mse": .2},
        {"architecture": "gru", "span": 12, "epoch": 2, "origin": "a", "seed": 1, "valid_raw_rmse": .1, "train_weighted_mse": .1},
        {"architecture": "gru", "span": 12, "epoch": 1, "origin": "b", "seed": 1, "valid_raw_rmse": .6, "train_weighted_mse": .3},
        {"architecture": "gru", "span": 12, "epoch": 1, "origin": "b", "seed": 2, "valid_raw_rmse": .8, "train_weighted_mse": .4},
    ]
    selected, ablated, stability = tmp_path / "selected", tmp_path / "ablated", tmp_path / "stability"
    _metrics(selected, rows)
    _metrics(ablated, [{**row, "valid_raw_rmse": row["valid_raw_rmse"] + .1} for row in rows])
    _metrics(stability, [
        {"architecture": "gru", "span": 12, "epoch": 1, "origin": "a", "seed": 3, "valid_raw_rmse": .65, "train_weighted_mse": .2},
        {"architecture": "gru", "span": 12, "epoch": 1, "origin": "b", "seed": 3, "valid_raw_rmse": .75, "train_weighted_mse": .3},
    ])
    report = analyze(selection_path=selection, selected_run_dir=selected, ablation_run_dir=ablated, stability_run_dir=stability)
    assert report["history_ablation"]["overall"]["paired_origin_seed_runs"] == 4
    assert report["history_ablation"]["overall"]["history_hidden_minus_selected_raw_rmse"] == pytest.approx(.1)
    assert report["third_seed_stability"]["third_seed_mean_raw_rmse"] == pytest.approx(.7)


def test_followup_analysis_rejects_missing_pair(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"chosen": {"architecture": "gru", "span": 12, "epoch": 1}}))
    base = [{"architecture": "gru", "span": 12, "epoch": 1, "origin": "a", "seed": 1, "valid_raw_rmse": .5, "train_weighted_mse": .1}]
    selected, ablated = tmp_path / "selected", tmp_path / "ablated"
    _metrics(selected, base)
    _metrics(ablated, [{**base[0], "seed": 2}])
    with pytest.raises(AssertionError, match="identical origin/seed coverage"):
        analyze(selection_path=selection, selected_run_dir=selected, ablation_run_dir=ablated)
