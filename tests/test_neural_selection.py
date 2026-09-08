from scripts.run_neural_sequence_comparison import _chosen_epoch, _selection


def test_inner_selection_keeps_static_mlp_null_span_control():
    rows = [
        {"architecture": "mlp", "span": None, "epoch": 1, "valid_raw_rmse": 0.50},
        {"architecture": "mlp", "span": None, "epoch": 1, "valid_raw_rmse": 0.51},
        {"architecture": "mlp", "span": None, "epoch": 1, "valid_raw_rmse": 0.49},
        {"architecture": "mlp", "span": None, "epoch": 1, "valid_raw_rmse": 0.50},
        {"architecture": "gru", "span": 6, "epoch": 1, "valid_raw_rmse": 0.60},
        {"architecture": "gru", "span": 6, "epoch": 1, "valid_raw_rmse": 0.61},
        {"architecture": "gru", "span": 6, "epoch": 1, "valid_raw_rmse": 0.59},
        {"architecture": "gru", "span": 6, "epoch": 1, "valid_raw_rmse": 0.60},
    ]

    selected = _selection(rows)

    assert selected["chosen"] == {
        "architecture": "mlp",
        "span": 0,
        "epoch": 1,
        "raw_rmse": 0.5,
        "runs": 4,
    }


def test_frozen_epoch_does_not_reselect_on_outer_validation_labels():
    rows = [
        {"epoch": 1, "valid_raw_rmse": .8},
        {"epoch": 2, "valid_raw_rmse": .2},
        {"epoch": 3, "valid_raw_rmse": .1},
    ]
    assert _chosen_epoch(rows, None)["epoch"] == 3
    assert _chosen_epoch(rows, 1)["epoch"] == 1
