import json

import numpy as np
import pandas as pd

from scripts.analyze_covariate_gap_augmentation import _metric, analyze


def _run(root, origin, recipe, seed, weighted):
    root.mkdir()
    manifest = {"status": "completed", "origin": origin, "recipe": recipe, "augmentation_seed": seed,
                "training_ids_hash": "ids", "training_labels_hash": "labels", "training_weights_hash": "weights", "validation_ids_hash": origin,
                "metrics": {"official_h1_7_weighted_rmse": weighted}}
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_inner_selection_requires_both_origin_improvements_and_prefers_d2_on_tie(tmp_path):
    paths = []
    for origin, d0, d1, d2 in [("2003-04", .6, .59, .58), ("2004-04", .7, .69, .68)]:
        for recipe, seed, metric in [("D0_dense", None, d0), ("D1_sparse", 20260909, d1), ("D1_sparse", 20260910, d1), ("D2_mixed", 20260909, d2), ("D2_mixed", 20260910, d2)]:
            path = tmp_path / f"{origin}_{recipe}_{seed}"; _run(path, origin, recipe, seed, metric); paths.append(path)
    report = analyze(paths)
    assert report["inner"]["selected_recipe"] == "D2_mixed"


def test_inner_rejects_recipe_that_regresses_one_origin(tmp_path):
    paths = []
    for origin, d0, d1, d2 in [("2003-04", .6, .59, .61), ("2004-04", .7, .71, .69)]:
        for recipe, seed, metric in [("D0_dense", None, d0), ("D1_sparse", 20260909, d1), ("D1_sparse", 20260910, d1), ("D2_mixed", 20260909, d2), ("D2_mixed", 20260910, d2)]:
            path = tmp_path / f"{origin}_{recipe}_{seed}"; _run(path, origin, recipe, seed, metric); paths.append(path)
    report = analyze(paths)
    assert report["inner"]["selected_recipe"] is None


def test_metric_reports_present_horizon_diagnostic_without_inventing_official_score():
    frame = pd.DataFrame(
        {
            "h": [1, 2, 3, 5, 6, 7, 8],
            "target": np.zeros(7),
            "prediction": np.ones(7),
        }
    )
    metric = _metric(frame)
    assert metric["official_h1_7_weighted_rmse"] is None
    assert metric["official_horizons_present"] == [1, 2, 3, 5, 6, 7]
    assert metric["present_h1_7_weighted_rmse"] == 1.0
    assert metric["present_horizon_weight_total"] < 1.0
    assert metric["stress_h_gt7_rows"] == 1
