from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_neural_sequence_results import analyze


def test_analysis_requires_exact_two_seed_alignment_and_writes_paired_metrics(tmp_path: Path):
    run = tmp_path / "run"
    (run / "baseline").mkdir(parents=True)
    rows = []
    for horizon in range(1, 8):
        rows.append({
            "sample_id": f"sample-{horizon}", "target": float(horizon), "h": horizon,
            "origin": "2007-09", "anchor_age_months": horizon % 3,
            "calendar_block": f"2007-{horizon:02d}", "geo5": f"g{horizon % 2}",
            "prediction": float(horizon) + .2,
        })
    baseline = pd.DataFrame(rows)
    baseline.to_csv(run / "baseline" / "2007-09_B3_oof.csv.gz", index=False, compression="gzip")
    neural = pd.concat([
        baseline.assign(prediction=baseline.target + offset, seed=seed, model="gru", span=12, epoch=1)
        for seed, offset in ((20260908, .1), (20260909, .3))
    ], ignore_index=True)
    neural.to_csv(run / "neural_best_oof.csv.gz", index=False, compression="gzip")

    result = analyze(run, tmp_path / "analysis", repeats=5)

    assert np.isclose(result["overall"]["B3_raw_rmse"], .2)
    assert (tmp_path / "analysis" / "neural_sequence_analysis.json").is_file()
    assert (tmp_path / "analysis" / "by_h_metrics.csv").is_file()
