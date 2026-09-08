"""Analyze availability-faithful B3 and compact-neural replay OOF artifacts.

This script never reads competition Test labels or writes predictions.  It joins
completed replay OOF files by exact sample id, averages the predeclared neural
seeds, and reports paired B3/neural/equal-blend comparisons.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import TEST_H_WEIGHTS, raw_rmse, score_by_horizon


MODELS = ("B3", "neural", "equal_blend")


def _metric_row(frame: pd.DataFrame, *, label: str) -> dict[str, object]:
    truth = frame["target"].to_numpy(float)
    row: dict[str, object] = {"slice": label, "rows": int(len(frame))}
    for model in MODELS:
        prediction = frame[f"prediction_{model}"].to_numpy(float)
        row[f"{model}_raw_rmse"] = raw_rmse(truth, prediction)
        if set(frame.h.astype(int)) == set(range(1, 8)):
            row[f"{model}_weighted_rmse"] = score_by_horizon(truth, prediction, frame.h)[0]
        else:
            row[f"{model}_weighted_rmse"] = None
    row["neural_minus_B3_raw_rmse"] = float(row["neural_raw_rmse"] - row["B3_raw_rmse"])
    row["blend_minus_B3_raw_rmse"] = float(row["equal_blend_raw_rmse"] - row["B3_raw_rmse"])
    if row["neural_weighted_rmse"] is not None:
        row["neural_minus_B3_weighted_rmse"] = float(row["neural_weighted_rmse"] - row["B3_weighted_rmse"])
        row["blend_minus_B3_weighted_rmse"] = float(row["equal_blend_weighted_rmse"] - row["B3_weighted_rmse"])
    return row


def _prepare(run_dir: Path) -> pd.DataFrame:
    paths = sorted((run_dir / "baseline").glob("*_B3_oof.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"no B3 OOF files under {run_dir / 'baseline'}")
    baseline = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    neural_path = run_dir / "neural_best_oof.csv.gz"
    neural = pd.read_csv(neural_path)
    metadata = ["sample_id", "target", "h", "origin", "anchor_age_months", "calendar_block", "geo5"]
    if neural.groupby("sample_id").size().nunique() != 1:
        raise AssertionError("neural seeds do not have complete, equal OOF coverage")
    averaged = neural.groupby("sample_id", as_index=False).agg(
        **{key: (key, "first") for key in metadata if key != "sample_id"},
        prediction_neural=("prediction", "mean"),
        neural_seed_count=("seed", "nunique"),
    )
    if not (averaged.neural_seed_count == 2).all():
        raise AssertionError("outer neural OOF does not contain exactly two predeclared seeds per ID")
    b3 = baseline.loc[:, metadata + ["prediction"]].rename(columns={"prediction": "prediction_B3"})
    joined = b3.merge(averaged, on="sample_id", how="inner", suffixes=("_B3_meta", "_neural_meta"), validate="one_to_one")
    if len(joined) != len(b3) or len(joined) != len(averaged):
        raise AssertionError("B3 and neural OOF ID coverage differs")
    for key in metadata[1:]:
        left, right = f"{key}_B3_meta", f"{key}_neural_meta"
        if left in joined and right in joined:
            if not joined[left].equals(joined[right]):
                raise AssertionError(f"B3/neural metadata differs for {key}")
            joined[key] = joined[left]
            joined = joined.drop(columns=[left, right])
    joined["prediction_equal_blend"] = (joined.prediction_B3 + joined.prediction_neural) / 2.0
    joined["residual_B3"] = joined.prediction_B3 - joined.target
    joined["residual_neural"] = joined.prediction_neural - joined.target
    return joined


def _bootstrap(frame: pd.DataFrame, *, block: str, repeats: int, seed: int) -> dict[str, object]:
    groups = list(frame.groupby(block, sort=True).indices.values())
    if len(groups) < 2:
        return {"block": block, "repeats": 0, "note": "fewer than two blocks"}
    rng = np.random.default_rng(seed)
    deltas = np.empty((repeats, 2), dtype=float)
    for index in range(repeats):
        picked = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[item] for item in picked])
        sample = frame.iloc[rows]
        metric = _metric_row(sample, label="bootstrap")
        deltas[index] = [metric["neural_minus_B3_raw_rmse"], metric["blend_minus_B3_raw_rmse"]]
    return {
        "block": block,
        "repeats": repeats,
        "neural_minus_B3_raw_rmse": {"p025": float(np.quantile(deltas[:, 0], .025)), "median": float(np.median(deltas[:, 0])), "p975": float(np.quantile(deltas[:, 0], .975))},
        "blend_minus_B3_raw_rmse": {"p025": float(np.quantile(deltas[:, 1], .025)), "median": float(np.median(deltas[:, 1])), "p975": float(np.quantile(deltas[:, 1], .975))},
    }


def analyze(run_dir: Path, output_dir: Path, *, repeats: int = 200) -> dict[str, object]:
    frame = _prepare(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = _metric_row(frame, label="all_outer_oof")
    by_origin = pd.DataFrame(_metric_row(group, label=str(origin)) for origin, group in frame.groupby("origin", sort=True))
    by_h = pd.DataFrame(_metric_row(group, label=f"h{horizon}") for horizon, group in frame.groupby("h", sort=True))
    age = pd.cut(frame.anchor_age_months, bins=[-1, 0, 1, 2, np.inf], labels=["0", "1", "2", "3+"])
    by_age = pd.DataFrame(_metric_row(group, label=f"anchor_age_{bucket}") for bucket, group in frame.groupby(age, observed=True, sort=True))
    correlation = float(np.corrcoef(frame.residual_B3, frame.residual_neural)[0, 1])
    report = {
        "run_dir": str(run_dir),
        "rows": int(len(frame)),
        "official_horizon_weights": {str(key): value for key, value in TEST_H_WEIGHTS.items()},
        "overall": overall,
        "residual_correlation_B3_neural": correlation,
        "paired_block_bootstrap": [_bootstrap(frame, block="calendar_block", repeats=repeats, seed=20260908), _bootstrap(frame, block="geo5", repeats=repeats, seed=20260909)],
        "limitations": ["Outer replays are development robustness evidence, not an untouched confirmation.", "Block resampling respects calendar or 5-degree geography separately; the limited number of calendar blocks constrains certainty."],
    }
    (output_dir / "neural_sequence_analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame([overall]).to_csv(output_dir / "overall_metrics.csv", index=False)
    by_origin.to_csv(output_dir / "by_origin_metrics.csv", index=False)
    by_h.to_csv(output_dir / "by_h_metrics.csv", index=False)
    by_age.to_csv(output_dir / "by_anchor_age_metrics.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir, args.output_dir, repeats=args.bootstrap_repeats), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
