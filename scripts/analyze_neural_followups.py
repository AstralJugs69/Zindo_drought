"""Analyze preregistered neural stability and temporal-history ablation runs.

This utility reads only replay metric artifacts.  It does not read competition
Test labels or create predictions.  The selected recipe is taken from the
immutable inner ``selection.json``; unlike a generic OOF analysis, this keeps
the comparison at the explicitly selected epoch even when a seed's individually
best checkpoint happened at another epoch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = ["origin", "seed", "valid_raw_rmse", "train_weighted_mse"]


def _selected(selection_path: Path) -> dict[str, object]:
    chosen = json.loads(selection_path.read_text(encoding="utf-8"))["chosen"]
    return {
        "architecture": str(chosen["architecture"]),
        "span": int(chosen["span"]),
        "epoch": int(chosen["epoch"]),
    }


def _rows(run_dir: Path, recipe: dict[str, object]) -> pd.DataFrame:
    metrics = pd.read_csv(run_dir / "neural_epoch_metrics.csv")
    chosen = metrics.loc[
        (metrics.architecture == recipe["architecture"])
        & (metrics.span == recipe["span"])
        & (metrics.epoch == recipe["epoch"]),
        METRIC_COLUMNS,
    ].copy()
    if chosen.empty:
        raise AssertionError(f"selected recipe absent from {run_dir}")
    if chosen.duplicated(["origin", "seed"]).any():
        raise AssertionError(f"duplicate selected origin/seed metrics in {run_dir}")
    return chosen.sort_values(["origin", "seed"], kind="mergesort").reset_index(drop=True)


def analyze(
    *,
    selection_path: Path,
    selected_run_dir: Path,
    ablation_run_dir: Path,
    stability_run_dir: Path | None = None,
) -> dict[str, object]:
    recipe = _selected(selection_path)
    selected = _rows(selected_run_dir, recipe).rename(columns={"valid_raw_rmse": "selected_raw_rmse", "train_weighted_mse": "selected_train_weighted_mse"})
    ablated = _rows(ablation_run_dir, recipe).rename(columns={"valid_raw_rmse": "history_hidden_raw_rmse", "train_weighted_mse": "history_hidden_train_weighted_mse"})
    paired = selected.merge(ablated, on=["origin", "seed"], how="inner", validate="one_to_one")
    if len(paired) != len(selected) or len(paired) != len(ablated):
        raise AssertionError("selected and history-hidden runs do not have identical origin/seed coverage")
    paired["history_hidden_minus_selected_raw_rmse"] = paired.history_hidden_raw_rmse - paired.selected_raw_rmse
    by_origin = paired.groupby("origin", as_index=False).agg(
        selected_raw_rmse=("selected_raw_rmse", "mean"),
        history_hidden_raw_rmse=("history_hidden_raw_rmse", "mean"),
        history_hidden_minus_selected_raw_rmse=("history_hidden_minus_selected_raw_rmse", "mean"),
        seeds=("seed", "nunique"),
    )
    overall = {
        "selected_raw_rmse": float(paired.selected_raw_rmse.mean()),
        "history_hidden_raw_rmse": float(paired.history_hidden_raw_rmse.mean()),
        "history_hidden_minus_selected_raw_rmse": float(paired.history_hidden_minus_selected_raw_rmse.mean()),
        "paired_origin_seed_runs": int(len(paired)),
    }
    report: dict[str, object] = {
        "selected_recipe": recipe,
        "history_ablation": {
            "selected_run_dir": str(selected_run_dir),
            "ablation_run_dir": str(ablation_run_dir),
            "overall": overall,
            "by_origin": by_origin.to_dict(orient="records"),
            "paired_rows": paired.to_dict(orient="records"),
            "interpretation_limit": "This ablation removes older raw temporal slots while retaining current-source and static channels; it tests reliance on older temporal inputs, not an information-free model.",
        },
    }
    if stability_run_dir is not None:
        stability = _rows(stability_run_dir, recipe).rename(columns={"valid_raw_rmse": "third_seed_raw_rmse", "train_weighted_mse": "third_seed_train_weighted_mse"})
        joined = selected.merge(stability, on="origin", how="inner", validate="many_to_one")
        if stability.origin.nunique() != selected.origin.nunique():
            raise AssertionError("third-seed stability run does not cover every selected origin")
        report["third_seed_stability"] = {
            "stability_run_dir": str(stability_run_dir),
            "by_origin": stability.loc[:, ["origin", "seed", "third_seed_raw_rmse", "third_seed_train_weighted_mse"]].to_dict(orient="records"),
            "selected_two_seed_mean_raw_rmse": float(selected.selected_raw_rmse.mean()),
            "third_seed_mean_raw_rmse": float(stability.third_seed_raw_rmse.mean()),
            "third_minus_selected_two_seed_mean_raw_rmse": float(stability.third_seed_raw_rmse.mean() - selected.selected_raw_rmse.mean()),
            "coverage": {"selected_origin_seed_rows": int(len(selected)), "third_seed_origin_rows": int(len(stability)), "joined_rows": int(len(joined))},
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selected-run-dir", type=Path, required=True)
    parser.add_argument("--ablation-run-dir", type=Path, required=True)
    parser.add_argument("--stability-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(selection_path=args.selection, selected_run_dir=args.selected_run_dir, ablation_run_dir=args.ablation_run_dir, stability_run_dir=args.stability_run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "neural_followup_analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(report["history_ablation"]["by_origin"]).to_csv(args.output_dir / "history_ablation_by_origin.csv", index=False)
    pd.DataFrame(report["history_ablation"]["paired_rows"]).to_csv(args.output_dir / "history_ablation_paired_origin_seed.csv", index=False)
    if "third_seed_stability" in report:
        pd.DataFrame(report["third_seed_stability"]["by_origin"]).to_csv(args.output_dir / "third_seed_stability_by_origin.csv", index=False)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
