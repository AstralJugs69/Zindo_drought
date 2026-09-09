"""Apply the predeclared D0/D1/D2 selection and outer promotion gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import TEST_H_WEIGHTS, raw_rmse, score_by_horizon


INNER = ("2003-04", "2004-04")
OUTER = ("2007-09", "2009-01", "2014-04", "2014-12")


def _load(run_dirs: list[Path]) -> pd.DataFrame:
    records = []
    for directory in run_dirs:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise AssertionError(f"run is not completed: {directory}")
        required = {"origin", "recipe", "training_ids_hash", "training_labels_hash", "training_weights_hash", "validation_ids_hash", "metrics"}
        if missing := required.difference(manifest):
            raise AssertionError(f"manifest missing {sorted(missing)}: {directory}")
        records.append({**manifest, "run_dir": str(directory)})
    return pd.DataFrame(records)


def _same_supervision(frame: pd.DataFrame) -> None:
    for origin, group in frame.groupby("origin", sort=True):
        for key in ("training_ids_hash", "training_labels_hash", "training_weights_hash", "validation_ids_hash"):
            if group[key].nunique() != 1:
                raise AssertionError(f"{origin}: {key} differs across D0/D1/D2")


def _inner(frame: pd.DataFrame) -> dict[str, object]:
    rows = frame.loc[frame.origin.isin(INNER)].copy()
    if set(rows.recipe) != {"D0_dense", "D1_sparse", "D2_mixed"}:
        raise AssertionError("inner analysis requires D0/D1/D2")
    _same_supervision(rows)
    metric = rows.metrics.map(lambda value: value["official_h1_7_weighted_rmse"])
    if metric.isna().any():
        raise AssertionError("inner replay is missing an official h1-7 metric")
    rows["weighted"] = metric.astype(float)
    d0 = rows.loc[rows.recipe == "D0_dense", ["origin", "weighted"]].rename(columns={"weighted": "D0_weighted"})
    if d0.groupby("origin").size().to_dict() != {origin: 1 for origin in INNER}:
        raise AssertionError("inner D0 must have one fit per origin")
    candidates = []
    for recipe in ("D1_sparse", "D2_mixed"):
        candidate = rows.loc[rows.recipe == recipe].groupby("origin", as_index=False).agg(weighted=("weighted", "mean"), seeds=("augmentation_seed", "nunique"))
        if candidate.set_index("origin").seeds.to_dict() != {origin: 2 for origin in INNER}:
            raise AssertionError(f"{recipe} requires exactly two seeds on both inner origins")
        paired = d0.merge(candidate, on="origin", validate="one_to_one")
        paired["gain_vs_D0"] = paired.D0_weighted - paired.weighted
        candidates.append({"recipe": recipe, "equal_origin_weighted_rmse": float(paired.weighted.mean()),
                           "qualifies": bool((paired.gain_vs_D0 > 0).all()), "by_origin": paired.to_dict(orient="records")})
    qualified = [item for item in candidates if item["qualifies"]]
    selected = None
    if qualified:
        qualified.sort(key=lambda item: (item["equal_origin_weighted_rmse"], 0 if item["recipe"] == "D2_mixed" else 1))
        if len(qualified) == 2 and abs(qualified[0]["equal_origin_weighted_rmse"] - qualified[1]["equal_origin_weighted_rmse"]) < .0001:
            selected = next(item for item in qualified if item["recipe"] == "D2_mixed")["recipe"]
        else:
            selected = qualified[0]["recipe"]
    return {"candidates": candidates, "selected_recipe": selected, "rule": "D1/D2 must improve D0 seed-mean official h1-7 weighted RMSE on both inner origins; lower equal-origin mean wins, with <0.0001 favouring D2."}


def _metric(frame: pd.DataFrame) -> dict[str, object]:
    supported = frame.loc[frame.h.between(1, 7)]
    present = sorted(int(value) for value in supported.h.unique())
    weighted, by_h = (None, [])
    if present == list(range(1, 8)):
        weighted, table = score_by_horizon(supported.target, supported.prediction, supported.h)
        by_h = table.reset_index().to_dict(orient="records")
    elif len(supported):
        by_h = []
        for horizon, group in supported.groupby("h", sort=True):
            error = group.prediction.to_numpy(float) - group.target.to_numpy(float)
            by_h.append({
                "h": int(horizon), "rows": int(len(group)),
                "mse": float(np.mean(np.square(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "bias": float(np.mean(error)),
            })
    present_weighted = None
    if len(supported):
        # This is a diagnostic for calendar blocks with absent horizons.  It
        # renormalizes the fixed Test horizon shares over only the horizons
        # that actually exist; it is never treated as the official score.
        weighted_mse = 0.0
        observed_weight = 0.0
        for horizon, group in supported.groupby("h", sort=True):
            h = int(horizon)
            weight = float(TEST_H_WEIGHTS[h])
            weighted_mse += weight * float(np.mean(np.square(group.prediction - group.target)))
            observed_weight += weight
        present_weighted = float(np.sqrt(weighted_mse / observed_weight))
    tail = frame.loc[~frame.h.between(1, 7)]
    return {"raw_rmse": raw_rmse(frame.target, frame.prediction), "official_h1_7_weighted_rmse": weighted,
            "official_horizons_present": present, "present_h1_7_weighted_rmse": present_weighted,
            "present_horizon_weight_total": float(sum(TEST_H_WEIGHTS[h] for h in present)), "by_h": by_h,
            "stress_h_gt7_rows": int(len(tail)), "stress_h_gt7_raw_rmse": None if tail.empty else raw_rmse(tail.target, tail.prediction)}


def _outer(frame: pd.DataFrame, selected: str | None) -> dict[str, object] | None:
    if selected is None:
        return None
    rows = frame.loc[frame.origin.isin(OUTER)].copy()
    if rows.empty:
        return None
    _same_supervision(rows)
    output = []
    for origin in OUTER:
        group = rows.loc[rows.origin == origin]
        d0 = group.loc[group.recipe == "D0_dense"]
        candidate = group.loc[group.recipe == selected]
        if len(d0) != 1 or len(candidate) != 2:
            raise AssertionError(f"{origin}: expected one D0 and two {selected} fits")
        base = pd.read_csv(Path(d0.iloc[0].run_dir) / "oof.csv.gz")
        predictions = []
        for directory in candidate.run_dir:
            current = pd.read_csv(Path(directory) / "oof.csv.gz")
            joined = base[["sample_id", "target", "h", "last_observed_TWS"]].merge(current[["sample_id", "target", "h", "last_observed_TWS", "prediction"]], on="sample_id", suffixes=("_d0", "_candidate"), validate="one_to_one")
            if len(joined) != len(base) or not np.array_equal(joined.target_d0, joined.target_candidate) or not np.array_equal(joined.h_d0, joined.h_candidate) or not np.array_equal(joined.last_observed_TWS_d0, joined.last_observed_TWS_candidate):
                raise AssertionError(f"{origin}: OOF provenance differs")
            predictions.append(joined.prediction.to_numpy(float))
        base_metric = _metric(base.rename(columns={"prediction": "prediction"}))
        candidate_frame = base.copy(); candidate_frame["prediction"] = np.mean(predictions, axis=0)
        candidate_metric = _metric(candidate_frame)
        delta = None
        if base_metric["official_h1_7_weighted_rmse"] is not None and candidate_metric["official_h1_7_weighted_rmse"] is not None:
            delta = float(candidate_metric["official_h1_7_weighted_rmse"] - base_metric["official_h1_7_weighted_rmse"])
        present_delta = None
        if (
            base_metric["present_h1_7_weighted_rmse"] is not None
            and candidate_metric["present_h1_7_weighted_rmse"] is not None
            and base_metric["official_horizons_present"] == candidate_metric["official_horizons_present"]
        ):
            present_delta = float(candidate_metric["present_h1_7_weighted_rmse"] - base_metric["present_h1_7_weighted_rmse"])
        output.append({"origin": origin, "D0": base_metric, selected: candidate_metric,
                       "candidate_minus_D0_official_weighted": delta,
                       "candidate_minus_D0_present_horizon_weighted": present_delta})
    deltas = [item["candidate_minus_D0_official_weighted"] for item in output]
    present_deltas = [item["candidate_minus_D0_present_horizon_weighted"] for item in output]
    official_complete = not any(value is None for value in deltas)
    recent = [item for item in output if item["origin"] in {"2014-04", "2014-12"}]
    promote = False
    if official_complete:
        promote = bool(float(np.mean(deltas)) < 0 and all(item["candidate_minus_D0_official_weighted"] <= 0 for item in recent) and all(value <= .003 for value in deltas))
    return {
        "selected_recipe": selected, "by_origin": output,
        "official_gate_complete": official_complete,
        "equal_origin_mean_candidate_minus_D0_official_weighted": None if not official_complete else float(np.mean(deltas)),
        "equal_origin_mean_candidate_minus_D0_present_horizon_weighted": None if any(value is None for value in present_deltas) else float(np.mean(present_deltas)),
        "promotion_blockers": [item["origin"] for item in output if item["candidate_minus_D0_official_weighted"] is None],
        "promote": promote,
        "rule": "Promote only if the complete official h1-7 metric improves on average, neither recent origin regresses, and no origin deteriorates by more than 0.003 weighted RMSE. Present-horizon scores are diagnostics only and cannot satisfy this gate.",
    }


def analyze(run_dirs: list[Path]) -> dict[str, object]:
    runs = _load(run_dirs)
    inner = _inner(runs)
    outer = _outer(runs, inner["selected_recipe"])
    return {"runs": runs[["origin", "recipe", "augmentation_seed", "run_dir"]].to_dict(orient="records"), "inner": inner, "outer": outer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gap_augmentation_analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
