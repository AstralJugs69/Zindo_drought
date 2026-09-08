"""Paired combined-regional + LightGBM capacity experiment (Kaggle only).

No Test predictions or submission files are produced.  The Test table is read
only to verify the supplied source-field/mask contract and to construct the
historical replay template.  Capacity is selected on two chronological inner
replays that both end before the earliest outer evaluation, then frozen across
all outer development/stress folds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiment import _attach_delta, _score
from scripts.run_lgbm_core import DEFAULT_PARAMS
from src.ml_features import (
    HYDRO_GAP_SAFE_FEATURE_COLUMNS,
    SOURCE_CORE_COLUMNS,
    SOURCE_HYDRO_HISTORY_COLUMNS,
    build_hydro_gap_safe_feature_matrix,
    build_sampled_training_rows,
    horizon_rebalance_weights,
)
from src.observation_simulator import SimulatedFold, build_mask_block_fold, build_template_replay_fold
from src.regional_context import (
    HYDRO_COLUMNS,
    attach_regional_features,
    build_regional_context,
    regional_bytes_hash,
    regional_feature_names,
)
from src.validation import build_test_mask_template, find_exact_template_starts

OUTER_ORIGINS = ("2007-09", "2009-01", "2014-04", "2014-12")
WIDTHS = (5.0, 15.0)
MAX_ROUNDS = 1500
PATIENCE = 100
SEED = 20260907
CAPACITY_CONFIGS = (
    {"name": "current_63_1000", "num_leaves": 63, "min_data_in_leaf": 1000},
    {"name": "larger_127_200", "num_leaves": 127, "min_data_in_leaf": 200},
)


def _json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _memory() -> dict[str, object]:
    try:
        import psutil
        p = psutil.Process()
        vm = psutil.virtual_memory()
        return {"rss_gib": p.memory_info().rss / 1024**3, "available_gib": vm.available / 1024**3}
    except Exception as exc:
        return {"note": f"memory unavailable: {type(exc).__name__}: {exc}"}


def _feature_frame(base: pd.DataFrame, regional: pd.DataFrame | None) -> pd.DataFrame:
    if regional is None:
        return base.reset_index(drop=True)
    out = pd.concat([base.reset_index(drop=True), regional.reset_index(drop=True)], axis=1)
    return out


def _fold(train: pd.DataFrame, template: pd.DataFrame, origin: str) -> tuple[SimulatedFold, str]:
    if origin == "2014-04":
        fold = build_mask_block_fold(train, anchor_month=origin, end_month="2014-10",
                                     scenario_id="regional_recent_2014_04", family="recent_mask_block")
        return fold, "recent_development"
    if origin == "2014-12":
        full = build_mask_block_fold(train, anchor_month=origin, end_month="2015-06",
                                     scenario_id="regional_recent_2014_12", family="recent_mask_block")
        keep = full.ledger.h.between(1, 7).to_numpy()
        fold = SimulatedFold(
            full.spec, full.ledger.loc[keep].reset_index(drop=True),
            full.labels.loc[keep].reset_index(drop=True),
            dict(full.exclusions, outside_competition_horizon=int((~keep).sum())),
        )
        return fold, "recent_development"
    return build_template_replay_fold(
        train, template, start_month=origin,
        scenario_id=f"regional_{origin}", family="development",
    ), "development"


def _target_months(fold: SimulatedFold) -> set[pd.Period]:
    return set(pd.to_datetime(fold.ledger["target_date"]).dt.to_period("M"))


def _choose_inner_replays(train: pd.DataFrame, template: pd.DataFrame) -> list[SimulatedFold]:
    earliest_outer = pd.Period("2007-09", freq="M")
    starts = find_exact_template_starts(train, template)
    offsets = sorted(int(v) for v in template["offset_months"].unique())
    max_offset = max(offsets)
    eligible = [s for s in starts if s + max_offset + 1 < earliest_outer]
    if len(eligible) < 2:
        raise AssertionError(f"need two inner replay starts before {earliest_outer}, found {len(eligible)}")
    # The 40-month transplanted Test geometry makes two fully disjoint inner
    # target calendars impossible before the earliest outer origin. Freeze the
    # deterministic pair with minimum target-month overlap instead. Ties favor
    # wider temporal separation, then the latest second replay. This uses only
    # calendar structure; no labels or scores participate in the choice.
    pairs: list[tuple[int, int, pd.Period, pd.Period]] = []
    for i, a in enumerate(eligible):
        ta = {a + off + 1 for off in offsets}
        for b in eligible[i + 1:]:
            tb = {b + off + 1 for off in offsets}
            overlap = len(ta.intersection(tb))
            separation = int(b.ordinal - a.ordinal)
            pairs.append((overlap, -separation, a, b))
    if not pairs:
        raise AssertionError("could not freeze two inner replay calendars")
    _, _, first, second = min(pairs, key=lambda x: (x[0], x[1], -x[3].ordinal))
    chosen = [first, second]
    return [build_template_replay_fold(
        train, template, start_month=start,
        scenario_id=f"capacity_inner_{start}", family="inner_capacity_selection",
    ) for start in chosen]

def _reserved_transfer_block(train: pd.DataFrame, excluded_targets: set[pd.Period]) -> dict[str, object] | None:
    periods = sorted(pd.to_datetime(train["time"]).dt.to_period("M").unique())
    # Search recent-to-old using coverage only; do not score candidates here.
    for start in reversed(periods):
        if start >= pd.Period("2014-04", freq="M") or start < pd.Period("2012-01", freq="M"):
            continue
        try:
            fold = build_mask_block_fold(train, anchor_month=start, end_month=start + 6,
                                         scenario_id=f"reserved_transfer_{start}", family="reserved_transfer")
        except Exception:
            continue
        targets = _target_months(fold)
        if not targets.isdisjoint(excluded_targets):
            continue
        if len(fold.ledger) < 50000 or len(set(fold.ledger.h.astype(int))) < 3:
            continue
        return {
            "anchor": str(start), "end": str(start + 6), "rows": int(len(fold.ledger)),
            "h_counts": {str(k): int(v) for k, v in fold.ledger.h.value_counts().sort_index().items()},
            "target_months": sorted(str(p) for p in targets),
            "status": "reserved_unscored",
        }
    return None


def _weighted_rmse_from_fold(y: np.ndarray, pred: np.ndarray, h: pd.Series) -> float:
    # Same per-horizon test weights as the competition metric when all 1..7 exist;
    # otherwise use rowwise RMSE for inner structural blocks.
    present = sorted(int(v) for v in pd.Series(h).unique())
    if present == list(range(1, 8)):
        from src.metrics import score_by_horizon
        value, _ = score_by_horizon(y, pred, h)
        return float(value)
    return float(np.sqrt(np.mean(np.square(pred - y))))


def _select_capacity(records: list[dict[str, object]], candidate: str) -> dict[str, object]:
    rows = [r for r in records if r["candidate"] == candidate]
    if not rows:
        raise AssertionError(f"no inner records for {candidate}")
    configs = sorted(set(str(r["config_name"]) for r in rows))
    summary = []
    for cfg in configs:
        rr = [r for r in rows if r["config_name"] == cfg]
        summary.append({
            "config_name": cfg,
            "mean_inner_score": float(np.mean([float(r["weighted_rmse"]) for r in rr])),
            "best_iterations": [int(r["best_iteration"]) for r in rr],
        })
    winner = min(summary, key=lambda r: (r["mean_inner_score"], r["config_name"]))
    # Predeclared iteration rule: upper median of inner best iterations.
    iterations = sorted(winner["best_iterations"])
    selected_rounds = int(iterations[len(iterations) // 2])
    cfg = next(c for c in CAPACITY_CONFIGS if c["name"] == winner["config_name"])
    return {**winner, "selected_rounds": selected_rounds,
            "num_leaves": cfg["num_leaves"], "min_data_in_leaf": cfg["min_data_in_leaf"],
            "iteration_rule": "upper_median_inner_best_iteration"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--expected-commit", required=True)
    args = ap.parse_args()

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit:
        raise RuntimeError({"expected_commit": args.expected_commit, "actual_commit": head})
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Refusing to run from a dirty checkout")
    required = [args.data_dir / n for n in ("Train.csv", "Test.csv", "SampleSubmission.csv")]
    if missing := [str(p) for p in required if not p.is_file()]:
        raise FileNotFoundError(f"required CSVs missing: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {
        "status": "running", "stage": "REGIONAL_COMBINED_CAPACITY", "commit": head,
        "started_at": datetime.now(timezone.utc).isoformat(), "no_test_predictions": True,
        "widths": WIDTHS, "max_rounds": MAX_ROUNDS, "patience": PATIENCE,
        "capacity_configs": CAPACITY_CONFIGS,
    }
    _json(args.output_dir / "manifest.json", manifest)

    try:
        import lightgbm as lgb

        usecols = ["sample_id", "time", "lat", "lon", "TWS_t", "target", "month_sin", "month_cos", *HYDRO_COLUMNS]
        train = pd.read_csv(args.data_dir / "Train.csv", usecols=usecols)
        test = pd.read_csv(args.data_dir / "Test.csv",
                           usecols=["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked", *HYDRO_COLUMNS])
        template = build_test_mask_template(test[["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]])

        structural = train[["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
        source = train[list(SOURCE_HYDRO_HISTORY_COLUMNS)].copy()
        labels = train[["sample_id", "target"]].copy()
        regional_source = train[["sample_id", "time", "lat", "lon", *HYDRO_COLUMNS]].copy()
        print(json.dumps({"phase": "regional_build_start", "rows": len(regional_source), "memory": _memory()}), flush=True)
        regional = build_regional_context(regional_source, widths=WIDTHS)

        # Structural inference parity on Test source fields; no predictions are made.
        test_source = test.rename(columns={"ID": "sample_id"})[["sample_id", "time", "lat", "lon", *HYDRO_COLUMNS]].copy()
        test_regional = build_regional_context(test_source, widths=WIDTHS)
        test_period = pd.to_datetime(test_source.time).dt.to_period("M")
        test_contract = {
            "rows": int(len(test_source)), "unique_periods": int(test_period.nunique()),
            "period_range": [str(test_period.min()), str(test_period.max())],
            "hydro_finite_fraction": {c: float(np.isfinite(test_source[c].to_numpy(float)).mean()) for c in HYDRO_COLUMNS},
            "regional_schema": regional_feature_names(WIDTHS),
            "regional_hash": regional_bytes_hash(test_regional.drop(columns=["sample_id"])),
        }

        outer_structural = {}
        recent_target_union: set[pd.Period] = set()
        frozen_folds: dict[str, tuple[SimulatedFold, str]] = {}
        for origin in OUTER_ORIGINS:
            fold, role = _fold(train, template, origin)
            frozen_folds[origin] = (fold, role)
            tm = _target_months(fold)
            if origin in {"2014-04", "2014-12"}:
                recent_target_union.update(tm)
            outer_structural[origin] = {
                "rows": int(len(fold.ledger)),
                "h_counts": {str(k): int(v) for k, v in fold.ledger.h.value_counts().sort_index().items()},
                "source_months": sorted(str(p) for p in pd.to_datetime(fold.ledger.source_date).dt.to_period("M").unique()),
                "target_months": sorted(str(p) for p in tm),
                "role": role,
            }
        reserved = _reserved_transfer_block(train, recent_target_union)
        inner_folds = _choose_inner_replays(train, template)
        inner_structural = [{
            "scenario_id": f.spec.scenario_id, "rows": int(len(f.ledger)),
            "first_source": f.spec.first_source_month, "last_source": f.spec.last_source_month,
            "target_months": sorted(str(p) for p in _target_months(f)),
            "h_counts": {str(k): int(v) for k, v in f.ledger.h.value_counts().sort_index().items()},
        } for f in inner_folds]
        prefit = {"outer": outer_structural, "inner": inner_structural,
                  "recent_target_overlap": sorted(set(outer_structural["2014-04"]["target_months"]).intersection(
                      outer_structural["2014-12"]["target_months"])),
                  "reserved_transfer": reserved, "test_contract": test_contract}
        _json(args.output_dir / "structural_preflight.json", prefit)
        print(json.dumps({"phase": "STRUCTURE_FROZEN_BEFORE_SCORING", **prefit}, indent=2), flush=True)

        params_base = dict(DEFAULT_PARAMS, num_threads=min(4, os.cpu_count() or 1), seed=SEED,
                           feature_fraction_seed=SEED, bagging_seed=SEED, data_random_seed=SEED)
        inner_records: list[dict[str, object]] = []

        for fold in inner_folds:
            start = pd.Period(fold.spec.first_source_month, freq="M")
            sampled = build_sampled_training_rows(structural, source[SOURCE_CORE_COLUMNS], max_target_month=start - 1, seed=SEED)
            rows = sampled.rows
            y_train = _attach_delta(rows, labels)
            w_train = horizon_rebalance_weights(rows.h)
            x0_train = build_hydro_gap_safe_feature_matrix(rows, source, structural)
            x0_valid = build_hydro_gap_safe_feature_matrix(fold.ledger, source, structural)
            reg_train = attach_regional_features(rows.sample_id, regional)
            reg_valid = attach_regional_features(fold.ledger.sample_id, regional)
            x1_train = _feature_frame(x0_train, reg_train)
            x1_valid = _feature_frame(x0_valid, reg_valid)
            y_valid_delta = fold.labels.target.to_numpy(np.float64) - fold.ledger.last_observed_TWS.to_numpy(np.float64)
            w_valid = horizon_rebalance_weights(fold.ledger.h)

            row_hash = hashlib.sha256(rows.sample_id.astype(str).str.cat(sep="\n").encode()).hexdigest()
            label_hash = hashlib.sha256(np.ascontiguousarray(y_train).tobytes()).hexdigest()
            for candidate, xt, xv in (("C0_safe", x0_train, x0_valid), ("C1_regional_5plus15", x1_train, x1_valid)):
                for cfg in CAPACITY_CONFIGS:
                    params = dict(params_base, num_leaves=cfg["num_leaves"], min_data_in_leaf=cfg["min_data_in_leaf"])
                    evals_result = {}
                    booster = lgb.train(
                        params,
                        lgb.Dataset(xt, label=y_train, weight=w_train, feature_name=list(xt.columns)),
                        num_boost_round=MAX_ROUNDS,
                        valid_sets=[lgb.Dataset(xv, label=y_valid_delta, weight=w_valid, reference=None)],
                        valid_names=["inner_valid"],
                        callbacks=[lgb.early_stopping(PATIENCE, verbose=False), lgb.record_evaluation(evals_result), lgb.log_evaluation(0)],
                    )
                    best_iter = int(booster.best_iteration or MAX_ROUNDS)
                    delta = booster.predict(xv, num_iteration=best_iter)
                    pred = fold.ledger.last_observed_TWS.to_numpy(np.float64) + delta
                    y = fold.labels.target.to_numpy(np.float64)
                    score = _weighted_rmse_from_fold(y, pred, fold.ledger.h)
                    rec = {
                        "candidate": candidate, "config_name": cfg["name"], "scenario_id": fold.spec.scenario_id,
                        "best_iteration": best_iter, "weighted_rmse": score, "training_rows": int(len(rows)),
                        "validation_rows": int(len(fold.ledger)), "row_hash": row_hash, "label_hash": label_hash,
                        "regional_train_hash": regional_bytes_hash(reg_train) if candidate.startswith("C1") else None,
                        "regional_valid_hash": regional_bytes_hash(reg_valid) if candidate.startswith("C1") else None,
                        "learning_curve": evals_result.get("inner_valid", {}).get("rmse", []),
                    }
                    inner_records.append(rec)
                    booster.free_dataset()
                    _json(args.output_dir / "inner_capacity.partial.json", {"records": inner_records})
                    print(json.dumps({"phase": "inner_fit_complete", **{k: rec[k] for k in (
                        "candidate", "config_name", "scenario_id", "best_iteration", "weighted_rmse", "training_rows", "validation_rows")}},
                        indent=2), flush=True)

        selections = {
            "C0_safe": _select_capacity(inner_records, "C0_safe"),
            "C1_regional_5plus15": _select_capacity(inner_records, "C1_regional_5plus15"),
        }
        _json(args.output_dir / "inner_capacity.json", {"records": inner_records, "selections": selections})
        print(json.dumps({"phase": "capacity_frozen_before_outer_scores", "selections": selections}, indent=2), flush=True)

        outer_results: list[dict[str, object]] = []
        for origin in OUTER_ORIGINS:
            fold, role = frozen_folds[origin]
            start = pd.Period(origin, freq="M")
            sampled = build_sampled_training_rows(structural, source[SOURCE_CORE_COLUMNS], max_target_month=start - 1, seed=SEED)
            rows = sampled.rows
            y_train = _attach_delta(rows, labels)
            w_train = horizon_rebalance_weights(rows.h)
            x0_train = build_hydro_gap_safe_feature_matrix(rows, source, structural)
            x0_valid = build_hydro_gap_safe_feature_matrix(fold.ledger, source, structural)
            reg_train = attach_regional_features(rows.sample_id, regional)
            reg_valid = attach_regional_features(fold.ledger.sample_id, regional)
            x1_train = _feature_frame(x0_train, reg_train)
            x1_valid = _feature_frame(x0_valid, reg_valid)
            row_hash = hashlib.sha256(rows.sample_id.astype(str).str.cat(sep="\n").encode()).hexdigest()
            label_hash = hashlib.sha256(np.ascontiguousarray(y_train).tobytes()).hexdigest()
            coverage_hash = hashlib.sha256(fold.ledger.sample_id.astype(str).str.cat(sep="\n").encode()).hexdigest()

            recipes = [
                ("C0_frozen173", x0_train, x0_valid, {"num_leaves": 63, "min_data_in_leaf": 1000}, 173),
                ("C0_capacity_selected", x0_train, x0_valid, selections["C0_safe"], selections["C0_safe"]["selected_rounds"]),
                ("C1_regional_matched_capacity", x1_train, x1_valid, selections["C0_safe"], selections["C0_safe"]["selected_rounds"]),
            ]
            for name, xt, xv, cfg, rounds in recipes:
                params = dict(params_base, num_leaves=int(cfg["num_leaves"]), min_data_in_leaf=int(cfg["min_data_in_leaf"]))
                fit_started = time.perf_counter()
                booster = lgb.train(params, lgb.Dataset(xt, label=y_train, weight=w_train, feature_name=list(xt.columns)),
                                    num_boost_round=int(rounds), callbacks=[lgb.log_evaluation(0)])
                delta = booster.predict(xv)
                pred = fold.ledger.last_observed_TWS.to_numpy(np.float64) + delta
                score, oof = _score(name, fold, role, pred)
                score.update({
                    "origin": origin, "candidate": name, "rounds": int(rounds),
                    "num_leaves": int(cfg["num_leaves"]), "min_data_in_leaf": int(cfg["min_data_in_leaf"]),
                    "training_rows": int(len(rows)), "training_row_hash": row_hash, "training_label_hash": label_hash,
                    "outer_coverage_hash": coverage_hash, "feature_count": int(xt.shape[1]),
                    "feature_names": list(xt.columns), "elapsed_seconds": time.perf_counter() - fit_started,
                })
                outer_results.append(score)
                oof["training_row_hash"] = row_hash
                oof.to_csv(args.output_dir / f"oof_{origin}_{name}.csv.gz", index=False, compression="gzip")
                booster.save_model(str(args.output_dir / f"model_{origin}_{name}.txt"))
                booster.free_dataset()
                _json(args.output_dir / "outer_metrics.partial.json", {"results": outer_results})
                print(json.dumps({"phase": "outer_fit_complete", "origin": origin, "candidate": name,
                                  "raw_rmse": score["raw_rmse"], "weighted_rmse": score["weighted_rmse"],
                                  "rounds": rounds, "training_rows": len(rows), "memory": _memory()}, indent=2), flush=True)

        # Paired effect table makes regional-information and capacity effects explicit.
        effect_rows = []
        by = {(r["origin"], r["candidate"]): r for r in outer_results}
        for origin in OUTER_ORIGINS:
            frozen = by[(origin, "C0_frozen173")]
            c0 = by[(origin, "C0_capacity_selected")]
            c1 = by[(origin, "C1_regional_matched_capacity")]
            effect_rows.append({
                "origin": origin,
                "capacity_gain_raw_rmse": float(frozen["raw_rmse"] - c0["raw_rmse"]),
                "regional_gain_at_matched_selected_capacity_raw_rmse": float(c0["raw_rmse"] - c1["raw_rmse"]),
                "frozen173_raw_rmse": frozen["raw_rmse"], "c0_selected_raw_rmse": c0["raw_rmse"],
                "c1_matched_capacity_raw_rmse": c1["raw_rmse"],
            })
        _json(args.output_dir / "metrics.json", {"results": outer_results, "effects": effect_rows,
                                                  "selections": selections, "structure": prefit})

        expected = len(OUTER_ORIGINS) * 3
        if len(outer_results) != expected:
            raise AssertionError(f"expected {expected} outer results, got {len(outer_results)}")
        manifest.update({
            "status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started, "outer_result_rows": len(outer_results),
            "selections": selections, "reserved_transfer": reserved,
            "output_checksums": {str(p.relative_to(args.output_dir)): _sha256(p)
                                 for p in args.output_dir.rglob("*") if p.is_file() and p.name != "manifest.json"},
        })
        _json(args.output_dir / "manifest.json", manifest)
        print(json.dumps({"status": "completed", "stage": "REGIONAL_COMBINED_CAPACITY",
                          "results": len(outer_results), "output_dir": str(args.output_dir)}, indent=2), flush=True)
    except Exception as exc:
        manifest.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(),
                         "elapsed_seconds": time.perf_counter() - started,
                         "error": f"{type(exc).__name__}: {exc}"})
        _json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
