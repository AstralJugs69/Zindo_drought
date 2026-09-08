"""Causal B2 dynamic-factor screen (Kaggle only; no Test predictions).

The target factor change is never part of the predictor matrix.  Bases, means,
hydrology centering, inner penalty selection, and ridge fits are all rebuilt
inside each outer fitting prefix.  Unsupported evaluation rows fall back to the
frozen R01 OOF prediction rather than reading a masked dense TWS field.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
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

from scripts.run_experiment import _score
from src.observation_simulator import SimulatedFold, build_mask_block_fold, build_template_replay_fold
from src.spatial_b2 import (
    ALPHAS,
    DEFAULT_HORIZONS,
    HYDRO_COLUMNS,
    build_training_episodes,
    fit_per_factor_ridge,
    fit_spatial_basis,
    predict_with_fallback,
    prefix_substantial_factors,
    save_bundle,
    select_alpha_chronologically,
    visible_eval_factors,
)
from src.validation import build_test_mask_template

ORIGINS = ("2007-09", "2009-01", "2014-12")
RANKS = (8, 16)
MIN_TWS_COVERAGE = 0.95
MIN_HYDRO_COVERAGE = 0.50


def _json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _memory() -> dict[str, object]:
    try:
        import psutil
        p = psutil.Process()
        vm = psutil.virtual_memory()
        return {"rss_gib": p.memory_info().rss / (1024**3), "available_gib": vm.available / (1024**3)}
    except Exception as exc:
        return {"note": f"memory unavailable: {type(exc).__name__}: {exc}"}


def _find_oof(run_dir: Path, origin: str) -> Path:
    matches = sorted({p.resolve() for pattern in (
        f"oof_*{origin}*r01_lgbm.csv.gz", f"ooof_*{origin}*r01_lgbm.csv.gz",
        f"*{origin}*r01_lgbm*.csv.gz",
    ) for p in run_dir.rglob(pattern)})
    if len(matches) != 1:
        raise FileNotFoundError({"origin": origin, "matches": [str(p) for p in matches], "run_dir": str(run_dir)})
    return matches[0]


def _fold(train: pd.DataFrame, template: pd.DataFrame, origin: str) -> tuple[SimulatedFold, str]:
    if origin == "2014-12":
        full = build_mask_block_fold(train, anchor_month=origin, end_month="2015-06",
                                     scenario_id="b2_recent_2014_12", family="recent_mask_block")
        keep = full.ledger["h"].between(1, 7).to_numpy()
        fold = SimulatedFold(
            full.spec,
            full.ledger.loc[keep].reset_index(drop=True),
            full.labels.loc[keep].reset_index(drop=True),
            dict(full.exclusions, outside_competition_horizon=int((~keep).sum())),
        )
        return fold, "recent_stress"
    return build_template_replay_fold(
        train, template, start_month=origin, scenario_id=f"b2_{origin}", family="development"
    ), "development"


def _align_r01(fold: SimulatedFold, oof_path: Path) -> np.ndarray:
    oof = pd.read_csv(oof_path)
    needed = {"sample_id", "prediction"}
    if missing := needed.difference(oof.columns):
        raise ValueError(f"R01 OOF {oof_path} missing {sorted(missing)}")
    oof = oof.loc[oof.get("h", pd.Series(np.ones(len(oof)))).between(1, 7)].copy()
    if oof["sample_id"].duplicated().any():
        raise AssertionError(f"R01 OOF contains duplicate sample_id: {oof_path}")
    aligned = fold.ledger.loc[:, ["sample_id"]].merge(
        oof.loc[:, ["sample_id", "prediction"]], on="sample_id", how="left", validate="one_to_one", sort=False
    )
    if aligned["prediction"].isna().any():
        raise AssertionError(f"R01 OOF missing {int(aligned.prediction.isna().sum())} fold rows")
    return aligned["prediction"].to_numpy(np.float64)


def _inner_cutoffs(train: pd.DataFrame, outer_cutoff: pd.Period) -> list[pd.Period]:
    periods = sorted(pd.to_datetime(train["time"]).dt.to_period("M").unique())
    eligible = [p for p in periods if p + 7 < outer_cutoff]
    if len(eligible) < 24:
        raise AssertionError(f"too few inner-prefix months before {outer_cutoff}: {len(eligible)}")
    picks = []
    for frac in (0.62, 0.82):
        idx = min(len(eligible) - 1, max(12, int(frac * len(eligible))))
        p = eligible[idx]
        if p not in picks:
            picks.append(p)
    if len(picks) < 2:
        raise AssertionError("need at least two chronological inner cutoffs")
    return picks


def _select_alpha_nested(train: pd.DataFrame, source: pd.DataFrame, *, outer_cutoff: pd.Period,
                         rank: int) -> tuple[float, list[dict[str, object]]]:
    """Select alpha with basis/means/scalers rebuilt inside each inner prefix."""
    rows: list[dict[str, object]] = []
    for cutoff in _inner_cutoffs(train, outer_cutoff):
        basis = fit_spatial_basis(train, cutoff=cutoff, rank=rank)
        episodes = build_training_episodes(
            train, basis, cutoff=cutoff, horizons=DEFAULT_HORIZONS,
            min_tws_coverage=MIN_TWS_COVERAGE, min_hydro_coverage=MIN_HYDRO_COVERAGE,
        )
        full = build_mask_block_fold(
            train, anchor_month=cutoff, end_month=cutoff + 6,
            scenario_id=f"b2_inner_{cutoff}", family="inner_capacity_selection",
        )
        keep = full.ledger["h"].between(1, 7).to_numpy()
        fold = SimulatedFold(
            full.spec, full.ledger.loc[keep].reset_index(drop=True),
            full.labels.loc[keep].reset_index(drop=True),
            dict(full.exclusions, outside_competition_horizon=int((~keep).sum())),
        )
        # Inner validation labels must remain strictly inside the outer fitting prefix.
        if not (pd.to_datetime(fold.ledger["target_date"]).dt.to_period("M") < outer_cutoff).all():
            raise AssertionError(f"inner fold {cutoff} reaches outer evaluation labels")
        prefix = prefix_substantial_factors(train, basis, cutoff=cutoff, min_coverage=MIN_TWS_COVERAGE)
        for alpha in ALPHAS:
            model = fit_per_factor_ridge(episodes, alpha=float(alpha))
            dummy = np.zeros(len(fold.ledger), dtype=np.float64)
            pred, provenance = predict_with_fallback(
                ledger=fold.ledger, source=source, basis=basis, model=model,
                prefix_factors=prefix, fallback_prediction=dummy,
                min_tws_coverage=MIN_TWS_COVERAGE, min_hydro_coverage=MIN_HYDRO_COVERAGE,
            )
            support = provenance["supported"].to_numpy(bool)
            if not support.any():
                raise AssertionError(f"inner fold {cutoff} rank{rank} has zero supported rows")
            y = fold.labels["target"].to_numpy(np.float64)
            mse = float(np.mean(np.square(pred[support] - y[support])))
            rows.append({
                "cutoff": str(cutoff), "alpha": float(alpha), "rank_used": basis.rank,
                "train_episodes": episodes.n,
                "distinct_train_sources": int(len(np.unique(episodes.source_period))),
                "valid_rows": int(len(fold.ledger)), "supported_rows": int(support.sum()),
                "support_fraction": float(support.mean()), "mse": mse,
            })
    means = {float(a): float(np.mean([r["mse"] for r in rows if r["alpha"] == float(a)])) for a in ALPHAS}
    best = min((v, -a, a) for a, v in means.items())[2]
    return float(best), rows

def _substantial_state_report(fold: SimulatedFold, factors: dict[pd.Period, tuple[np.ndarray, float]]) -> list[dict[str, object]]:
    source_periods = pd.to_datetime(fold.ledger["source_date"]).dt.to_period("M")
    rows = []
    available = sorted(factors)
    for p in sorted(source_periods.unique()):
        prior = [a for a in available if a <= p]
        last = prior[-1] if prior else None
        rows.append({
            "source_period": str(p), "latest_substantial_period": str(last) if last is not None else None,
            "global_field_age_months": int(p.ordinal - last.ordinal) if last is not None else None,
            "coverage": float(factors[last][1]) if last is not None else None,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True, help="Completed R01/local-response artifact root")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--expected-commit", required=True)
    ap.add_argument("--origins", default=",".join(ORIGINS))
    ap.add_argument("--ranks", default=",".join(map(str, RANKS)))
    args = ap.parse_args()

    origins = tuple(x.strip() for x in args.origins.split(",") if x.strip())
    ranks = tuple(int(x) for x in args.ranks.split(",") if x.strip())
    if not origins or not set(origins).issubset(ORIGINS):
        raise ValueError(f"origins must be subset of {ORIGINS}")
    if not ranks or not set(ranks).issubset(RANKS):
        raise ValueError(f"ranks must be subset of {RANKS}")

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit:
        raise RuntimeError({"expected_commit": args.expected_commit, "actual_commit": head})
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Refusing to run B2 from a dirty checkout")
    required = [args.data_dir / n for n in ("Train.csv", "Test.csv", "SampleSubmission.csv")]
    missing_files = [str(p) for p in required if not p.is_file()]
    if missing_files:
        raise FileNotFoundError(f"required competition files missing: {missing_files}")
    oof_paths = {origin: _find_oof(args.run_dir, origin) for origin in origins}

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest: dict[str, object] = {
        "status": "running", "stage": "B2_CAUSAL_DYNAMIC_FACTOR", "commit": head,
        "started_at": datetime.now(timezone.utc).isoformat(), "origins": origins, "ranks": ranks,
        "min_tws_coverage": MIN_TWS_COVERAGE, "min_hydro_coverage": MIN_HYDRO_COVERAGE,
        "alphas": ALPHAS, "no_test_predictions": True,
        "oof_inputs": {k: {"path": str(v), "sha256": _sha256(v)} for k, v in oof_paths.items()},
    }
    _json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"phase": "preflight", "commit": head, "origins": origins, "ranks": ranks,
                      "memory": _memory()}, indent=2), flush=True)

    try:
        usecols = ["sample_id", "time", "lat", "lon", "TWS_t", "target", "month_sin", "month_cos", *HYDRO_COLUMNS]
        train = pd.read_csv(args.data_dir / "Train.csv", usecols=usecols)
        test = pd.read_csv(args.data_dir / "Test.csv", usecols=["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"])
        template = build_test_mask_template(test)
        source = train.loc[:, ["sample_id", "time", "lat", "lon", "month_sin", "month_cos", *HYDRO_COLUMNS]].copy()
        metrics: list[dict[str, object]] = []
        state_reports: dict[str, object] = {}

        for origin in origins:
            outer_started = time.perf_counter()
            cutoff = pd.Period(origin, freq="M")
            fold, role = _fold(train, template, origin)
            fallback = _align_r01(fold, oof_paths[origin])
            print(json.dumps({"phase": "outer_start", "origin": origin, "rows": len(fold.ledger),
                              "h_counts": {str(k): int(v) for k, v in fold.ledger.h.value_counts().sort_index().items()},
                              "memory": _memory()}, indent=2), flush=True)

            for rank in ranks:
                rank_started = time.perf_counter()
                basis = fit_spatial_basis(train, cutoff=cutoff, rank=rank)
                episodes = build_training_episodes(
                    train, basis, cutoff=cutoff, horizons=DEFAULT_HORIZONS,
                    min_tws_coverage=MIN_TWS_COVERAGE, min_hydro_coverage=MIN_HYDRO_COVERAGE,
                )
                distinct_sources = int(len(np.unique(episodes.source_period)))
                distinct_targets = int(len(np.unique(episodes.target_period)))
                if distinct_sources < 16:
                    raise AssertionError(f"{origin} rank{rank}: too few independent source fields: {distinct_sources}")
                alpha, inner = _select_alpha_nested(train, source, outer_cutoff=cutoff, rank=rank)
                model = fit_per_factor_ridge(episodes, alpha=alpha)
                parameter_count = int(model.coef.size + model.intercept.size)
                if parameter_count >= episodes.n * basis.rank:
                    # This is not a hard statistical theorem, but catches accidental huge VAR-like growth.
                    print(json.dumps({"warning": "parameter_count_large_relative_to_scalar_outcomes",
                                      "parameters": parameter_count, "episodes": episodes.n,
                                      "rank": basis.rank}), flush=True)

                prefix = prefix_substantial_factors(train, basis, cutoff=cutoff, min_coverage=MIN_TWS_COVERAGE)
                eval_visible = visible_eval_factors(fold.ledger, basis, min_coverage=MIN_TWS_COVERAGE)
                all_factors = dict(prefix); all_factors.update(eval_visible)
                pred, provenance = predict_with_fallback(
                    ledger=fold.ledger, source=source, basis=basis, model=model,
                    prefix_factors=prefix, fallback_prediction=fallback,
                    min_tws_coverage=MIN_TWS_COVERAGE, min_hydro_coverage=MIN_HYDRO_COVERAGE,
                )
                score, oof = _score(f"B2_rank{rank}", fold, role, pred)
                r01_score, _ = _score("R01_fallback", fold, role, fallback)
                support = provenance["supported"].to_numpy(bool)
                reasons = provenance.loc[~support, "fallback_reason"].value_counts().to_dict()
                score.update({
                    "origin": origin, "rank_requested": rank, "rank_used": basis.rank,
                    "basis_explained_variance": basis.explained_variance,
                    "alpha": alpha, "inner_scores": inner,
                    "training_episodes": episodes.n, "distinct_source_fields": distinct_sources,
                    "distinct_target_fields": distinct_targets,
                    "episode_source_ordinal_min": int(episodes.source_period.min()),
                    "episode_source_ordinal_max": int(episodes.source_period.max()),
                    "parameter_count": parameter_count,
                    "support_rows": int(support.sum()), "support_fraction": float(support.mean()),
                    "fallback_rows": int((~support).sum()), "fallback_reasons": {str(k): int(v) for k, v in reasons.items()},
                    "r01_raw_rmse": r01_score["raw_rmse"], "r01_weighted_rmse": r01_score["weighted_rmse"],
                    "elapsed_seconds": time.perf_counter() - rank_started,
                })
                metrics.append(score)

                oof = pd.concat([oof.reset_index(drop=True), provenance.reset_index(drop=True)], axis=1)
                oof["r01_prediction"] = fallback
                oof.to_csv(args.output_dir / f"oof_{origin}_b2_rank{rank}.csv.gz", index=False, compression="gzip")
                provenance.to_csv(args.output_dir / f"provenance_{origin}_rank{rank}.csv.gz", index=False, compression="gzip")
                state_reports[f"{origin}:rank{rank}"] = _substantial_state_report(fold, all_factors)
                save_bundle(args.output_dir / f"model_{origin}_rank{rank}", basis, model, {
                    "origin": origin, "rank_requested": rank, "cutoff": str(cutoff),
                    "alpha_selection": inner, "training_episodes": episodes.n,
                    "distinct_source_fields": distinct_sources, "distinct_target_fields": distinct_targets,
                    "parameter_count": parameter_count, "min_tws_coverage": MIN_TWS_COVERAGE,
                    "min_hydro_coverage": MIN_HYDRO_COVERAGE,
                    "inference_equation": "anchor_tws_i + U_i @ predicted_factor_change(anchor->target); R01 fallback unless exact row-anchor factor is from a >=95% visible global field",
                })
                _json(args.output_dir / "metrics.partial.json", {"results": metrics})
                _json(args.output_dir / "substantial_state.partial.json", state_reports)
                print(json.dumps({"phase": "rank_complete", "origin": origin, "rank": rank,
                                  "raw_rmse": score["raw_rmse"], "weighted_rmse": score["weighted_rmse"],
                                  "r01_raw_rmse": r01_score["raw_rmse"], "support_fraction": float(support.mean()),
                                  "alpha": alpha, "training_episodes": episodes.n,
                                  "distinct_source_fields": distinct_sources, "parameters": parameter_count,
                                  "elapsed_seconds": time.perf_counter()-rank_started, "memory": _memory()}, indent=2), flush=True)
            print(json.dumps({"phase": "outer_complete", "origin": origin,
                              "elapsed_seconds": time.perf_counter()-outer_started}), flush=True)

        _json(args.output_dir / "metrics.json", {"results": metrics})
        _json(args.output_dir / "substantial_state.json", state_reports)
        expected_outputs = len(origins) * len(ranks)
        if len(metrics) != expected_outputs:
            raise AssertionError(f"expected {expected_outputs} metric rows, got {len(metrics)}")
        expected_oof = [args.output_dir / f"oof_{o}_b2_rank{r}.csv.gz" for o in origins for r in ranks]
        absent = [str(p) for p in expected_oof if not p.is_file()]
        if absent:
            raise AssertionError(f"missing expected OOF outputs: {absent}")
        manifest.update({
            "status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "result_rows": len(metrics), "expected_result_rows": expected_outputs,
            "output_checksums": {str(p.relative_to(args.output_dir)): _sha256(p)
                                 for p in args.output_dir.rglob("*") if p.is_file() and p.name != "manifest.json"},
        })
        _json(args.output_dir / "manifest.json", manifest)
        print(json.dumps({"status": "completed", "stage": "B2_CAUSAL_DYNAMIC_FACTOR",
                          "results": len(metrics), "output_dir": str(args.output_dir)}, indent=2), flush=True)
    except Exception as exc:
        manifest.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(),
                         "elapsed_seconds": time.perf_counter() - started,
                         "error": f"{type(exc).__name__}: {exc}"})
        _json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
