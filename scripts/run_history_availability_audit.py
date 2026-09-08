"""Label-free availability and distribution audit for the regional B0 contract.

This script deliberately creates no competition predictions.  It compares exactly
the target-blind B0 features constructed from historical replay ledgers with the
same construction from the supplied Test availability ledger.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.availability import build_test_availability_ledger
from src.ml_features import SOURCE_HYDRO_HISTORY_COLUMNS, build_hydro_gap_safe_feature_matrix
from src.observation_simulator import build_mask_block_fold, build_template_replay_fold
from src.regional_context import HYDRO_COLUMNS, attach_regional_features, build_regional_context
from src.validation import build_test_mask_template

SEED = 20260908
OUTER_ORIGINS = ("2007-09", "2009-01", "2014-04", "2014-12")


def _json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fold(train: pd.DataFrame, template: pd.DataFrame, origin: str):
    if origin == "2014-04":
        return build_mask_block_fold(
            train, anchor_month=origin, end_month="2014-10",
            scenario_id="history_audit_2014_04", family="recent_mask_block",
        )
    if origin == "2014-12":
        full = build_mask_block_fold(
            train, anchor_month=origin, end_month="2015-06",
            scenario_id="history_audit_2014_12", family="recent_mask_block",
        )
        keep = full.ledger.h.between(1, 7).to_numpy()
        from src.observation_simulator import SimulatedFold
        return SimulatedFold(
            full.spec,
            full.ledger.loc[keep].reset_index(drop=True),
            full.labels.loc[keep].reset_index(drop=True),
            dict(full.exclusions, outside_competition_horizon=int((~keep).sum())),
        )
    return build_template_replay_fold(
        train, template, start_month=origin,
        scenario_id=f"history_audit_{origin}", family="development",
    )


def _build_b0(ledger: pd.DataFrame, source: pd.DataFrame, structural: pd.DataFrame, regional: pd.DataFrame) -> pd.DataFrame:
    base = build_hydro_gap_safe_feature_matrix(ledger, source, structural)
    context = attach_regional_features(ledger.sample_id, regional)
    out = pd.concat([base.reset_index(drop=True), context.reset_index(drop=True)], axis=1)
    if out.columns.duplicated().any():
        raise AssertionError("B0 feature schema contains duplicate names")
    return out


def _metadata(ledger: pd.DataFrame, domain: str) -> pd.DataFrame:
    x = ledger.loc[:, ["sample_id", "source_date", "last_observed_date", "lat", "lon", "h"]].copy()
    source = pd.to_datetime(x.source_date).dt.to_period("M")
    anchor = pd.to_datetime(x.last_observed_date).dt.to_period("M")
    x["domain"] = domain
    x["season"] = pd.to_datetime(x.source_date).dt.month.astype(np.int8)
    x["source_month"] = source.astype(str)
    x["anchor_age_months"] = (source.astype("int64") - anchor.astype("int64")).astype(np.int16)
    x["geo5"] = (
        (np.floor(x.lat.astype(float) / 5.0) * 5.0).astype(int).astype(str)
        + ":" + (np.floor(x.lon.astype(float) / 5.0) * 5.0).astype(int).astype(str)
    )
    return x.reset_index(drop=True)


def _categorical_gap(hist: pd.Series, test: pd.Series, name: str) -> dict[str, object]:
    left = hist.astype(str).value_counts(normalize=True)
    right = test.astype(str).value_counts(normalize=True)
    keys = left.index.union(right.index)
    delta = (left.reindex(keys, fill_value=0.0) - right.reindex(keys, fill_value=0.0)).abs()
    return {
        "field": name,
        "total_variation": float(delta.sum() / 2.0),
        "hist_categories": int(len(left)),
        "test_categories": int(len(right)),
        "test_mass_unseen_in_history": float(right.loc[right.index.difference(left.index)].sum()),
    }


def _feature_discrepancies(hist: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in hist.columns:
        a = hist[column].to_numpy(dtype=np.float64)
        b = test[column].to_numpy(dtype=np.float64)
        a_finite, b_finite = a[np.isfinite(a)], b[np.isfinite(b)]
        scale = float(np.std(a_finite)) if len(a_finite) else np.nan
        mean_gap = float(np.mean(b_finite) - np.mean(a_finite)) if len(a_finite) and len(b_finite) else np.nan
        rows.append({
            "feature": column,
            "hist_missing_share": float(1.0 - len(a_finite) / len(a)),
            "test_missing_share": float(1.0 - len(b_finite) / len(b)),
            "missing_share_gap": float((1.0 - len(b_finite) / len(b)) - (1.0 - len(a_finite) / len(a))),
            "hist_mean": float(np.mean(a_finite)) if len(a_finite) else np.nan,
            "test_mean": float(np.mean(b_finite)) if len(b_finite) else np.nan,
            "hist_std": scale,
            "standardized_mean_gap": float(mean_gap / max(scale, 1e-8)) if np.isfinite(mean_gap) else np.nan,
            "hist_q05": float(np.quantile(a_finite, 0.05)) if len(a_finite) else np.nan,
            "test_q05": float(np.quantile(b_finite, 0.05)) if len(b_finite) else np.nan,
            "hist_q95": float(np.quantile(a_finite, 0.95)) if len(a_finite) else np.nan,
            "test_q95": float(np.quantile(b_finite, 0.95)) if len(b_finite) else np.nan,
        })
    out = pd.DataFrame(rows)
    out["rank_score"] = out.missing_share_gap.abs().fillna(0.0) + out.standardized_mean_gap.abs().fillna(0.0)
    return out.sort_values(["rank_score", "feature"], ascending=[False, True]).reset_index(drop=True)


def _balanced_broad(hist_x: pd.DataFrame, hist_m: pd.DataFrame, test_x: pd.DataFrame, test_m: pd.DataFrame, limit: int, rng: np.random.Generator):
    n = min(limit, len(hist_x), len(test_x))
    h_idx = rng.choice(len(hist_x), size=n, replace=False)
    t_idx = rng.choice(len(test_x), size=n, replace=False)
    x = pd.concat([hist_x.iloc[h_idx], test_x.iloc[t_idx]], ignore_index=True)
    m = pd.concat([hist_m.iloc[h_idx], test_m.iloc[t_idx]], ignore_index=True)
    y = np.concatenate([np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)])
    return x, m, y


def _matched_sample(hist_x: pd.DataFrame, hist_m: pd.DataFrame, test_x: pd.DataFrame, test_m: pd.DataFrame, limit: int, rng: np.random.Generator):
    key_columns = ["h", "season", "geo5"]
    h = hist_m.loc[:, key_columns].copy(); h["_row"] = np.arange(len(h))
    t = test_m.loc[:, key_columns].copy(); t["_row"] = np.arange(len(t))
    groups_h = h.groupby(key_columns, observed=True)["_row"].agg(list)
    groups_t = t.groupby(key_columns, observed=True)["_row"].agg(list)
    hist_rows: list[np.ndarray] = []
    test_rows: list[np.ndarray] = []
    for key in groups_h.index.intersection(groups_t.index):
        left, right = np.asarray(groups_h.loc[key]), np.asarray(groups_t.loc[key])
        count = min(len(left), len(right))
        hist_rows.append(rng.choice(left, size=count, replace=False))
        test_rows.append(rng.choice(right, size=count, replace=False))
    if not hist_rows:
        raise AssertionError("No overlapping horizon/season/geography strata for matched audit")
    h_idx, t_idx = np.concatenate(hist_rows), np.concatenate(test_rows)
    if len(h_idx) > limit:
        keep = rng.choice(len(h_idx), size=limit, replace=False)
        h_idx, t_idx = h_idx[keep], t_idx[keep]
    x = pd.concat([hist_x.iloc[h_idx], test_x.iloc[t_idx]], ignore_index=True)
    m = pd.concat([hist_m.iloc[h_idx], test_m.iloc[t_idx]], ignore_index=True)
    y = np.concatenate([np.zeros(len(h_idx), dtype=np.int8), np.ones(len(t_idx), dtype=np.int8)])
    meta = {
        "matching_keys": key_columns,
        "matched_rows_per_domain": int(len(h_idx)),
        "hist_rows_excluded_for_no_test_stratum": int(len(hist_m) - len(np.concatenate(hist_rows))),
        "test_rows_excluded_for_no_history_stratum": int(len(test_m) - len(np.concatenate(test_rows))),
    }
    return x, m, y, meta


def _domain_auc(x: pd.DataFrame, meta: pd.DataFrame, y: np.ndarray) -> dict[str, object]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for the predeclared domain diagnostic") from exc

    forbidden = {"lat", "lon"}
    columns = [c for c in x.columns if c not in forbidden]
    groups = meta.geo5.to_numpy()
    splitter = GroupKFold(n_splits=5)
    scores: list[float] = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups), start=1):
        if len(np.unique(y[valid_idx])) < 2:
            raise AssertionError(f"Spatial domain fold {fold} has only one domain class")
        clf = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.08, max_leaf_nodes=31,
            min_samples_leaf=100, l2_regularization=1.0, random_state=SEED,
        )
        clf.fit(x.iloc[train_idx][columns], y[train_idx])
        scores.append(float(roc_auc_score(y[valid_idx], clf.predict_proba(x.iloc[valid_idx][columns])[:, 1])))
    return {
        "rows": int(len(x)), "features": columns, "spatial_group_count": int(pd.Series(groups).nunique()),
        "fold_auc": scores, "mean_auc": float(np.mean(scores)), "std_auc": float(np.std(scores, ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a target-blind B0-vs-Test availability audit on Kaggle.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--max-domain-rows", type=int, default=150_000)
    args = parser.parse_args()

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit:
        raise RuntimeError({"expected_commit": args.expected_commit, "actual_commit": head})
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Refusing to audit from a dirty checkout")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {"status": "running", "stage": "HISTORY_AVAILABILITY_AUDIT", "commit": head,
                "started_at": datetime.now(timezone.utc).isoformat(), "no_test_predictions": True,
                "seed": SEED, "max_domain_rows": args.max_domain_rows}
    _json(args.output_dir / "manifest.json", manifest)

    try:
        train_cols = ["sample_id", "time", "lat", "lon", "TWS_t", "target", "month_sin", "month_cos", *HYDRO_COLUMNS]
        test_cols = ["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked", *HYDRO_COLUMNS]
        train = pd.read_csv(args.data_dir / "Train.csv", usecols=train_cols)
        test = pd.read_csv(args.data_dir / "Test.csv", usecols=test_cols)
        template = build_test_mask_template(test[["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]])
        structural = train[["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
        source = train.loc[:, SOURCE_HYDRO_HISTORY_COLUMNS].copy()
        regional = build_regional_context(train[["sample_id", "time", "lat", "lon", *HYDRO_COLUMNS]])

        hist_xs: list[pd.DataFrame] = []
        hist_ms: list[pd.DataFrame] = []
        replay_rows: list[dict[str, object]] = []
        for origin in OUTER_ORIGINS:
            fold = _fold(train, template, origin)
            x = _build_b0(fold.ledger, source, structural, regional)
            m = _metadata(fold.ledger, "historical")
            hist_xs.append(x); hist_ms.append(m)
            replay_rows.append({"origin": origin, "rows": int(len(m)), "h_counts": {str(k): int(v) for k, v in m.h.value_counts().sort_index().items()}})
        hist_x = pd.concat(hist_xs, ignore_index=True)
        hist_m = pd.concat(hist_ms, ignore_index=True)

        test_ledger = build_test_availability_ledger(test[["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"]]).rename(columns={"ID": "sample_id"})
        test_source = test.rename(columns={"ID": "sample_id"})[["sample_id", "time", "lat", "lon", *HYDRO_COLUMNS]].copy()
        test_month = pd.to_datetime(test_source.time).dt.month.to_numpy(dtype=np.float64)
        test_source["month_sin"] = np.sin(2.0 * np.pi * test_month / 12.0).astype(np.float32)
        test_source["month_cos"] = np.cos(2.0 * np.pi * test_month / 12.0).astype(np.float32)
        test_structural = test.rename(columns={"ID": "sample_id"})[["sample_id", "time", "lat", "lon", "TWS_t"]].copy()
        test_regional = build_regional_context(test_source)
        test_x = _build_b0(test_ledger, test_source, test_structural, test_regional)
        test_m = _metadata(test_ledger, "test")
        if hist_x.columns.tolist() != test_x.columns.tolist():
            raise AssertionError("Historical/Test B0 feature schemas differ")

        discrepancies = _feature_discrepancies(hist_x, test_x)
        discrepancies.to_csv(args.output_dir / "feature_discrepancies.csv", index=False)
        categorical = pd.DataFrame([
            _categorical_gap(hist_m.h, test_m.h, "horizon"),
            _categorical_gap(hist_m.season, test_m.season, "season"),
            _categorical_gap(hist_m.geo5, test_m.geo5, "geo5"),
            _categorical_gap(hist_m.anchor_age_months, test_m.anchor_age_months, "anchor_age_months"),
        ])
        categorical.to_csv(args.output_dir / "metadata_discrepancies.csv", index=False)

        rng = np.random.default_rng(SEED)
        broad_x, broad_m, broad_y = _balanced_broad(hist_x, hist_m, test_x, test_m, args.max_domain_rows, rng)
        matched_x, matched_m, matched_y, matching = _matched_sample(hist_x, hist_m, test_x, test_m, args.max_domain_rows, rng)
        broad = _domain_auc(broad_x, broad_m, broad_y)
        matched = _domain_auc(matched_x, matched_m, matched_y)

        report = {
            "status": "completed", "historical_replays": replay_rows,
            "historical_rows": int(len(hist_x)), "test_rows": int(len(test_x)),
            "feature_count": int(hist_x.shape[1]), "feature_schema": hist_x.columns.tolist(),
            "horizon_historical": {str(k): int(v) for k, v in hist_m.h.value_counts().sort_index().items()},
            "horizon_test": {str(k): int(v) for k, v in test_m.h.value_counts().sort_index().items()},
            "broad_domain_classifier": broad, "matched_domain_classifier": matched,
            "matching": matching,
            "top_feature_discrepancies": discrepancies.head(30).to_dict(orient="records"),
            "metadata_discrepancies": categorical.to_dict(orient="records"),
            "limitation": "Historical and Test calendar periods are disjoint; spatial grouping and horizon/season/geography matching reduce, but cannot eliminate, time-period confounding.",
            "no_test_predictions": True,
        }
        _json(args.output_dir / "metrics.json", report)
        manifest.update({
            "status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "output_checksums": {str(p.relative_to(args.output_dir)): _sha256(p) for p in args.output_dir.rglob("*") if p.is_file() and p.name != "manifest.json"},
        })
        _json(args.output_dir / "manifest.json", manifest)
        print(json.dumps({"status": "completed", "output_dir": str(args.output_dir), "broad_auc": broad["mean_auc"], "matched_auc": matched["mean_auc"]}, indent=2), flush=True)
    except Exception as exc:
        manifest.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(),
                         "elapsed_seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}"})
        _json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
