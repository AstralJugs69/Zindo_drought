"""Phase A spatial error-budget diagnostics (execute on Kaggle only).

This script consumes a completed local-response run's OOF tables and Train.csv.
It does not read Test.csv, fit a forecast model, or emit submission predictions.
Spatial bases are fitted independently inside each evaluation fold from the
training prefix and are labelled ORACLE_DIAGNOSTIC when evaluated on OOF errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ORIGINS = ("2007-09", "2009-01", "2014-12")
RANKS = (4, 8, 16, 32)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _basis(train: pd.DataFrame, cutoff: pd.Period, rank: int) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    """Fit a location-centred anomaly basis using only rows before cutoff."""
    x = train.loc[pd.to_datetime(train["time"]).dt.to_period("M") < cutoff].copy()
    x["period"] = pd.to_datetime(x["time"]).dt.to_period("M")
    x = x.dropna(subset=["TWS_t", "lat", "lon"])
    # One value per location/month; this is a historical state basis only.
    field = x.pivot_table(index="period", columns=["lat", "lon"], values="TWS_t", aggfunc="mean")
    field = field.sort_index().sort_index(axis=1).dropna(axis=1, how="all")
    if field.empty or field.shape[1] < 2:
        return field, np.empty((0, 0)), {"rows": int(len(x)), "cells": int(field.shape[1]), "rank": 0}
    # Historical location means are fit once; missing prefix months become a
    # zero anomaly, never an interpolation across time or the eval boundary.
    centered = field - field.mean(axis=0)
    centered = centered.fillna(0.0)
    u, s, vt = np.linalg.svd(centered.to_numpy(dtype=float), full_matrices=False)
    k = min(rank, vt.shape[0])
    explained = float((s[:k] ** 2).sum() / max((s**2).sum(), 1e-12))
    return field, vt[:k].T, {"rows": int(len(x)), "cells": int(field.shape[1]), "rank": int(k), "train_explained_variance": explained}


def _error_budget(frame: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    x = frame.copy()
    x["err"] = x["truth"] - x["prediction"]
    x["block_lat"] = np.floor(x["lat"] / 10.0) * 10.0
    x["block_lon"] = np.floor(x["lon"] / 10.0) * 10.0
    rows = []
    for date, g in x.groupby("source_date", sort=True):
        e = g["err"].to_numpy(float)
        rows.append({"source_date": str(date), "rows": int(len(g)), "target_mean": float(g.truth.mean()), "prediction_mean": float(g.prediction.mean()), "target_std": float(g.truth.std(ddof=0)), "prediction_std": float(g.prediction.std(ddof=0)), "bias": float(e.mean()), "centered_error_variance": float(e.var()), "sse": float(np.square(e).sum()), "sse_share": float(np.square(e).sum() / max(np.square(x.err).sum(), 1e-12))})
    blocks = []
    total = max(float(np.square(x.err).sum()), 1e-12)
    for (date, blat, blon), g in x.groupby(["source_date", "block_lat", "block_lon"], sort=True):
        e = g.err.to_numpy(float)
        blocks.append({"source_date": str(date), "block_lat": float(blat), "block_lon": float(blon), "rows": int(len(g)), "bias": float(e.mean()), "within_sse": float(np.square(e - e.mean()).sum()), "squared_block_bias": float(len(e) * e.mean() ** 2), "sse": float(np.square(e).sum()), "sse_share": float(np.square(e).sum() / total)})
    return rows, blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--expected-commit")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.expected_commit:
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True).strip()
        if actual != args.expected_commit:
            raise RuntimeError({"expected_commit": args.expected_commit, "actual_commit": actual})
    train = pd.read_csv(args.data_dir / "Train.csv", usecols=["time", "lat", "lon", "TWS_t", "target"])
    oof_files = sorted(args.run_dir.glob("oof_*_r01_lgbm.csv.gz"))
    if not oof_files:
        raise FileNotFoundError(f"No R01 OOF tables in {args.run_dir}")
    all_budget, all_blocks, basis_rows, projections = [], [], [], []
    for path in oof_files:
        origin = next((o for o in ORIGINS if o in path.name), None)
        if origin is None:
            continue
        oof = pd.read_csv(path)
        oof = oof.loc[oof["h"].between(1, 7)].copy()
        oof["source_date"] = pd.to_datetime(oof["source_date"]).dt.strftime("%Y-%m-%d")
        budget, blocks = _error_budget(oof)
        all_budget.extend({"origin": origin, **r} for r in budget)
        all_blocks.extend({"origin": origin, **r} for r in blocks)
        cutoff = pd.Period(origin, freq="M")
        field, basis32, _meta32 = _basis(train, cutoff, 32)
        lookup = {tuple(c): i for i, c in enumerate(field.columns)}
        for rank in RANKS:
            basis = basis32[:, : min(rank, basis32.shape[1])] if basis32.size else basis32
            rank_used = int(basis.shape[1]) if basis.ndim == 2 else 0
            requested_meta = dict(_meta32)
            requested_meta["rank"] = rank_used
            requested_meta["requested_rank"] = rank
            if basis32.size and rank_used:
                _, _, requested_meta = _basis(train, cutoff, rank)
                requested_meta["requested_rank"] = rank
                requested_meta["rank_used"] = rank_used
            basis_rows.append({"origin": origin, **requested_meta})
            if basis.size == 0:
                continue
            for date, day in oof.groupby("source_date", sort=True):
                loc = [lookup.get((float(a), float(b))) for a, b in zip(day.lat, day.lon)]
                keep = np.array([i is not None for i in loc])
                if not keep.any():
                    continue
                residual = (day.loc[keep, "truth"] - day.loc[keep, "prediction"]).to_numpy(float)
                design = basis[np.asarray([loc[i] for i in np.flatnonzero(keep)], dtype=int), :]
                coef, *_ = np.linalg.lstsq(design, residual, rcond=None)
                fitted = design @ coef
                raw_sse = float(np.square(residual).sum())
                explained = float(raw_sse - np.square(residual - fitted).sum())
                projections.append({"origin": origin, "source_date": date, "rank": rank, "rows": int(keep.sum()), "coverage": float(keep.mean()), "raw_sse": raw_sse, "remaining_sse": float(np.square(residual - fitted).sum()), "explained_sse": explained, "fraction_residual_sse": float(explained / max(raw_sse, 1e-12)), "numerical_rank": int(np.linalg.matrix_rank(design)), "condition_number": float(np.linalg.cond(design)) if design.shape[1] else None, "label": "ORACLE_DIAGNOSTIC"})
    pd.DataFrame(all_budget).to_csv(args.output_dir / "error_budget_by_date.csv", index=False)
    pd.DataFrame(all_blocks).to_csv(args.output_dir / "error_budget_by_block.csv", index=False)
    pd.DataFrame(basis_rows).to_csv(args.output_dir / "basis_fit_summary.csv", index=False)
    proj = pd.DataFrame(projections)
    proj.to_csv(args.output_dir / "oracle_basis_projection_by_date.csv", index=False)
    if not proj.empty:
        agg = proj.groupby(["origin", "rank"], as_index=False).agg(
            supported_rows=("rows", "sum"), supported_sse=("raw_sse", "sum"),
            explained_sse=("explained_sse", "sum"), remaining_sse=("remaining_sse", "sum"))
        agg["supported_opportunity"] = agg["explained_sse"] / agg["supported_sse"].clip(lower=1e-12)
        # Denominator is computed once from every finite OOF residual, including
        # locations unsupported by the historical basis and without repeating it
        # for each requested rank/date projection.
        # Reconstruct from the emitted per-date error budget, which covers all
        # finite OOF rows exactly once.
        budget_df = pd.DataFrame(all_budget)
        all_sse = budget_df.groupby("origin")["sse"].sum().rename("all_finite_sse")
        agg = agg.join(all_sse, on="origin")
        agg["all_finite_opportunity"] = agg["explained_sse"] / agg["all_finite_sse"].clip(lower=1e-12)
        agg["label"] = "ORACLE_DIAGNOSTIC"
        agg.to_csv(args.output_dir / "oracle_basis_projection_aggregate.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.output_dir / "oracle_basis_projection_aggregate.csv", index=False)
    manifest = {"status": "completed", "phase": "A", "label": "ORACLE_DIAGNOSTIC", "origins": ORIGINS, "ranks": RANKS, "run_dir": str(args.run_dir), "files": {p.name: _sha256(p) for p in args.output_dir.glob("*.csv")}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "output_dir": str(args.output_dir), "projection_rows": len(projections), "label": "ORACLE_DIAGNOSTIC"}))


if __name__ == "__main__":
    main()
