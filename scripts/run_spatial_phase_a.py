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



def _project_residuals_by_date(
    oof: pd.DataFrame,
    basis: np.ndarray,
    lookup: dict[tuple[float, float], int],
    *,
    origin: str,
    requested_rank: int,
) -> list[dict[str, object]]:
    """Project residuals independently by date onto a fixed past-fitted basis."""
    if basis.ndim != 2:
        raise ValueError("basis must be 2-D")
    rank_used = int(basis.shape[1])
    rows: list[dict[str, object]] = []
    x = oof.copy()
    if "err" not in x.columns:
        x["err"] = x["truth"] - x["prediction"]
    for date, day in x.groupby("source_date", sort=True):
        loc = [lookup.get((float(a), float(b))) for a, b in zip(day.lat, day.lon)]
        keep = np.asarray([i is not None for i in loc], dtype=bool)
        all_residual = day["err"].to_numpy(float)
        all_sse = float(np.square(all_residual).sum())
        if keep.any() and rank_used:
            residual = day.loc[keep, "err"].to_numpy(float)
            design = basis[np.asarray([loc[i] for i in np.flatnonzero(keep)], dtype=int), :]
            coef, *_ = np.linalg.lstsq(design, residual, rcond=None)
            fitted = design @ coef
            supported_sse = float(np.square(residual).sum())
            remaining = float(np.square(residual - fitted).sum())
            explained = supported_sse - remaining
            numerical_rank = int(np.linalg.matrix_rank(design))
            condition = float(np.linalg.cond(design)) if design.shape[1] else None
        else:
            supported_sse = remaining = explained = 0.0
            numerical_rank = 0
            condition = None
        rows.append({
            "origin": origin,
            "source_date": str(date),
            "rank": requested_rank,
            "requested_rank": requested_rank,
            "rank_used": rank_used,
            "rows": int(keep.sum()),
            "all_rows": int(len(day)),
            "unsupported_rows": int((~keep).sum()),
            "coverage": float(keep.mean()) if len(keep) else 0.0,
            "all_finite_sse": all_sse,
            "raw_sse": supported_sse,
            "remaining_sse": remaining,
            "explained_sse": explained,
            "fraction_residual_sse": float(explained / max(supported_sse, 1e-12)),
            "numerical_rank": numerical_rank,
            "condition_number": condition,
            "label": "ORACLE_DIAGNOSTIC",
        })
    return rows


def _aggregate_projection_opportunity(projections: pd.DataFrame, error_budget: pd.DataFrame) -> pd.DataFrame:
    """Aggregate opportunity without repeating the all-finite denominator by rank/date."""
    if projections.empty:
        return pd.DataFrame()
    agg = projections.groupby(["origin", "rank"], as_index=False).agg(
        supported_rows=("rows", "sum"), supported_sse=("raw_sse", "sum"),
        explained_sse=("explained_sse", "sum"), remaining_sse=("remaining_sse", "sum"),
        unsupported_rows=("unsupported_rows", "sum"),
        rank_used=("rank_used", "max"),
    )
    # error_budget has exactly one row per origin/date and is independent of the
    # number of requested projection ranks, so the denominator cannot multiply.
    all_sse = error_budget.groupby("origin")["sse"].sum().rename("all_finite_sse")
    agg = agg.join(all_sse, on="origin")
    agg["supported_opportunity"] = agg["explained_sse"] / agg["supported_sse"].clip(lower=1e-12)
    agg["all_finite_opportunity"] = agg["explained_sse"] / agg["all_finite_sse"].clip(lower=1e-12)
    agg["label"] = "ORACLE_DIAGNOSTIC"
    return agg

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
            projections.extend(_project_residuals_by_date(
                oof, basis, lookup, origin=origin, requested_rank=rank
            ))
    pd.DataFrame(all_budget).to_csv(args.output_dir / "error_budget_by_date.csv", index=False)
    pd.DataFrame(all_blocks).to_csv(args.output_dir / "error_budget_by_block.csv", index=False)
    pd.DataFrame(basis_rows).to_csv(args.output_dir / "basis_fit_summary.csv", index=False)
    proj = pd.DataFrame(projections)
    proj.to_csv(args.output_dir / "oracle_basis_projection_by_date.csv", index=False)
    if not proj.empty:
        agg = _aggregate_projection_opportunity(proj, pd.DataFrame(all_budget))
        agg.to_csv(args.output_dir / "oracle_basis_projection_aggregate.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.output_dir / "oracle_basis_projection_aggregate.csv", index=False)
    manifest = {"status": "completed", "phase": "A", "label": "ORACLE_DIAGNOSTIC", "origins": ORIGINS, "ranks": RANKS, "run_dir": str(args.run_dir), "files": {p.name: _sha256(p) for p in args.output_dir.glob("*.csv")}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "output_dir": str(args.output_dir), "projection_rows": len(projections), "label": "ORACLE_DIAGNOSTIC"}))


if __name__ == "__main__":
    main()
