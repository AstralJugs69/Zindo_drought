"""Causal dynamic-factor helpers for the B2 spatial forecasting screen.

This module is deliberately target-blind at inference time.  Target labels are
used only by :func:`build_training_episodes` to construct supervised factor
changes inside a fitting prefix.  Evaluation features consume a simulator
ledger plus source-date hydrometeorology and prefix-fitted state.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

HYDRO_COLUMNS = (
    "SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"
)
DEFAULT_HORIZONS = (1, 2, 3, 4, 5, 6, 7)
ALPHAS = (1.0, 10.0, 100.0)


def _period(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.to_period("M")


def _month_ordinal(period: pd.Period) -> int:
    return int(period.ordinal)


@dataclass(frozen=True)
class SpatialBasis:
    rank: int
    cutoff: str
    cells: tuple[tuple[float, float], ...]
    tws_mean: np.ndarray
    loadings: np.ndarray
    hydro_mean: np.ndarray
    training_periods: tuple[str, ...]
    explained_variance: float

    def location_lookup(self) -> dict[tuple[float, float], int]:
        return {cell: i for i, cell in enumerate(self.cells)}


@dataclass(frozen=True)
class EpisodeSet:
    source_period: np.ndarray
    target_period: np.ndarray
    anchor_period: np.ndarray
    h: np.ndarray
    month_sin: np.ndarray
    month_cos: np.ndarray
    z_anchor: np.ndarray
    hydro_source: np.ndarray
    hydro_anchor: np.ndarray
    source_coverage: np.ndarray
    anchor_coverage: np.ndarray
    y_delta: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.h))


@dataclass(frozen=True)
class PerFactorRidge:
    rank: int
    alpha: float
    x_mean: np.ndarray
    x_scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    feature_names: tuple[str, ...]

    def predict(self, *, z_anchor: np.ndarray, h: np.ndarray,
                month_sin: np.ndarray, month_cos: np.ndarray,
                hydro_source: np.ndarray, hydro_anchor: np.ndarray,
                source_coverage: np.ndarray, anchor_coverage: np.ndarray) -> np.ndarray:
        n = len(h)
        out = np.empty((n, self.rank), dtype=np.float64)
        for j in range(self.rank):
            x = factor_design(
                z_anchor=z_anchor, h=h, month_sin=month_sin, month_cos=month_cos,
                hydro_source=hydro_source, hydro_anchor=hydro_anchor,
                source_coverage=source_coverage, anchor_coverage=anchor_coverage,
                factor=j,
            )
            xs = (x - self.x_mean[j]) / self.x_scale[j]
            out[:, j] = xs @ self.coef[j] + self.intercept[j]
        return out


def fit_spatial_basis(train: pd.DataFrame, *, cutoff: pd.Period, rank: int) -> SpatialBasis:
    """Fit location means, an EOF basis, and hydro means strictly before cutoff."""
    needed = {"time", "lat", "lon", "TWS_t", *HYDRO_COLUMNS}
    missing = needed.difference(train.columns)
    if missing:
        raise ValueError(f"train missing required columns: {sorted(missing)}")
    x = train.loc[_period(train["time"]) < cutoff, ["time", "lat", "lon", "TWS_t", *HYDRO_COLUMNS]].copy()
    if x.empty:
        raise AssertionError("basis prefix is empty")
    x["period"] = _period(x["time"])
    tws = x.pivot_table(index="period", columns=["lat", "lon"], values="TWS_t", aggfunc="mean")
    tws = tws.sort_index().sort_index(axis=1).dropna(axis=1, how="all")
    if tws.shape[1] < 2 or tws.shape[0] < 2:
        raise AssertionError(f"insufficient basis field shape: {tws.shape}")
    means = tws.mean(axis=0).to_numpy(np.float64)
    centered = tws.sub(means, axis=1).fillna(0.0).to_numpy(np.float64)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(int(rank), vt.shape[0], vt.shape[1])
    if k < 1:
        raise AssertionError("numerical basis rank is zero")
    loadings = vt[:k].T.astype(np.float64, copy=False)
    explained = float(np.square(s[:k]).sum() / max(float(np.square(s).sum()), 1e-12))
    cells = tuple((float(a), float(b)) for a, b in tws.columns.tolist())

    cell_index = pd.MultiIndex.from_tuples(cells, names=["lat", "lon"])
    loc_mean = x.groupby(["lat", "lon"], sort=False)[list(HYDRO_COLUMNS)].mean()
    hmean = loc_mean.reindex(cell_index).to_numpy(np.float64)
    # Training-only fallback for a cell/variable with no finite historical mean.
    global_mean = np.nanmean(hmean, axis=0)
    inds = np.where(~np.isfinite(hmean))
    if len(inds[0]):
        hmean[inds] = global_mean[inds[1]]
    if not np.isfinite(hmean).all():
        raise AssertionError("hydrology mean contains non-finite values")
    return SpatialBasis(
        rank=k, cutoff=str(cutoff), cells=cells, tws_mean=means, loadings=loadings,
        hydro_mean=hmean, training_periods=tuple(str(p) for p in tws.index),
        explained_variance=explained,
    )


def _aligned_values(frame: pd.DataFrame, value_col: str, basis: SpatialBasis) -> tuple[np.ndarray, float]:
    if frame.empty:
        return np.full(len(basis.cells), np.nan), 0.0
    s = frame.groupby(["lat", "lon"], sort=False)[value_col].mean()
    cell_index = pd.MultiIndex.from_tuples(basis.cells, names=["lat", "lon"])
    vals = s.reindex(cell_index).to_numpy(np.float64)
    coverage = float(np.isfinite(vals).mean())
    return vals, coverage


def project_tws_field(frame: pd.DataFrame, value_col: str, basis: SpatialBasis,
                      *, min_coverage: float) -> tuple[np.ndarray | None, float]:
    vals, coverage = _aligned_values(frame, value_col, basis)
    if coverage < min_coverage:
        return None, coverage
    keep = np.isfinite(vals)
    design = basis.loadings[keep]
    y = vals[keep] - basis.tws_mean[keep]
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef.astype(np.float64), coverage


def project_hydro_period(frame: pd.DataFrame, basis: SpatialBasis,
                         *, min_coverage: float = 0.50) -> tuple[np.ndarray | None, float]:
    """Project five source-date hydro anomaly fields onto the fixed TWS basis."""
    cell_index = pd.MultiIndex.from_tuples(basis.cells, names=["lat", "lon"])
    aligned = frame.groupby(["lat", "lon"], sort=False)[list(HYDRO_COLUMNS)].mean().reindex(cell_index)
    values = aligned.to_numpy(np.float64)
    coeffs = np.full((len(HYDRO_COLUMNS), basis.rank), np.nan, dtype=np.float64)
    coverages = np.mean(np.isfinite(values), axis=0)
    if float(np.min(coverages)) < min_coverage:
        return None, float(np.min(coverages))
    for v in range(len(HYDRO_COLUMNS)):
        keep = np.isfinite(values[:, v])
        y = values[keep, v] - basis.hydro_mean[keep, v]
        design = basis.loadings[keep]
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        coeffs[v] = coef
    return coeffs, float(np.min(coverages))


def period_hydro_projections(source: pd.DataFrame, basis: SpatialBasis,
                             periods: Iterable[pd.Period], *, min_coverage: float = 0.50
                             ) -> dict[pd.Period, tuple[np.ndarray, float]]:
    x = source.loc[:, ["time", "lat", "lon", *HYDRO_COLUMNS]].copy()
    x["period"] = _period(x["time"])
    groups = {p: g for p, g in x.groupby("period", sort=False)}
    out: dict[pd.Period, tuple[np.ndarray, float]] = {}
    for p in sorted(set(periods)):
        g = groups.get(p)
        if g is None:
            continue
        proj, cov = project_hydro_period(g, basis, min_coverage=min_coverage)
        if proj is not None:
            out[p] = (proj, cov)
    return out


def prefix_substantial_factors(train: pd.DataFrame, basis: SpatialBasis, *, cutoff: pd.Period,
                               min_coverage: float = 0.95) -> dict[pd.Period, tuple[np.ndarray, float]]:
    x = train.loc[_period(train["time"]) < cutoff, ["time", "lat", "lon", "TWS_t"]].copy()
    x["period"] = _period(x["time"])
    out: dict[pd.Period, tuple[np.ndarray, float]] = {}
    for p, g in x.groupby("period", sort=True):
        z, cov = project_tws_field(g, "TWS_t", basis, min_coverage=min_coverage)
        if z is not None:
            out[p] = (z, cov)
    return out


def visible_eval_factors(ledger: pd.DataFrame, basis: SpatialBasis, *, min_coverage: float = 0.95
                         ) -> dict[pd.Period, tuple[np.ndarray, float]]:
    """Construct only genuinely visible evaluation TWS fields from simulator state."""
    req = {"source_date", "lat", "lon", "tws_visible", "last_observed_date", "last_observed_TWS"}
    missing = req.difference(ledger.columns)
    if missing:
        raise ValueError(f"ledger missing required columns: {sorted(missing)}")
    x = ledger.copy()
    x["period"] = pd.to_datetime(x["source_date"]).dt.to_period("M")
    visible = x.loc[x["tws_visible"].astype(bool) &
                    (pd.to_datetime(x["last_observed_date"]).dt.to_period("M") == x["period"])].copy()
    out: dict[pd.Period, tuple[np.ndarray, float]] = {}
    for p, g in visible.groupby("period", sort=True):
        z, cov = project_tws_field(g.rename(columns={"last_observed_TWS": "visible_tws"}),
                                   "visible_tws", basis, min_coverage=min_coverage)
        if z is not None:
            out[p] = (z, cov)
    return out


def build_training_episodes(train: pd.DataFrame, basis: SpatialBasis, *, cutoff: pd.Period,
                            horizons: tuple[int, ...] = DEFAULT_HORIZONS,
                            min_tws_coverage: float = 0.95,
                            min_hydro_coverage: float = 0.50) -> EpisodeSet:
    """Build direct anchor/source/target factor episodes with target=source+1 month.

    Labels are consumed only here.  Every episode's target month is strictly before
    ``cutoff``.  Missing calendar months are not compressed: anchor dates are
    computed by calendar arithmetic and absent anchor fields simply yield no episode.
    """
    req = {"time", "lat", "lon", "TWS_t", "target", "month_sin", "month_cos", *HYDRO_COLUMNS}
    missing = req.difference(train.columns)
    if missing:
        raise ValueError(f"train missing required columns: {sorted(missing)}")
    x = train.copy()
    x["period"] = _period(x["time"])
    source_periods = sorted(p for p in x.loc[x["period"] < cutoff, "period"].unique()
                            if p + 1 < cutoff)
    anchor_factors = prefix_substantial_factors(x, basis, cutoff=cutoff, min_coverage=min_tws_coverage)
    hydro = period_hydro_projections(x, basis,
                                     set(source_periods) | set(anchor_factors),
                                     min_coverage=min_hydro_coverage)
    groups = {p: g for p, g in x.groupby("period", sort=False)}
    target_factors: dict[pd.Period, tuple[np.ndarray, float]] = {}
    for p in source_periods:
        g = groups[p]
        z, cov = project_tws_field(g, "target", basis, min_coverage=min_tws_coverage)
        if z is not None:
            target_factors[p] = (z, cov)

    rows = []
    for src in source_periods:
        if src not in hydro or src not in target_factors:
            continue
        g = groups[src]
        msin = float(np.nanmean(g["month_sin"].to_numpy(np.float64)))
        mcos = float(np.nanmean(g["month_cos"].to_numpy(np.float64)))
        z_target = target_factors[src][0]
        for h in horizons:
            anchor = src + (1 - int(h))
            if anchor not in anchor_factors or anchor not in hydro:
                continue
            z_anchor, _ = anchor_factors[anchor]
            hs, scov = hydro[src]
            ha, acov = hydro[anchor]
            rows.append((src, src + 1, anchor, int(h), msin, mcos, z_anchor, hs, ha,
                         float(scov), float(acov), z_target - z_anchor))
    if not rows:
        raise AssertionError("no legal B2 training episodes")
    return EpisodeSet(
        source_period=np.asarray([_month_ordinal(r[0]) for r in rows], dtype=np.int32),
        target_period=np.asarray([_month_ordinal(r[1]) for r in rows], dtype=np.int32),
        anchor_period=np.asarray([_month_ordinal(r[2]) for r in rows], dtype=np.int32),
        h=np.asarray([r[3] for r in rows], dtype=np.float64),
        month_sin=np.asarray([r[4] for r in rows], dtype=np.float64),
        month_cos=np.asarray([r[5] for r in rows], dtype=np.float64),
        z_anchor=np.stack([r[6] for r in rows]),
        hydro_source=np.stack([r[7] for r in rows]),
        hydro_anchor=np.stack([r[8] for r in rows]),
        source_coverage=np.asarray([r[9] for r in rows], dtype=np.float64),
        anchor_coverage=np.asarray([r[10] for r in rows], dtype=np.float64),
        y_delta=np.stack([r[11] for r in rows]),
    )


FEATURE_NAMES = (
    "anchor_factor", "h", "month_sin", "month_cos",
    *(f"source_{v}" for v in HYDRO_COLUMNS),
    *(f"anchor_{v}" for v in HYDRO_COLUMNS),
    "source_hydro_coverage", "anchor_hydro_coverage",
)


def factor_design(*, z_anchor: np.ndarray, h: np.ndarray, month_sin: np.ndarray,
                  month_cos: np.ndarray, hydro_source: np.ndarray,
                  hydro_anchor: np.ndarray, source_coverage: np.ndarray,
                  anchor_coverage: np.ndarray, factor: int) -> np.ndarray:
    """Single production design builder shared by training and inference."""
    return np.column_stack([
        z_anchor[:, factor], h, month_sin, month_cos,
        hydro_source[:, :, factor], hydro_anchor[:, :, factor],
        source_coverage, anchor_coverage,
    ]).astype(np.float64, copy=False)


def _episode_subset(e: EpisodeSet, mask: np.ndarray) -> EpisodeSet:
    return EpisodeSet(**{name: getattr(e, name)[mask] for name in e.__dataclass_fields__})


def fit_per_factor_ridge(episodes: EpisodeSet, *, alpha: float) -> PerFactorRidge:
    rank = int(episodes.y_delta.shape[1])
    p = len(FEATURE_NAMES)
    means = np.empty((rank, p)); scales = np.empty((rank, p))
    coef = np.empty((rank, p)); intercept = np.empty(rank)
    for j in range(rank):
        x = factor_design(
            z_anchor=episodes.z_anchor, h=episodes.h,
            month_sin=episodes.month_sin, month_cos=episodes.month_cos,
            hydro_source=episodes.hydro_source, hydro_anchor=episodes.hydro_anchor,
            source_coverage=episodes.source_coverage, anchor_coverage=episodes.anchor_coverage,
            factor=j,
        )
        mean = x.mean(axis=0)
        scale = x.std(axis=0, ddof=0)
        scale[scale < 1e-12] = 1.0
        xs = (x - mean) / scale
        y = episodes.y_delta[:, j].astype(np.float64, copy=False)
        y_mean = float(y.mean())
        gram = xs.T @ xs
        rhs = xs.T @ (y - y_mean)
        ridge = gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64)
        beta = np.linalg.solve(ridge, rhs)
        means[j] = mean; scales[j] = scale
        coef[j] = beta; intercept[j] = y_mean
    return PerFactorRidge(rank, float(alpha), means, scales, coef, intercept, FEATURE_NAMES)


def select_alpha_chronologically(episodes: EpisodeSet, alphas: tuple[float, ...] = ALPHAS
                                 ) -> tuple[float, list[dict[str, float]]]:
    """Choose ridge penalty on inner future blocks using only earlier episodes."""
    unique = np.unique(episodes.source_period)
    if len(unique) < 16:
        raise AssertionError(f"need >=16 distinct source months for inner selection, got {len(unique)}")
    # Three expanding-prefix cuts, each followed by a non-overlapping future block.
    cut_idx = sorted(set(max(8, min(len(unique)-2, int(frac * len(unique)))) for frac in (0.55, 0.70, 0.82)))
    splits = []
    for idx, ci in enumerate(cut_idx):
        cutoff = unique[ci]
        next_cut = unique[cut_idx[idx+1]] if idx + 1 < len(cut_idx) else unique[-1] + 1
        tr = episodes.target_period < cutoff
        va = (episodes.source_period >= cutoff) & (episodes.source_period < next_cut)
        if tr.sum() and va.sum() and len(np.unique(episodes.source_period[tr])) >= 8:
            splits.append((cutoff, tr, va))
    if len(splits) < 2:
        raise AssertionError("insufficient chronological inner splits")
    scores = []
    for alpha in alphas:
        fold_mse = []
        for cutoff, tr, va in splits:
            model = fit_per_factor_ridge(_episode_subset(episodes, tr), alpha=float(alpha))
            v = _episode_subset(episodes, va)
            pred = model.predict(z_anchor=v.z_anchor, h=v.h, month_sin=v.month_sin,
                                 month_cos=v.month_cos, hydro_source=v.hydro_source,
                                 hydro_anchor=v.hydro_anchor, source_coverage=v.source_coverage,
                                 anchor_coverage=v.anchor_coverage)
            fold_mse.append(float(np.mean(np.square(pred - v.y_delta))))
            scores.append({"alpha": float(alpha), "cutoff_ordinal": float(cutoff),
                           "train_episodes": float(tr.sum()), "valid_episodes": float(va.sum()),
                           "mse": fold_mse[-1]})
    means = {float(a): float(np.mean([r["mse"] for r in scores if r["alpha"] == float(a)])) for a in alphas}
    # Deterministic tie break prefers stronger regularization.
    best = min((v, -a, a) for a, v in means.items())[2]
    return float(best), scores


def _period_stats(source: pd.DataFrame) -> tuple[dict[pd.Period, float], dict[pd.Period, float]]:
    x = source.loc[:, ["time", "month_sin", "month_cos"]].copy()
    x["period"] = _period(x["time"])
    g = x.groupby("period", sort=False)
    return (g["month_sin"].mean().to_dict(), g["month_cos"].mean().to_dict())


def predict_with_fallback(*, ledger: pd.DataFrame, source: pd.DataFrame, basis: SpatialBasis,
                          model: PerFactorRidge, prefix_factors: dict[pd.Period, tuple[np.ndarray, float]],
                          fallback_prediction: np.ndarray, min_tws_coverage: float = 0.95,
                          min_hydro_coverage: float = 0.50) -> tuple[np.ndarray, pd.DataFrame]:
    """Forecast only rows with a legal substantial anchor factor; otherwise fallback."""
    if len(ledger) != len(fallback_prediction):
        raise ValueError("fallback prediction length mismatch")
    x = ledger.copy().reset_index(drop=True)
    x["source_period"] = pd.to_datetime(x["source_date"]).dt.to_period("M")
    x["anchor_period"] = pd.to_datetime(x["last_observed_date"]).dt.to_period("M")
    factors = dict(prefix_factors)
    factors.update(visible_eval_factors(x, basis, min_coverage=min_tws_coverage))
    needed_hydro = set(x["source_period"]) | set(x["anchor_period"])
    hydro = period_hydro_projections(source, basis, needed_hydro, min_coverage=min_hydro_coverage)
    msin, mcos = _period_stats(source)
    cell_index = pd.MultiIndex.from_tuples(basis.cells, names=["lat", "lon"])
    row_index = pd.MultiIndex.from_arrays([x["lat"].astype(float), x["lon"].astype(float)], names=["lat", "lon"])
    loc_idx = cell_index.get_indexer(row_index)

    pred = np.asarray(fallback_prediction, dtype=np.float64).copy()
    reason = np.full(len(x), "supported", dtype=object)
    supported = loc_idx >= 0
    reason[~supported] = "location_not_in_basis"

    # Every row in a (source, anchor) group has the same temporal/hydrology
    # design. Predict one factor-change vector per group, then map it to all
    # locations in a single matrix-vector product.
    valid = x.loc[supported].copy()
    for (src, anchor), group in valid.groupby(["source_period", "anchor_period"], sort=False):
        idx = group.index.to_numpy(dtype=int)
        why = None
        if anchor not in factors:
            why = "anchor_not_substantial"
        elif src not in hydro:
            why = "source_hydro_unavailable"
        elif anchor not in hydro:
            why = "anchor_hydro_unavailable"
        elif src not in msin or src not in mcos:
            why = "source_month_encoding_unavailable"
        if why is not None:
            supported[idx] = False
            reason[idx] = why
            continue
        hvals = group["h"].to_numpy(np.float64)
        if not np.allclose(hvals, hvals[0]):
            raise AssertionError("shared source/anchor group has inconsistent h")
        z_anchor, _ = factors[anchor]
        hs, scov = hydro[src]
        ha, acov = hydro[anchor]
        dz = model.predict(
            z_anchor=z_anchor.reshape(1, -1), h=np.asarray([hvals[0]]),
            month_sin=np.asarray([float(msin[src])]), month_cos=np.asarray([float(mcos[src])]),
            hydro_source=hs.reshape(1, *hs.shape), hydro_anchor=ha.reshape(1, *ha.shape),
            source_coverage=np.asarray([scov]), anchor_coverage=np.asarray([acov]),
        )[0]
        locs = loc_idx[idx]
        spatial_delta = basis.loadings[locs] @ dz
        pred[idx] = group["last_observed_TWS"].to_numpy(np.float64) + spatial_delta
    provenance = pd.DataFrame({"supported": supported, "fallback_reason": reason,
                               "source_period": x["source_period"].astype(str),
                               "anchor_period": x["anchor_period"].astype(str)})
    return pred, provenance


def save_bundle(path: Path, basis: SpatialBasis, model: PerFactorRidge, metadata: dict[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(path / "arrays.npz", tws_mean=basis.tws_mean, loadings=basis.loadings,
                        hydro_mean=basis.hydro_mean, x_mean=model.x_mean, x_scale=model.x_scale,
                        coef=model.coef, intercept=model.intercept)
    payload = {
        "basis": {"rank": basis.rank, "cutoff": basis.cutoff, "cells": [list(c) for c in basis.cells],
                  "training_periods": list(basis.training_periods), "explained_variance": basis.explained_variance},
        "model": {"rank": model.rank, "alpha": model.alpha, "feature_names": list(model.feature_names)},
        "metadata": metadata,
    }
    (path / "bundle.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_bundle(path: Path) -> tuple[SpatialBasis, PerFactorRidge, dict[str, object]]:
    meta = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    a = np.load(path / "arrays.npz")
    b = meta["basis"]; m = meta["model"]
    basis = SpatialBasis(int(b["rank"]), str(b["cutoff"]), tuple(tuple(map(float, c)) for c in b["cells"]),
                         a["tws_mean"], a["loadings"], a["hydro_mean"], tuple(b["training_periods"]),
                         float(b["explained_variance"]))
    model = PerFactorRidge(int(m["rank"]), float(m["alpha"]), a["x_mean"], a["x_scale"],
                           a["coef"], a["intercept"], tuple(m["feature_names"]))
    return basis, model, dict(meta.get("metadata", {}))
