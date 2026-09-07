"""Local linear hydrological responses, regularized toward a global fit.

All learned quantities are fitted on supplied training examples only. The caller
owns the temporal cutoff and observation simulator. No validation labels enter
normalization, the global prior, or per-location coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml_features import HYDRO_GAP_SAFE_FEATURE_COLUMNS


BASE_COLUMNS = [c for c in HYDRO_GAP_SAFE_FEATURE_COLUMNS if c not in {"lat", "lon"}]
INTERACTIONS = ("last_observed_TWS", "SOIL_MOISTURE_t", "SPEI_12_t")


def response_basis(features: pd.DataFrame) -> np.ndarray:
    """Location is the grouping key, not a linear latitude/longitude predictor."""
    x = features.loc[:, BASE_COLUMNS].to_numpy(dtype=np.float64)
    h = features["h"].to_numpy(dtype=np.float64) / 7.0
    extras = [features[c].to_numpy(dtype=np.float64) * h for c in INTERACTIONS]
    result = np.column_stack([x, *extras])
    if not np.isfinite(result).all():
        raise ValueError("Local response requires finite availability-safe features")
    return result


@dataclass
class LocalResponse:
    mean: np.ndarray
    scale: np.ndarray
    locations: pd.DataFrame
    global_coef: np.ndarray
    local_coef: np.ndarray
    counts: np.ndarray
    alpha: float

    def predict(self, features: pd.DataFrame, *, local: bool = True) -> np.ndarray:
        x = np.column_stack([np.ones(len(features)), (response_basis(features) - self.mean) / self.scale])
        if not local:
            return x @ self.global_coef
        keys = pd.MultiIndex.from_frame(self.locations[["lat", "lon"]])
        ids = keys.get_indexer(pd.MultiIndex.from_frame(features[["lat", "lon"]]))
        coef = np.repeat(self.global_coef[None, :], len(x), axis=0)
        known = ids >= 0
        coef[known] = self.local_coef[ids[known]]
        return np.einsum("ij,ij->i", x, coef)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, mean=self.mean, scale=self.scale,
                            locations=self.locations.to_numpy(), global_coef=self.global_coef,
                            local_coef=self.local_coef, counts=self.counts, alpha=self.alpha)

    @classmethod
    def load(cls, path: Path) -> "LocalResponse":
        with np.load(path, allow_pickle=False) as state:
            return cls(state["mean"], state["scale"],
                       pd.DataFrame(state["locations"], columns=["lat", "lon"]),
                       state["global_coef"], state["local_coef"], state["counts"], float(state["alpha"]))


def fit_local_response(features: pd.DataFrame, y: np.ndarray, weights: np.ndarray,
                       *, alpha: float = 30.0) -> LocalResponse:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    raw = response_basis(features)
    y, w = np.asarray(y, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if len(y) != len(raw) or len(w) != len(raw) or len(raw) == 0:
        raise ValueError("Nonempty features, labels and weights must align")
    if not np.isfinite(y).all() or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Labels must be finite and weights finite/positive")
    mean = np.average(raw, axis=0, weights=w)
    scale = np.sqrt(np.average((raw - mean) ** 2, axis=0, weights=w))
    scale = np.where(scale > 1e-8, scale, 1.0)
    x = np.column_stack([np.ones(len(raw)), (raw - mean) / scale])
    locations = features[["lat", "lon"]].drop_duplicates().sort_values(["lat", "lon"]).reset_index(drop=True)
    ids = pd.MultiIndex.from_frame(locations).get_indexer(pd.MultiIndex.from_frame(features[["lat", "lon"]]))
    n, d = len(locations), x.shape[1]
    counts = np.bincount(ids, minlength=n)
    gram = np.zeros((n, d, d), dtype=np.float64)
    rhs = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        weighted = w * x[:, j]
        rhs[:, j] = np.bincount(ids, weights=weighted * y, minlength=n)
        for k in range(j + 1):
            values = np.bincount(ids, weights=weighted * x[:, k], minlength=n)
            gram[:, j, k] = values
            gram[:, k, j] = values
    global_gram = gram.sum(axis=0) + np.eye(d) * 1e-6
    prior = np.linalg.solve(global_gram, rhs.sum(axis=0))
    gram[:, np.arange(d), np.arange(d)] += alpha
    local_coef = np.linalg.solve(gram, (rhs + alpha * prior)[..., None])[..., 0]
    if not np.isfinite(local_coef).all():
        raise AssertionError("Nonfinite local coefficients")
    return LocalResponse(mean, scale, locations, prior, local_coef, counts, alpha)
