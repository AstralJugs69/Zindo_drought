from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


HARMONIC_COLUMNS = ["b0", "b_trend", "b_sin1", "b_cos1", "b_sin2", "b_cos2"]


def _month_number(values: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(values)
    return (dt.dt.year.to_numpy(dtype=np.int32) * 12 + dt.dt.month.to_numpy(dtype=np.int32))


def _design_from_month_number(month_number: np.ndarray, origin: int) -> np.ndarray:
    month_number = np.asarray(month_number, dtype=np.float64)
    # Trend is in years so the coefficient is numerically interpretable/stable.
    trend = (month_number - float(origin)) / 12.0
    month_phase = np.mod(month_number - 1.0, 12.0)
    angle = 2.0 * np.pi * month_phase / 12.0
    return np.column_stack(
        [
            np.ones(len(month_number), dtype=np.float64),
            trend,
            np.sin(angle),
            np.cos(angle),
            np.sin(2.0 * angle),
            np.cos(2.0 * angle),
        ]
    )


@dataclass(frozen=True)
class HarmonicTrendModel:
    coefficients: pd.DataFrame
    origin_month_number: int
    fit_before: pd.Period


def fit_location_harmonic_trend(
    train: pd.DataFrame,
    *,
    fit_before: str | pd.Period,
    min_observations: int = 18,
) -> HarmonicTrendModel:
    """Fit per-location trend + annual + semiannual harmonic models causally.

    Only raw TWS observations strictly before `fit_before` are used. The target
    column is neither required nor inspected. This strict cutoff keeps a TWS value
    in the first validation source month from leaking into synthetic h>1 examples.
    """
    required = {"time", "lat", "lon", "TWS_t"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Missing columns for harmonic fit: {sorted(missing)}")

    cutoff = pd.Period(fit_before, freq="M")
    history = train.loc[:, ["time", "lat", "lon", "TWS_t"]].copy()
    history["date"] = pd.to_datetime(history["time"])
    history["period"] = history["date"].dt.to_period("M")
    history = history.loc[(history["period"] < cutoff) & history["TWS_t"].notna()].copy()
    if history.empty:
        raise ValueError(f"No TWS history exists before {cutoff}")

    month_num = _month_number(history["date"])
    origin = int(month_num.min())
    history["month_number"] = month_num

    rows: list[tuple[float, ...]] = []
    for (lat, lon), group in history.groupby(["lat", "lon"], sort=False):
        if len(group) < min_observations:
            continue
        X = _design_from_month_number(group["month_number"].to_numpy(), origin)
        y = group["TWS_t"].to_numpy(dtype=np.float64)
        if not np.isfinite(y).all():
            continue
        coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        if rank < X.shape[1] or not np.isfinite(coef).all():
            continue
        rows.append((float(lat), float(lon), *[float(v) for v in coef]))

    if not rows:
        raise AssertionError("No location harmonic models could be fitted")

    coefficients = pd.DataFrame(rows, columns=["lat", "lon", *HARMONIC_COLUMNS])
    if coefficients.duplicated(["lat", "lon"]).any():
        raise AssertionError("Duplicate location coefficients in harmonic model")

    return HarmonicTrendModel(
        coefficients=coefficients,
        origin_month_number=origin,
        fit_before=cutoff,
    )


def predict_harmonic_trend(
    model: HarmonicTrendModel,
    rows: pd.DataFrame,
    *,
    date_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a harmonic/trend value and return availability mask."""
    required = {"lat", "lon", date_column}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    base = rows.loc[:, ["lat", "lon", date_column]].copy()
    base["_order"] = np.arange(len(base), dtype=np.int64)
    merged = base.merge(
        model.coefficients,
        how="left",
        on=["lat", "lon"],
        validate="many_to_one",
        sort=False,
    ).sort_values("_order")

    available = merged["b0"].notna().to_numpy()
    pred = np.full(len(rows), np.nan, dtype=np.float64)
    if available.any():
        idx = np.flatnonzero(available)
        month_num = _month_number(merged.loc[available, date_column])
        X = _design_from_month_number(month_num, model.origin_month_number)
        coef = merged.loc[available, HARMONIC_COLUMNS].to_numpy(dtype=np.float64)
        pred[idx] = np.einsum("ij,ij->i", X, coef)
    return pred, available


def predict_persistence(ledger: pd.DataFrame) -> np.ndarray:
    if "last_observed_TWS" not in ledger.columns:
        raise ValueError("Ledger must contain last_observed_TWS")
    pred = ledger["last_observed_TWS"].to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all():
        raise ValueError("Persistence produced non-finite predictions")
    return pred


def predict_lag12_with_persistence_fallback(
    train: pd.DataFrame,
    ledger: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Same-calendar-month last-year TWS, falling back to legal persistence.

    For every validation target month m, lag-12 uses TWS at m-12. Since h<=7,
    that observation is older than the synthetic hidden-state window and is legal.
    """
    required_train = {"time", "lat", "lon", "TWS_t"}
    required_ledger = {"target_date", "lat", "lon", "last_observed_TWS"}
    if required_train.difference(train.columns):
        raise ValueError("Train missing columns required for lag-12 baseline")
    if required_ledger.difference(ledger.columns):
        raise ValueError("Ledger missing columns required for lag-12 baseline")

    lookup = train.loc[:, ["time", "lat", "lon", "TWS_t"]].copy()
    lookup["lookup_period"] = pd.to_datetime(lookup["time"]).dt.to_period("M")
    lookup = lookup.loc[:, ["lat", "lon", "lookup_period", "TWS_t"]].rename(
        columns={"TWS_t": "lag12_tws"}
    )

    x = ledger.loc[:, ["lat", "lon", "target_date", "last_observed_TWS"]].copy()
    x["_order"] = np.arange(len(x), dtype=np.int64)
    target_period = pd.to_datetime(x["target_date"]).dt.to_period("M")
    x["lookup_period"] = target_period - 12
    x = x.merge(
        lookup,
        how="left",
        on=["lat", "lon", "lookup_period"],
        validate="many_to_one",
        sort=False,
    ).sort_values("_order")

    available = x["lag12_tws"].notna().to_numpy()
    pred = x["lag12_tws"].fillna(x["last_observed_TWS"]).to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all():
        raise ValueError("Lag-12 baseline produced non-finite predictions")
    return pred, available


def predict_harmonic_with_persistence_fallback(
    model: HarmonicTrendModel,
    ledger: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    base, available = predict_harmonic_trend(model, ledger, date_column="target_date")
    persistence = predict_persistence(ledger)
    pred = np.where(available, base, persistence)
    return pred, available


def predict_seasonal_anomaly_persistence(
    model: HarmonicTrendModel,
    ledger: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast target baseline plus persistence of the last legal TWS anomaly.

    This is intentionally parameter-free at this stage: anomaly persistence rho=1.
    Horizon-specific shrinkage can be estimated later only if this baseline shows
    enough signal to justify another degree of freedom.
    """
    target_base, target_ok = predict_harmonic_trend(model, ledger, date_column="target_date")
    anchor_base, anchor_ok = predict_harmonic_trend(model, ledger, date_column="last_observed_date")
    available = target_ok & anchor_ok
    persistence = predict_persistence(ledger)

    pred = persistence.copy()
    if available.any():
        pred[available] = (
            target_base[available]
            + (ledger.loc[available, "last_observed_TWS"].to_numpy(dtype=np.float64) - anchor_base[available])
        )
    if not np.isfinite(pred).all():
        raise ValueError("Seasonal-anomaly persistence produced non-finite predictions")
    return pred, available
