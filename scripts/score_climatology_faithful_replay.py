from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import predict_persistence
from src.metrics import score_by_horizon
from src.ml_features import validation_horizon_weights
from src.validation import (
    build_exact_historical_mask_fold,
    build_test_mask_template,
    find_exact_template_starts,
)


def _fit_location_month_climatology(
    train: pd.DataFrame,
    *,
    fit_before: pd.Period,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit target-blind TWS climatology strictly before the replay window."""
    x = train.loc[:, ["time", "lat", "lon", "TWS_t"]].copy()
    x["period"] = pd.to_datetime(x["time"]).dt.to_period("M")
    x = x.loc[(x["period"] < fit_before) & x["TWS_t"].notna()].copy()
    if x.empty:
        raise AssertionError("No causal TWS history available for climatology")
    x["month"] = x["period"].dt.month.astype(np.int8)

    cell_month = (
        x.groupby(["lat", "lon", "month"], sort=False)["TWS_t"]
        .agg(clim_mean="mean", clim_std="std", clim_count="size")
        .reset_index()
    )
    cell = (
        x.groupby(["lat", "lon"], sort=False)["TWS_t"]
        .agg(cell_mean="mean", cell_std="std", cell_count="size")
        .reset_index()
    )
    month = (
        x.groupby("month", sort=False)["TWS_t"]
        .agg(global_month_mean="mean", global_month_std="std", global_month_count="size")
        .reset_index()
    )
    return cell_month, cell, month


def _lookup_climatology(
    rows: pd.DataFrame,
    *,
    date_column: str,
    cell_month: pd.DataFrame,
    cell: pd.DataFrame,
    month: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    base = rows.loc[:, ["lat", "lon", date_column]].copy()
    base["_order"] = np.arange(len(base), dtype=np.int64)
    base["month"] = pd.to_datetime(base[date_column]).dt.month.astype(np.int8)
    base = base.merge(
        cell_month,
        how="left",
        on=["lat", "lon", "month"],
        validate="many_to_one",
        sort=False,
    )
    native = base["clim_mean"].notna()
    base = base.merge(
        cell,
        how="left",
        on=["lat", "lon"],
        validate="many_to_one",
        sort=False,
    )
    base = base.merge(
        month,
        how="left",
        on="month",
        validate="many_to_one",
        sort=False,
    ).sort_values("_order")

    mean = (
        base["clim_mean"]
        .fillna(base["cell_mean"])
        .fillna(base["global_month_mean"])
        .to_numpy(dtype=np.float64)
    )
    std = (
        base["clim_std"]
        .fillna(base["cell_std"])
        .fillna(base["global_month_std"])
        .fillna(1.0)
        .to_numpy(dtype=np.float64)
    )
    count = base["clim_count"].fillna(0).to_numpy(dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise AssertionError("Climatology lookup produced non-finite values")
    return mean, std, count, float(native.mean())


def _weighted_optimal_rho(
    y: np.ndarray,
    target_clim: np.ndarray,
    anchor_anomaly: np.ndarray,
    weights: np.ndarray,
) -> float:
    residual = y - target_clim
    denom = float(np.sum(weights * anchor_anomaly * anchor_anomaly))
    if denom <= 0:
        return 0.0
    return float(np.sum(weights * anchor_anomaly * residual) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Kaggle-only causal diagnostic for discrete cell-by-calendar-month TWS "
            "climatology on the latest exact Test-calendar replay. No ML model training."
        )
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()

    print("[1/6] Reading structural Train/Test columns...")
    train = pd.read_csv(
        args.data_dir / "Train.csv",
        usecols=["sample_id", "time", "lat", "lon", "TWS_t", "target"],
    )
    test = pd.read_csv(
        args.data_dir / "Test.csv",
        usecols=["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"],
    )

    print("[2/6] Building latest exact historical Test-calendar replay...")
    template = build_test_mask_template(test)
    starts = find_exact_template_starts(train, template)
    if not starts:
        raise AssertionError("No exact historical Test replay available")
    start = starts[-1]
    fold = build_exact_historical_mask_fold(train, template, start)
    print("Replay:", fold.start_month, "->", fold.end_month)

    print("[3/6] Fitting causal cell x calendar-month climatology...")
    cell_month, cell, month = _fit_location_month_climatology(train, fit_before=start)
    print(
        json.dumps(
            {
                "fit_before": str(start),
                "cell_month_groups": int(len(cell_month)),
                "cell_groups": int(len(cell)),
                "median_cell_month_count": float(cell_month["clim_count"].median()),
                "min_cell_month_count": int(cell_month["clim_count"].min()),
                "max_cell_month_count": int(cell_month["clim_count"].max()),
            },
            indent=2,
        )
    )

    print("[4/6] Building target/anchor climatology predictions...")
    target_clim, target_std, target_count, target_native = _lookup_climatology(
        fold.ledger,
        date_column="target_date",
        cell_month=cell_month,
        cell=cell,
        month=month,
    )
    anchor_clim, anchor_std, anchor_count, anchor_native = _lookup_climatology(
        fold.ledger,
        date_column="last_observed_date",
        cell_month=cell_month,
        cell=cell,
        month=month,
    )
    anchor = fold.ledger["last_observed_TWS"].to_numpy(dtype=np.float64)
    anchor_anomaly = anchor - anchor_clim
    seasonal_step = target_clim - anchor_clim

    labels = fold.labels.set_index("sample_id").loc[fold.ledger["sample_id"]]
    y = labels["target"].to_numpy(dtype=np.float64)
    h = fold.ledger["h"].to_numpy(dtype=np.int8)
    weights = validation_horizon_weights(h).astype(np.float64)

    persistence = predict_persistence(fold.ledger)
    rho_star = _weighted_optimal_rho(y, target_clim, anchor_anomaly, weights)
    print(
        json.dumps(
            {
                "target_climatology_native_coverage": target_native,
                "anchor_climatology_native_coverage": anchor_native,
                "target_clim_count_median": float(np.median(target_count)),
                "anchor_clim_count_median": float(np.median(anchor_count)),
                "anchor_anomaly_std": float(np.std(anchor_anomaly)),
                "seasonal_step_std": float(np.std(seasonal_step)),
                "diagnostic_optimal_rho_unclipped": rho_star,
            },
            indent=2,
        )
    )

    print("[5/6] Scoring climatology baselines...")
    candidates: list[tuple[str, np.ndarray]] = [
        ("persistence", persistence),
        ("target_month_climatology", target_clim),
    ]
    for rho in (0.25, 0.50, 0.75, 1.00, 1.25):
        candidates.append(
            (
                f"climatology_plus_anchor_anomaly_rho{rho:.2f}",
                target_clim + rho * anchor_anomaly,
            )
        )
    candidates.append(
        (
            "climatology_plus_anchor_anomaly_rho_star",
            target_clim + rho_star * anchor_anomaly,
        )
    )

    results = []
    for name, pred in candidates:
        score, by_h = score_by_horizon(y, pred, h)
        row = {
            "model": name,
            "weighted_rmse": float(score),
            "by_h": {
                int(k): float(by_h.loc[k, "rmse"])
                for k in range(1, 8)
            },
        }
        results.append(row)
        print(f"{name:<48} weighted_RMSE={score:.6f}")

    print("[6/6] Summary...")
    best_fixed = min(results[:-1], key=lambda r: r["weighted_rmse"])
    report = {
        "replay_start": str(fold.start_month),
        "replay_end": str(fold.end_month),
        "rows": int(len(fold.ledger)),
        "target_climatology_native_coverage": target_native,
        "anchor_climatology_native_coverage": anchor_native,
        "diagnostic_optimal_rho_unclipped": rho_star,
        "best_fixed_candidate": best_fixed,
        "results": results,
    }
    print("JSON_RESULTS")
    print(json.dumps(report, indent=2))
    print("\nCLIMATOLOGY REPLAY COMPLETE")


if __name__ == "__main__":
    main()
