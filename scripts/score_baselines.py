from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import (
    fit_location_harmonic_trend,
    predict_harmonic_with_persistence_fallback,
    predict_lag12_with_persistence_fallback,
    predict_persistence,
    predict_seasonal_anomaly_persistence,
)
from src.metrics import raw_rmse, score_by_horizon
from src.validation import build_direct_horizon_fold, recent_observed_month_blocks


def _score_model(
    *,
    fold_name: str,
    model_name: str,
    labels: pd.DataFrame,
    ledger: pd.DataFrame,
    pred: np.ndarray,
    native_available: np.ndarray | None = None,
) -> dict:
    y = labels["target"].to_numpy(dtype=np.float64)
    h = ledger["h"].to_numpy(dtype=np.int16)
    weighted, by_h = score_by_horizon(y, pred, h)
    result = {
        "fold": fold_name,
        "model": model_name,
        "weighted_rmse": weighted,
        "raw_rmse": raw_rmse(y, pred),
        "rows": int(len(y)),
        "native_coverage": (
            1.0 if native_available is None else float(np.mean(native_available))
        ),
        "by_h": {
            str(int(idx)): {
                "rows": int(row["rows"]),
                "rmse": float(row["rmse"]),
                "bias": float(row["bias"]),
            }
            for idx, row in by_h.iterrows()
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["dev1", "dev2", "dev3"],
        choices=["dev1", "dev2", "dev3", "lockbox"],
        help="Folds to score. Lockbox is deliberately excluded by default.",
    )
    args = parser.parse_args()

    train_path = args.data_dir / "Train.csv"
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    print("[1/4] Reading structural TWS columns...")
    train = pd.read_csv(
        train_path,
        usecols=["sample_id", "time", "lat", "lon", "TWS_t", "target"],
    )

    print("[2/4] Building development fold definitions...")
    blocks = recent_observed_month_blocks(train, block_size=18, n_blocks=4)
    block_map = dict(zip(["dev1", "dev2", "dev3", "lockbox"], blocks))

    if "lockbox" in args.folds:
        print("WARNING: lockbox scoring was explicitly requested.")
    else:
        print("Lockbox remains untouched.")

    all_results: list[dict] = []
    print("[3/4] Scoring causal baseline suite...")
    for fold_name in args.folds:
        t0 = time.perf_counter()
        months = block_map[fold_name]
        fold = build_direct_horizon_fold(train, months)
        ledger = fold.ledger
        labels = fold.labels

        # Alignment is a hard invariant before any score is trusted.
        if not np.array_equal(ledger["example_id"].to_numpy(), labels["example_id"].to_numpy()):
            raise AssertionError("Ledger/label alignment failed")
        if "target" in ledger.columns:
            raise AssertionError("Target leaked into baseline ledger")

        persistence = predict_persistence(ledger)
        all_results.append(
            _score_model(
                fold_name=fold_name,
                model_name="persistence",
                labels=labels,
                ledger=ledger,
                pred=persistence,
            )
        )

        lag12, lag12_ok = predict_lag12_with_persistence_fallback(train, ledger)
        all_results.append(
            _score_model(
                fold_name=fold_name,
                model_name="lag12_or_persistence",
                labels=labels,
                ledger=ledger,
                pred=lag12,
                native_available=lag12_ok,
            )
        )

        # Fit target-derived structural artifacts strictly before the validation
        # source block. No target column is passed to the fitter.
        harmonic = fit_location_harmonic_trend(
            train.loc[:, ["time", "lat", "lon", "TWS_t"]],
            fit_before=fold.source_start,
        )

        harmonic_pred, harmonic_ok = predict_harmonic_with_persistence_fallback(
            harmonic, ledger
        )
        all_results.append(
            _score_model(
                fold_name=fold_name,
                model_name="harmonic_trend_or_persistence",
                labels=labels,
                ledger=ledger,
                pred=harmonic_pred,
                native_available=harmonic_ok,
            )
        )

        anomaly_pred, anomaly_ok = predict_seasonal_anomaly_persistence(
            harmonic, ledger
        )
        all_results.append(
            _score_model(
                fold_name=fold_name,
                model_name="seasonal_anomaly_persistence",
                labels=labels,
                ledger=ledger,
                pred=anomaly_pred,
                native_available=anomaly_ok,
            )
        )

        elapsed = time.perf_counter() - t0
        print(f"\n=== {fold_name.upper()} ({fold.source_start} -> {fold.source_end}) ===")
        for result in [r for r in all_results if r["fold"] == fold_name]:
            print(
                f"{result['model']:<34} "
                f"weighted_RMSE={result['weighted_rmse']:.6f} "
                f"raw_RMSE={result['raw_rmse']:.6f} "
                f"native_coverage={result['native_coverage']:.4f}"
            )
            h_text = "  ".join(
                f"h{h}:{result['by_h'][str(h)]['rmse']:.4f}" for h in range(1, 8)
            )
            print("  " + h_text)
        print(f"Fold runtime: {elapsed:.1f}s")

    print("[4/4] Summarizing development scores...")
    summary_rows = [
        {
            "fold": r["fold"],
            "model": r["model"],
            "weighted_rmse": r["weighted_rmse"],
            "raw_rmse": r["raw_rmse"],
            "native_coverage": r["native_coverage"],
        }
        for r in all_results
    ]
    table = pd.DataFrame(summary_rows)
    print("\n=== BASELINE SCORE TABLE ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    dev_mean = (
        table.groupby("model", sort=False)["weighted_rmse"]
        .agg(["mean", "std", "min", "max"])
        .sort_values("mean")
    )
    print("\n=== MEAN ACROSS REQUESTED FOLDS ===")
    print(dev_mean.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\nJSON_RESULTS")
    print(json.dumps(all_results, indent=2))
    print("\nBASELINE SCORING PASSED: target-blind dev baselines scored with exact test-horizon weights.")


if __name__ == "__main__":
    main()
