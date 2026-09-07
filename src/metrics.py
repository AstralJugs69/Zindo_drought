from __future__ import annotations

import math

import numpy as np
import pandas as pd


TEST_H_COUNTS = {
    1: 94048,
    2: 62576,
    3: 46777,
    4: 31076,
    5: 15560,
    6: 15479,
    7: 15445,
}

_TEST_TOTAL = float(sum(TEST_H_COUNTS.values()))
TEST_H_WEIGHTS = {h: n / _TEST_TOTAL for h, n in TEST_H_COUNTS.items()}


def score_by_horizon(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    horizons: np.ndarray | pd.Series,
) -> tuple[float, pd.DataFrame]:
    """Score predictions using the exact test horizon mixture.

    Each horizon first receives its own rowwise MSE on the available validation
    examples. Those horizon MSE values are mixed with the exact test row shares,
    and RMSE is applied once at the end. This mirrors the intended competition
    mixture more faithfully than averaging horizon RMSE values.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    h = np.asarray(horizons, dtype=np.int16)

    if not (len(y) == len(p) == len(h)):
        raise ValueError("y_true, y_pred and horizons must have equal length")
    if len(y) == 0:
        raise ValueError("Cannot score an empty prediction set")
    if not np.isfinite(y).all():
        raise ValueError("y_true contains non-finite values")
    if not np.isfinite(p).all():
        raise ValueError("y_pred contains non-finite values")

    rows = []
    weighted_mse = 0.0
    for horizon in range(1, 8):
        mask = h == horizon
        if not mask.any():
            raise ValueError(f"No validation rows for h={horizon}")
        err = p[mask] - y[mask]
        mse = float(np.mean(np.square(err)))
        rmse = math.sqrt(mse)
        bias = float(np.mean(err))
        weight = TEST_H_WEIGHTS[horizon]
        weighted_mse += weight * mse
        rows.append(
            {
                "h": horizon,
                "rows": int(mask.sum()),
                "test_weight": weight,
                "mse": mse,
                "rmse": rmse,
                "bias": bias,
            }
        )

    return math.sqrt(weighted_mse), pd.DataFrame(rows).set_index("h")


def raw_rmse(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if len(y) != len(p):
        raise ValueError("y_true and y_pred must have equal length")
    return float(np.sqrt(np.mean(np.square(p - y))))
