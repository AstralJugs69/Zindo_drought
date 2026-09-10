"""Submission-compliant, coordinate-free B3-C feature schema.

The historical B3 builder is intentionally preserved for reproducibility of
older experiments.  B3-C calls that builder, verifies its complete schema, and
then selects an explicit ordered allowlist that excludes raw coordinates.  The
coordinates in the source panel are still used as indexing metadata by the
causal regional/local aggregators; they are never passed to the fitted model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from src.hydro_trajectory import HYDRO_COLUMNS, WINDOWS, trajectory_feature_names
from src.ml_features import HYDRO_GAP_SAFE_FEATURE_COLUMNS
from src.neural_sequence import B3FeatureMaps, build_b3_matrix
from src.regional_context import regional_feature_names


def _regional_trajectory_feature_names(widths: Iterable[float] = (5.0, 15.0)) -> list[str]:
    """Return the exact column order emitted by ``build_regional_trajectory_map``."""
    names: list[str] = []
    for width in widths:
        tag = f"{float(width):g}"
        names.extend(trajectory_feature_names(f"regional{tag}_"))
        for variable in HYDRO_COLUMNS:
            for window in WINDOWS:
                stem = f"{variable}_trail{window}"
                names.extend([
                    f"local_minus_reg{tag}_{stem}_mean",
                    f"local_minus_reg{tag}_{stem}_slope",
                    f"local_reg{tag}_{stem}_direction_disagree",
                ])
    return names


# This is the schema produced by ``src.neural_sequence.build_b3_matrix`` at the
# frozen B3-C baseline.  Keep it explicit so an upstream feature-order change
# cannot silently alter a supposedly compliant experiment.
ORIGINAL_B3_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    [
        *HYDRO_GAP_SAFE_FEATURE_COLUMNS,
        *regional_feature_names(),
        *trajectory_feature_names("local_"),
        *_regional_trajectory_feature_names(),
    ]
)

if len(ORIGINAL_B3_FEATURE_COLUMNS) != 446:
    raise AssertionError(
        f"unexpected historical B3 schema width: {len(ORIGINAL_B3_FEATURE_COLUMNS)}"
    )
if len(set(ORIGINAL_B3_FEATURE_COLUMNS)) != len(ORIGINAL_B3_FEATURE_COLUMNS):
    raise AssertionError("historical B3 schema contains duplicate feature names")

COORDINATE_COLUMNS = frozenset({"lat", "lon"})
COORDINATE_FREE_B3_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column for column in ORIGINAL_B3_FEATURE_COLUMNS if column not in COORDINATE_COLUMNS
)

if len(COORDINATE_FREE_B3_FEATURE_COLUMNS) != 444:
    raise AssertionError(
        "coordinate-free B3-C schema must remove exactly the two raw coordinates"
    )

# Exact names are the primary safety boundary.  This secondary check catches a
# future accidental rename that would otherwise evade a review by introducing a
# coordinate/location representation under a new spelling.
_PROHIBITED_NAME = re.compile(
    r"(?:^|[_-])(lat|lon|cell_id|location_id|coordinate|coord|embedding)(?:$|[_-])",
    flags=re.IGNORECASE,
)


def _provenance_group(column: str) -> str:
    if column in HYDRO_GAP_SAFE_FEATURE_COLUMNS:
        return "causal_hydro_gap_safe"
    if column.startswith(("reg5_", "dev5_", "count5_", "coverage5_")):
        return "observed_hydrology_regional_context_5deg"
    if column.startswith(("reg15_", "dev15_", "count15_", "coverage15_")):
        return "observed_hydrology_regional_context_15deg"
    if column.startswith("local_"):
        return "observed_hydrology_local_trajectory"
    if column.startswith("regional5_"):
        return "observed_hydrology_regional_trajectory_5deg"
    if column.startswith("regional15_"):
        return "observed_hydrology_regional_trajectory_15deg"
    if column.startswith("local_minus_reg5_") or column.startswith("local_reg5_"):
        return "observed_hydrology_local_minus_regional_5deg"
    if column.startswith("local_minus_reg15_") or column.startswith("local_reg15_"):
        return "observed_hydrology_local_minus_regional_15deg"
    raise AssertionError(f"no B3-C provenance group for {column!r}")


COORDINATE_FREE_B3_PROVENANCE: Mapping[str, str] = {
    column: _provenance_group(column) for column in COORDINATE_FREE_B3_FEATURE_COLUMNS
}


def coordinate_free_b3_feature_columns() -> tuple[str, ...]:
    """Return the frozen ordered B3-C model-input allowlist."""
    return COORDINATE_FREE_B3_FEATURE_COLUMNS


def coordinate_free_b3_feature_provenance() -> dict[str, str]:
    """Return a copy of the ordered feature-to-provenance map."""
    return dict(COORDINATE_FREE_B3_PROVENANCE)


def assert_coordinate_free_b3_schema(columns: Iterable[str]) -> None:
    """Reject any model matrix whose ordered columns differ from B3-C."""
    actual = list(columns)
    expected = list(COORDINATE_FREE_B3_FEATURE_COLUMNS)
    if len(actual) != len(set(actual)):
        raise AssertionError("B3-C model schema contains duplicate feature names")
    if actual != expected:
        missing = [column for column in expected if column not in actual]
        unexpected = [column for column in actual if column not in expected]
        raise AssertionError(
            "B3-C model schema/order mismatch: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    prohibited = [column for column in actual if _PROHIBITED_NAME.search(column)]
    if prohibited:
        raise AssertionError(f"B3-C model schema contains prohibited location inputs: {prohibited}")


def build_coordinate_free_b3_matrix(
    ledger: pd.DataFrame,
    source: pd.DataFrame,
    structural: pd.DataFrame,
    maps: B3FeatureMaps,
) -> pd.DataFrame:
    """Build B3-C from the legal causal source view and drop raw coordinates.

    ``build_b3_matrix`` remains the single implementation of the causal
    feature values.  The wrapper performs an exact pre-selection schema check,
    making it impossible for a newly added coordinate-bearing column to enter a
    fitted B3-C model unnoticed.
    """
    historical = build_b3_matrix(ledger, source, structural, maps)
    if list(historical.columns) != list(ORIGINAL_B3_FEATURE_COLUMNS):
        raise AssertionError(
            "historical B3 builder schema changed; review before rebuilding B3-C"
        )
    result = historical.loc[:, COORDINATE_FREE_B3_FEATURE_COLUMNS].copy()
    assert_coordinate_free_b3_schema(result.columns)
    return result.astype("float32", copy=False).reset_index(drop=True)


def write_coordinate_free_b3_audit(path: str | Path) -> None:
    """Persist the ordered allowlist and provenance used by an experiment."""
    output = {
        "schema": "B3-C",
        "historical_schema_width": len(ORIGINAL_B3_FEATURE_COLUMNS),
        "model_schema_width": len(COORDINATE_FREE_B3_FEATURE_COLUMNS),
        "removed_model_inputs": sorted(COORDINATE_COLUMNS),
        "ordered_allowlist": list(COORDINATE_FREE_B3_FEATURE_COLUMNS),
        "provenance": coordinate_free_b3_feature_provenance(),
        "coordinate_policy": (
            "lat/lon may be used only as source-panel indexing metadata for legal "
            "regional/local observed-hydrology aggregation; no raw coordinate, "
            "cell ID, coordinate encoding, or learned location embedding is a model input"
        ),
    }
    Path(path).write_text(json.dumps(output, indent=2, sort_keys=False) + "\n", encoding="utf-8")
