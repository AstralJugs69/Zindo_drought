import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.run_spatial_phase_a import (  # noqa: E402
    _aggregate_projection_opportunity,
    _basis,
    _error_budget,
    _project_residuals_by_date,
)


def test_basis_ignores_post_cutoff_and_target():
    rows = []
    for t, a in [("2000-01", 1.0), ("2000-02", 2.0), ("2001-01", 999.0)]:
        for lat, lon in [(0.0, 0.0), (10.0, 10.0)]:
            rows.append({"time": t, "lat": lat, "lon": lon, "TWS_t": a + lat, "target": 7.0})
    x = pd.DataFrame(rows)
    _, u1, m1 = _basis(x, pd.Period("2001-01", freq="M"), 2)
    x.loc[x.time == "2001-01", "TWS_t"] = -12345
    x["target"] = np.random.default_rng(1).normal(size=len(x))
    _, u2, m2 = _basis(x, pd.Period("2001-01", freq="M"), 2)
    assert m1 == m2
    assert np.allclose(np.abs(u1), np.abs(u2))


def _oof_two_dates(with_unsupported=False):
    rows = []
    for date, sign in [("2001-01-01", 1.0), ("2001-02-01", -1.0)]:
        rows.extend([
            {"source_date": date, "lat": 0.0, "lon": 0.0, "truth": sign, "prediction": 0.0},
            {"source_date": date, "lat": 0.0, "lon": 10.0, "truth": -sign, "prediction": 0.0},
        ])
        if with_unsupported:
            rows.append({"source_date": date, "lat": 99.0, "lon": 99.0, "truth": 2.0, "prediction": 0.0})
    return pd.DataFrame(rows)


def test_projection_is_fit_independently_by_date_so_opposite_maps_do_not_cancel():
    oof = _oof_two_dates()
    basis = np.array([[1.0], [-1.0]]) / np.sqrt(2.0)
    lookup = {(0.0, 0.0): 0, (0.0, 10.0): 1}
    rows = _project_residuals_by_date(oof, basis, lookup, origin="x", requested_rank=1)
    assert len(rows) == 2
    assert all(np.isclose(r["fraction_residual_sse"], 1.0) for r in rows)
    # A single pooled coefficient would cancel because the maps have opposite signs.
    pooled_residual = (oof.truth - oof.prediction).to_numpy(float)
    pooled_design = np.tile(basis, (2, 1))
    coef, *_ = np.linalg.lstsq(pooled_design, pooled_residual, rcond=None)
    assert np.allclose(coef, 0.0)


def test_all_finite_denominator_is_invariant_to_number_of_requested_ranks():
    oof = _oof_two_dates()
    budget, _ = _error_budget(oof)
    budget = pd.DataFrame([{"origin": "x", **r} for r in budget])
    base = pd.DataFrame([
        {"origin": "x", "rank": 1, "rows": 2, "unsupported_rows": 0, "raw_sse": 2.0,
         "explained_sse": 1.0, "remaining_sse": 1.0, "rank_used": 1},
        {"origin": "x", "rank": 1, "rows": 2, "unsupported_rows": 0, "raw_sse": 2.0,
         "explained_sse": 1.0, "remaining_sse": 1.0, "rank_used": 1},
    ])
    one = _aggregate_projection_opportunity(base, budget)
    doubled = pd.concat([base, base.assign(rank=2, rank_used=2)], ignore_index=True)
    two = _aggregate_projection_opportunity(doubled, budget)
    assert np.isclose(one.loc[0, "all_finite_sse"], two.loc[two["rank"] == 1, "all_finite_sse"].iloc[0])
    assert np.isclose(two.loc[two["rank"] == 1, "all_finite_sse"].iloc[0],
                      two.loc[two["rank"] == 2, "all_finite_sse"].iloc[0])


def test_unsupported_cells_stay_in_all_finite_denominator():
    oof = _oof_two_dates(with_unsupported=True)
    basis = np.array([[1.0], [-1.0]]) / np.sqrt(2.0)
    lookup = {(0.0, 0.0): 0, (0.0, 10.0): 1}
    proj = pd.DataFrame(_project_residuals_by_date(oof, basis, lookup, origin="x", requested_rank=1))
    budget, _ = _error_budget(oof)
    agg = _aggregate_projection_opportunity(proj, pd.DataFrame([{"origin": "x", **r} for r in budget]))
    assert int(agg.loc[0, "unsupported_rows"]) == 2
    assert agg.loc[0, "all_finite_sse"] > agg.loc[0, "supported_sse"]
    assert agg.loc[0, "all_finite_opportunity"] < agg.loc[0, "supported_opportunity"]


def test_requested_and_actual_rank_metadata_are_distinct():
    oof = _oof_two_dates()
    basis = np.eye(2)[:, :1]
    lookup = {(0.0, 0.0): 0, (0.0, 10.0): 1}
    rows = _project_residuals_by_date(oof, basis, lookup, origin="x", requested_rank=16)
    assert all(r["requested_rank"] == 16 for r in rows)
    assert all(r["rank"] == 16 for r in rows)
    assert all(r["rank_used"] == 1 for r in rows)
