import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.spatial_b2 import (  # noqa: E402
    FEATURE_NAMES,
    build_training_episodes,
    factor_design,
    fit_per_factor_ridge,
    fit_spatial_basis,
    load_bundle,
    predict_with_fallback,
    prefix_substantial_factors,
    save_bundle,
)


def _panel(months=30):
    periods = pd.period_range("2000-01", periods=months, freq="M")
    cells = [(0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (10.0, 10.0)]
    rows = []
    for ti, p in enumerate(periods):
        for ci, (lat, lon) in enumerate(cells):
            seasonal = np.sin(2 * np.pi * (p.month - 1) / 12)
            tws = 0.2 * ti + 0.4 * ci + seasonal * (1 + 0.1 * ci)
            target = 0.2 * (ti + 1) + 0.4 * ci + np.sin(2 * np.pi * p.month / 12) * (1 + 0.1 * ci)
            rows.append({
                "sample_id": f"{p}_{ci}", "time": str(p), "lat": lat, "lon": lon,
                "TWS_t": tws, "target": target,
                "month_sin": np.sin(2 * np.pi * p.month / 12),
                "month_cos": np.cos(2 * np.pi * p.month / 12),
                "SPEI_01_t": 0.1 * ti + ci,
                "SPEI_03_t": 0.2 * ti - ci,
                "SPEI_06_t": seasonal + ci,
                "SPEI_12_t": seasonal - ci,
                "SOIL_MOISTURE_t": 0.3 * ti + 0.5 * ci,
            })
    return pd.DataFrame(rows)


def _ledger(panel, source="2001-10", anchor="2001-09"):
    src = pd.Period(source, freq="M")
    anc = pd.Period(anchor, freq="M")
    s = panel.loc[pd.to_datetime(panel.time).dt.to_period("M") == src].copy().reset_index(drop=True)
    a = panel.loc[pd.to_datetime(panel.time).dt.to_period("M") == anc,
                  ["lat", "lon", "TWS_t"]].rename(columns={"TWS_t": "last_observed_TWS"})
    s = s.merge(a, on=["lat", "lon"], validate="one_to_one")
    return pd.DataFrame({
        "sample_id": s.sample_id,
        "source_date": src.to_timestamp(),
        "lat": s.lat, "lon": s.lon,
        "tws_visible": False,
        "last_observed_date": anc.to_timestamp(),
        "last_observed_TWS": s.last_observed_TWS,
        "h": (src + 1).ordinal - anc.ordinal,
    })


def test_basis_is_row_order_invariant_and_reports_rank():
    x = _panel()
    cutoff = pd.Period("2001-09", freq="M")
    a = fit_spatial_basis(x, cutoff=cutoff, rank=2)
    b = fit_spatial_basis(x.sample(frac=1.0, random_state=7), cutoff=cutoff, rank=2)
    assert a.rank == b.rank == 2
    assert a.cells == b.cells
    assert np.allclose(a.tws_mean, b.tws_mean)
    assert np.allclose(a.loadings @ a.loadings.T, b.loadings @ b.loadings.T)


def test_training_episodes_preserve_missing_calendar_gap():
    x = _panel()
    # Remove one whole calendar source field. h=2 for 2000-07 would require
    # the absent 2000-06 anchor, while h=3 can still use 2000-05.
    x = x.loc[pd.to_datetime(x.time).dt.to_period("M") != pd.Period("2000-06", freq="M")].copy()
    cutoff = pd.Period("2001-09", freq="M")
    basis = fit_spatial_basis(x, cutoff=cutoff, rank=2)
    e = build_training_episodes(x, basis, cutoff=cutoff, min_tws_coverage=0.95)
    src = pd.Period("2000-07", freq="M").ordinal
    rows = e.source_period == src
    hs = set(e.h[rows].astype(int))
    assert 2 not in hs
    assert 3 in hs
    j = np.flatnonzero(rows & (e.h == 3))[0]
    assert e.target_period[j] - e.anchor_period[j] == 3


def test_training_and_inference_design_are_identical_for_same_episode():
    x = _panel()
    cutoff = pd.Period("2001-09", freq="M")
    basis = fit_spatial_basis(x, cutoff=cutoff, rank=2)
    e = build_training_episodes(x, basis, cutoff=cutoff)
    i = 3
    j = 1
    train_design = factor_design(
        z_anchor=e.z_anchor[i:i+1], h=e.h[i:i+1], month_sin=e.month_sin[i:i+1],
        month_cos=e.month_cos[i:i+1], hydro_source=e.hydro_source[i:i+1],
        hydro_anchor=e.hydro_anchor[i:i+1], source_coverage=e.source_coverage[i:i+1],
        anchor_coverage=e.anchor_coverage[i:i+1], factor=j,
    )
    same_design = factor_design(
        z_anchor=e.z_anchor[i:i+1].copy(), h=e.h[i:i+1].copy(), month_sin=e.month_sin[i:i+1].copy(),
        month_cos=e.month_cos[i:i+1].copy(), hydro_source=e.hydro_source[i:i+1].copy(),
        hydro_anchor=e.hydro_anchor[i:i+1].copy(), source_coverage=e.source_coverage[i:i+1].copy(),
        anchor_coverage=e.anchor_coverage[i:i+1].copy(), factor=j,
    )
    assert train_design.shape[1] == len(FEATURE_NAMES)
    assert np.array_equal(train_design, same_design)


def test_inference_ignores_target_future_and_raw_tws_mutations():
    x = _panel()
    cutoff = pd.Period("2001-09", freq="M")
    basis = fit_spatial_basis(x, cutoff=cutoff, rank=2)
    episodes = build_training_episodes(x, basis, cutoff=cutoff)
    model = fit_per_factor_ridge(episodes, alpha=10.0)
    factors = prefix_substantial_factors(x, basis, cutoff=cutoff)
    ledger = _ledger(x, source="2001-10", anchor="2001-08")
    fallback = np.arange(len(ledger), dtype=float) + 100.0

    p1, prov1 = predict_with_fallback(ledger=ledger, source=x, basis=basis, model=model,
                                      prefix_factors=factors, fallback_prediction=fallback)
    mutated = x.copy()
    mutated["target"] = 1e9
    mutated["TWS_t"] = -1e9
    future = pd.to_datetime(mutated.time).dt.to_period("M") > pd.Period("2001-10", freq="M")
    for c in ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]:
        mutated.loc[future, c] = 9e8
    p2, prov2 = predict_with_fallback(ledger=ledger, source=mutated, basis=basis, model=model,
                                      prefix_factors=factors, fallback_prediction=fallback)
    assert np.allclose(p1, p2)
    assert prov1.equals(prov2)


def test_bundle_roundtrip_reproduces_predictions(tmp_path):
    x = _panel()
    cutoff = pd.Period("2001-09", freq="M")
    basis = fit_spatial_basis(x, cutoff=cutoff, rank=2)
    e = build_training_episodes(x, basis, cutoff=cutoff)
    model = fit_per_factor_ridge(e, alpha=10.0)
    before = model.predict(z_anchor=e.z_anchor, h=e.h, month_sin=e.month_sin,
                           month_cos=e.month_cos, hydro_source=e.hydro_source,
                           hydro_anchor=e.hydro_anchor, source_coverage=e.source_coverage,
                           anchor_coverage=e.anchor_coverage)
    path = tmp_path / "bundle"
    save_bundle(path, basis, model, {"purpose": "roundtrip"})
    b2, m2, meta = load_bundle(path)
    after = m2.predict(z_anchor=e.z_anchor, h=e.h, month_sin=e.month_sin,
                       month_cos=e.month_cos, hydro_source=e.hydro_source,
                       hydro_anchor=e.hydro_anchor, source_coverage=e.source_coverage,
                       anchor_coverage=e.anchor_coverage)
    assert meta["purpose"] == "roundtrip"
    assert b2.cells == basis.cells
    assert np.allclose(before, after)


def test_explicit_fallback_when_anchor_is_not_substantial():
    x = _panel()
    cutoff = pd.Period("2001-09", freq="M")
    basis = fit_spatial_basis(x, cutoff=cutoff, rank=2)
    e = build_training_episodes(x, basis, cutoff=cutoff)
    model = fit_per_factor_ridge(e, alpha=10.0)
    ledger = _ledger(x, source="2001-10", anchor="2001-09")
    fallback = np.arange(len(ledger), dtype=float) + 50.0
    pred, prov = predict_with_fallback(ledger=ledger, source=x, basis=basis, model=model,
                                       prefix_factors={}, fallback_prediction=fallback)
    assert np.array_equal(pred, fallback)
    assert set(prov.fallback_reason) == {"anchor_not_substantial"}
    assert not prov.supported.any()
