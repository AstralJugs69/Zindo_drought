import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.run_spatial_phase_a import _basis  # noqa: E402


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


def test_sse_decomposition_identity():
    e = np.array([1.0, 0.0])
    p = np.array([1.0, 0.0])
    raw = np.square(e).sum()
    remaining = np.square(e - p).sum()
    explained = raw - remaining
    assert np.isclose(raw, explained + remaining)
