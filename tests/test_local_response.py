import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.local_response import LocalResponse, fit_local_response
from src.ml_features import HYDRO_GAP_SAFE_FEATURE_COLUMNS


class LocalResponseTests(unittest.TestCase):
    def test_local_slopes_recovery_unknown_fallback_and_serialization(self):
        rng = np.random.default_rng(73)
        x = pd.DataFrame(rng.normal(size=(600, len(HYDRO_GAP_SAFE_FEATURE_COLUMNS))), columns=HYDRO_GAP_SAFE_FEATURE_COLUMNS)
        x["lat"] = np.repeat([0., 1.], 300)
        x["lon"] = 0.
        x["h"] = rng.integers(1, 8, len(x))
        y = np.where(x.lat == 0, 1., -1.) * x.SOIL_MOISTURE_t
        model = fit_local_response(x, y.to_numpy(), np.ones(len(x)), alpha=10)
        local_error = np.mean((model.predict(x) - y) ** 2)
        global_error = np.mean((model.predict(x, local=False) - y) ** 2)
        self.assertLess(local_error, global_error * .1)
        unknown = x.iloc[:5].copy(); unknown["lat"] = 99.
        np.testing.assert_allclose(model.predict(unknown), model.predict(unknown, local=False))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"; model.save(path)
            np.testing.assert_allclose(LocalResponse.load(path).predict(x), model.predict(x), atol=1e-12)
        order = rng.permutation(len(x))
        shuffled = fit_local_response(x.iloc[order], y.to_numpy()[order], np.ones(len(x)), alpha=10)
        np.testing.assert_allclose(shuffled.predict(x), model.predict(x), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
