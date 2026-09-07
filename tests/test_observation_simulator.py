from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.observation_simulator import ScenarioSpec, conservative_training_cutoff, simulate_observations
from src.metrics import TEST_H_COUNTS, score_by_horizon
from src.validation import build_test_mask_template


class ObservationSimulatorTests(unittest.TestCase):
    def _scenario(self) -> ScenarioSpec:
        return ScenarioSpec("synthetic", "test", "2020-01", "2020-04", "2019-12", "explicit")

    def _panel(self) -> pd.DataFrame:
        return pd.DataFrame({
            "sample_id": ["a", "b", "c", "d"],
            "time": ["2020-01-01", "2020-02-01", "2020-04-01", "2020-05-01"],
            "lat": [0.0] * 4, "lon": [0.0] * 4,
            "TWS_t": [1.0, 999.0, 3.0, 4.0],
            "tws_visible": [True, False, True, True],
        })

    def test_hidden_truth_and_future_changes_do_not_affect_earlier_state(self) -> None:
        panel = self._panel()
        labels = pd.DataFrame({"sample_id": panel.sample_id, "target": [1.1, 2.1, 3.1, 4.1]})
        one = simulate_observations(panel, score_ids=pd.Series(["b"]), labels=labels, scenario=self._scenario())
        changed = panel.copy()
        changed.loc[changed.sample_id.isin(["b", "c", "d"]), "TWS_t"] = [-4.0, 9999.0, -9999.0]
        two = simulate_observations(changed, score_ids=pd.Series(["b"]), labels=labels, scenario=self._scenario())
        self.assertEqual(one.ledger.loc[0, "last_observed_TWS"], 1.0)
        self.assertEqual(one.ledger.loc[0, "last_observed_TWS"], two.ledger.loc[0, "last_observed_TWS"])
        self.assertEqual(int(one.ledger.loc[0, "h"]), 2)

    def test_calendar_gap_is_not_compressed(self) -> None:
        panel = self._panel()
        panel.loc[panel.sample_id == "c", "tws_visible"] = False
        labels = pd.DataFrame({"sample_id": panel.sample_id, "target": [1.1, 2.1, 3.1, 4.1]})
        fold = simulate_observations(panel, score_ids=pd.Series(["c"]), labels=labels, scenario=self._scenario())
        self.assertEqual(int(fold.ledger.loc[0, "h"]), 4)

    def test_labels_are_not_features_and_align_after_shuffled_input(self) -> None:
        panel = self._panel().sample(frac=1, random_state=1)
        labels = pd.DataFrame({"sample_id": ["d", "c", "b", "a"], "target": [4.1, 3.1, 2.1, 1.1]})
        fold = simulate_observations(panel, score_ids=pd.Series(["b", "c"]), labels=labels, scenario=self._scenario())
        self.assertNotIn("target", fold.ledger.columns)
        self.assertEqual(fold.labels.sample_id.tolist(), fold.ledger.sample_id.tolist())

    def test_multiple_mask_cycles_keep_hidden_truth_out_of_later_visible_history(self) -> None:
        panel = pd.DataFrame({
            "sample_id": list("abcdef"),
            "time": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-05-01", "2020-06-01", "2020-08-01"],
            "lat": [0.0] * 6, "lon": [0.0] * 6,
            "TWS_t": [1.0, 200.0, 3.0, 500.0, 6.0, 800.0],
            "tws_visible": [True, False, True, False, True, False],
        })
        labels = pd.DataFrame({"sample_id": list("abcdef"), "target": np.arange(6, dtype=float)})
        fold = simulate_observations(panel, score_ids=pd.Series(["d", "f"]), labels=labels, scenario=self._scenario())
        d, f = fold.ledger.iloc[0], fold.ledger.iloc[1]
        self.assertEqual(d.last_observed_TWS, 3.0)
        self.assertEqual(d.previous_visible_TWS, 1.0)
        self.assertEqual(f.last_observed_TWS, 6.0)
        self.assertEqual(f.previous_visible_TWS, 3.0)
        self.assertEqual(f.older_visible_TWS, 1.0)
        changed = panel.copy()
        changed.loc[changed.sample_id.isin(["b", "d", "f"]), "TWS_t"] = [-2.0, -5.0, -8.0]
        changed_fold = simulate_observations(changed, score_ids=pd.Series(["d", "f"]), labels=labels, scenario=self._scenario())
        np.testing.assert_equal(
            fold.ledger.previous_visible_TWS.to_numpy(), changed_fold.ledger.previous_visible_TWS.to_numpy()
        )
        np.testing.assert_equal(
            fold.ledger.older_visible_TWS.to_numpy(), changed_fold.ledger.older_visible_TWS.to_numpy()
        )

    def test_h1_uses_current_visible_state_and_cutoff_is_strict(self) -> None:
        panel = self._panel()
        labels = pd.DataFrame({"sample_id": panel.sample_id, "target": [1.1, 2.1, 3.1, 4.1]})
        fold = simulate_observations(panel, score_ids=pd.Series(["a", "b"]), labels=labels, scenario=self._scenario())
        h1 = fold.ledger.loc[fold.ledger.sample_id == "a"].iloc[0]
        h2 = fold.ledger.loc[fold.ledger.sample_id == "b"].iloc[0]
        self.assertEqual(int(h1.h), 1)
        self.assertEqual(h1.last_observed_TWS, 1.0)
        self.assertEqual(int(h2.h), 2)
        self.assertNotEqual(h2.last_observed_TWS, 999.0)
        self.assertEqual(str(conservative_training_cutoff("2020-01")), "2019-12")

    def test_weighted_metric_mixes_mse_before_square_root(self) -> None:
        horizons = np.arange(1, 8)
        truth = np.zeros(7)
        prediction = np.arange(1, 8, dtype=float)
        score, details = score_by_horizon(truth, prediction, horizons)
        expected = np.sqrt(sum(TEST_H_COUNTS[h] / sum(TEST_H_COUNTS.values()) * h * h for h in range(1, 8)))
        self.assertAlmostEqual(score, expected)
        self.assertEqual(details["rows"].tolist(), [1] * 7)

    def test_real_test_template_reproduces_ids_and_horizon_counts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "Test.csv"
        if not path.is_file():
            self.skipTest("challenge Test.csv is not available in this checkout")
        test = pd.read_csv(path, usecols=["ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked"])
        template = build_test_mask_template(test)
        self.assertEqual(len(template), 280_961)
        self.assertEqual(template.ID.nunique(), len(template))
        self.assertEqual(template.template_h.value_counts().sort_index().to_dict(), TEST_H_COUNTS)


if __name__ == "__main__":
    unittest.main()
