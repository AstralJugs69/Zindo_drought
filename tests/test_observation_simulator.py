from __future__ import annotations

import unittest

import pandas as pd

from src.observation_simulator import ScenarioSpec, simulate_observations


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


if __name__ == "__main__":
    unittest.main()
