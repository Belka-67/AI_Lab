"""Проверки учебных сценариев: python -m unittest discover -s tests."""

import unittest

import numpy as np
import pandas as pd

from lab_core import (MODELS, class_scores, data_fingerprint, generate_data,
                      logistic_boundary, neighbor_details, train, tree_steps)
from missions import MISSIONS, mission_progress


class LabTests(unittest.TestCase):
    def setUp(self):
        self.test_data = generate_data(120, 20, 25, 0, 0, 0, 2026)

    def test_generator_changes_one_factor_at_a_time(self):
        base = generate_data(100, 20, 25, 0, 0, 0, 42)
        noisy = generate_data(100, 20, 25, 80, 0, 0, 42)
        mistakes = generate_data(100, 20, 25, 0, 30, 0, 42)
        unusual = generate_data(100, 20, 25, 0, 0, 20, 42)
        self.assertEqual((base["Верный класс"] == "Яблоко").sum(), 20)
        self.assertEqual((mistakes["Метка"] != mistakes["Верный класс"]).sum(), 30)
        self.assertEqual(unusual["Выброс"].sum(), 20)
        self.assertTrue(base[["Размер", "Округлость"]].equals(
            mistakes[["Размер", "Округлость"]]))
        self.assertTrue(base["Верный класс"].equals(noisy["Верный класс"]))
        self.assertFalse(base[["Размер", "Округлость"]].equals(
            noisy[["Размер", "Округлость"]]))
        self.assertTrue(base.loc[~unusual["Выброс"], ["Размер", "Округлость"]].equals(
            unusual.loc[~unusual["Выброс"], ["Размер", "Округлость"]]))
        self.assertTrue(base.equals(generate_data(100, 20, 25, 0, 0, 0, 42)))

    def test_imbalance_is_visible_in_class_metrics(self):
        scores = class_scores(self.test_data["Верный класс"], np.full(120, "Груша"))
        self.assertEqual(scores["correct"], 96)
        self.assertEqual(scores["by_class"]["Яблоко"], (0, 24))
        self.assertEqual(scores["by_class"]["Груша"], (96, 96))
        self.assertTrue(np.array_equal(scores["matrix"], [[0, 24], [0, 96]]))

    def test_three_model_explanations_match_their_predictions(self):
        training = generate_data(60, 50, 25, 0, 0, 0, 42)
        results = train(training, self.test_data, k=9, depth=8)
        point = pd.DataFrame({"Размер": [6.0], "Округлость": [7.0]})
        neighbors, _ = neighbor_details(results[MODELS[0]]["model"], training, point)
        self.assertEqual(len(neighbors), 9)
        self.assertEqual(neighbors["Метка"].value_counts().idxmax(),
                         results[MODELS[0]]["model"].predict(point)[0])
        self.assertTrue(tree_steps(results[MODELS[1]]["model"], point))
        weights, offset, classes = logistic_boundary(results[MODELS[2]]["model"])
        side = classes[1] if np.dot(weights, point.iloc[0]) + offset >= 0 else classes[0]
        self.assertEqual(side, results[MODELS[2]]["model"].predict(point)[0])
        for result in results.values():
            self.assertEqual(result["matrix"].sum(), 120)
            self.assertEqual(result["matrix"].trace(), result["test_correct"])

    def test_fewer_than_k_cards_still_uses_odd_vote(self):
        training = generate_data(6, 50, 25, 0, 0, 0, 42)
        results = train(training, self.test_data, k=9)
        self.assertEqual(results[MODELS[0]]["model"].steps[-1][1].n_neighbors, 5)

    def test_all_six_missions_require_real_experiments(self):
        def run(mid, **changes):
            p = dict(MISSIONS[mid][2], **changes)
            data = generate_data(*(p[k] for k in
                                   ("count", "apple", "spread", "measurement", "mistakes", "outliers")), 42)
            results = train(data, self.test_data, p["k"], p["depth"])
            return {"mission": mid, "seed": 42, "params": p,
                    "count": len(data), "fingerprint": data_fingerprint(data),
                    "test_pct": {n: round(100 * r["test_correct"] / 120, 1)
                                 for n, r in results.items()}}
        cases = {
            "few": [run("few"), run("few", count=60)],
            "labels": [run("labels"), run("labels", mistakes=30)],
            "balance": [run("balance")],
            "outliers": [run("outliers"), run("outliers", outliers=20)],
            "tree": [run("tree"), run("tree", depth=8)],
            "repair": [run("repair"), run("repair", count=80, apple=50,
                                           mistakes=0, outliers=0)],
        }
        for mid, runs in cases.items():
            with self.subTest(mission=mid):
                self.assertTrue(mission_progress(mid, runs)[0])
                self.assertFalse(mission_progress(mid, [])[0])
        self.assertFalse(mission_progress("few", [run("few"),
            run("few", count=60, mistakes=15)])[0])


if __name__ == "__main__":
    unittest.main()
