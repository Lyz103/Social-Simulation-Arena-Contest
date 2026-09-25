import math
import unittest

from ssa_harness.ensemble import BaselineFactory, EnsembleEngine, ForecastMember
from ssa_harness.ensemble.models import InvalidEnsembleMember, InvalidEnsembleWeights


class EnsembleTests(unittest.TestCase):
    def test_topline_default_baselines_and_mixture_variance(self):
        request = {
            "round_id": "number-1",
            "target_type": "continuous_normal",
            "history": [40, 41, 42],
            "minimum": 0,
            "maximum": 100,
        }
        result = EnsembleEngine().combine(
            request,
            llm_predictions=[{"mean": 44, "sd": 2}],
        )
        expected_mean = 0.60 * 44 + 0.25 * 42 + 0.15 * 41
        self.assertAlmostEqual(result.prediction["mean"], expected_mean)
        self.assertGreater(result.prediction["sd"], 2 * math.sqrt(0.60))
        self.assertAlmostEqual(sum(result.weights.values()), 1.0)

    def test_multiple_llms_share_llm_weight(self):
        request = {
            "round_id": "number-2",
            "target_type": "continuous_normal",
            "history": [10, 10, 10],
        }
        result = EnsembleEngine().combine(
            request,
            llm_predictions=[{"mean": 12, "sd": 1}, {"mean": 14, "sd": 1}],
        )
        self.assertAlmostEqual(result.weights["llm-1"], 0.30)
        self.assertAlmostEqual(result.weights["llm-2"], 0.30)

    def test_profile_combines_every_cell(self):
        request = {
            "round_id": "profile-1",
            "target_type": "profile_energy",
            "cells": ["a", "b"],
            "history": [
                {"cells": {"a": 10, "b": 20}},
                {"cells": {"a": 12, "b": 22}},
                {"cells": {"a": 14, "b": 24}},
            ],
            "minimum": 0,
            "maximum": 100,
        }
        llm = {
            "cells": {
                "a": {"mean": 16, "sd": 2},
                "b": {"mean": 26, "sd": 2},
            }
        }
        result = EnsembleEngine().combine(request, [llm])
        self.assertEqual(set(result.prediction["cells"]), {"a", "b"})
        self.assertGreater(result.prediction["cells"]["a"]["sd"], 0)

    def test_ranking_ensemble_uses_scores(self):
        request = {
            "round_id": "ranking-1",
            "target_type": "ranking_list",
            "candidates": ["A", "B", "C"],
            "history": [
                {"ranking": ["A", "B", "C"]},
                {"ranking": ["A", "B", "C"]},
            ],
        }
        llm = {
            "scores": {"A": 0, "B": 10, "C": 5},
            "ranking": ["B", "C", "A"],
        }
        result = EnsembleEngine().combine(request, [llm])
        self.assertEqual(set(result.prediction["ranking"]), {"A", "B", "C"})
        self.assertEqual(result.prediction["ranking"][0], "B")

    def test_explicit_baselines_support_mapping_adapter(self):
        request = {
            "round_id": "number-3",
            "target_type": "continuous_normal",
            "history": [],
        }
        baselines = [
            {
                "member_id": "persistence",
                "member_type": "persistence",
                "prediction": {"mean": 9, "sd": 1},
            },
            ForecastMember("recent_mean", "recent_mean", {"mean": 10, "sd": 1}),
        ]
        result = EnsembleEngine().combine(
            request,
            [{"model_id": "model-x", "prediction": {"mean": 11, "sd": 1}}],
            baselines,
        )
        self.assertIn("model-x", result.weights)

    def test_baseline_history_common_value_shape(self):
        request = {
            "target_type": "continuous_normal",
            "history": [{"value": 1}, {"value": 2}, {"value": 3}],
        }
        members = BaselineFactory().build(request)
        self.assertEqual(members[0].prediction["mean"], 3)
        self.assertEqual(members[1].prediction["mean"], 2)

    def test_default_baseline_sd_is_not_ensemble_floor(self):
        request = {
            "round_id": "number-4",
            "target_type": "continuous_normal",
            "history": [1, 2, 3],
        }
        result = EnsembleEngine().combine(request, [{"mean": 3, "sd": 1}])
        persistence = next(
            member for member in result.members if member.member_id == "persistence"
        )
        self.assertEqual(persistence.prediction["sd"], 0.25)

    def test_duplicate_ranking_history_is_not_accepted(self):
        request = {
            "target_type": "ranking_list",
            "candidates": ["A", "B", "C"],
            "history": [{"ranking": ["A", "A", "B", "C"]}],
        }
        with self.assertRaises(InvalidEnsembleMember):
            BaselineFactory().build(request)

    def test_explicit_empty_weight_config_fails(self):
        request = {
            "round_id": "number-5",
            "target_type": "continuous_normal",
            "history": [],
        }
        with self.assertRaises(InvalidEnsembleWeights):
            EnsembleEngine(weights={}).combine(
                request, [{"mean": 3, "sd": 1}], baselines=[]
            )


if __name__ == "__main__":
    unittest.main()
