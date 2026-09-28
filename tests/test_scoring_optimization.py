"""Counterexamples for official skill selection and target isolation."""

import copy
from datetime import datetime, timedelta, timezone
import unittest

from ssa_harness.contracts import ForecastRequest
from ssa_harness.evaluation import calibrate, crps_normal, score_records


def request(day="2026-08-01", **kwargs):
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    return ForecastRequest(
        **{
            "round_id": day,
            "question": "Next published approval rate",
            "target_type": "continuous_normal",
            "unit": "points",
            "source_name": "test polling series",
            "population": "all adults",
            "metric": "approval",
            "asof": start.isoformat(),
            "deadline": (start + timedelta(days=1)).isoformat(),
            **kwargs,
        }
    ).to_dict()


def cases():
    # At mean=truth, CRPS scales linearly with sd. The model reduces total
    # loss from 101*k to 52*k, but its skills -1 and +0.5 average to -0.25.
    result = []
    for day, model_sd, persistence_sd in (
        ("2026-08-01", 2, 1),
        ("2026-08-08", 50, 100),
        ("2026-09-01", 4, 2),
    ):
        observed = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        result.append(
            {
                "request": request(day),
                "truth": 0,
                "truth_available_at": (observed + timedelta(days=2)).isoformat(),
                "persistence": {"mean": 0, "sd": persistence_sd},
                "members": [
                    {
                        "member_id": "model",
                        "member_type": "llm",
                        "prediction": {"mean": 0, "sd": model_sd},
                    },
                    {
                        "member_id": "persistence",
                        "member_type": "persistence",
                        "prediction": {"mean": 0, "sd": persistence_sd},
                    },
                ],
            }
        )
    return result


def select(data, **kwargs):
    return calibrate(
        data,
        validation_start="2026-08-01T00:00:00Z",
        test_start="2026-09-01T00:00:00Z",
        weight_grid=[{"llm": 1, "persistence": 0}, {"llm": 0, "persistence": 1}],
        sd_scales=(1,),
        **kwargs,
    )


class OfficialSkillTests(unittest.TestCase):
    def test_mean_skill_and_legacy_aggregate_can_have_opposite_signs(self):
        records = [
            {
                **{k: case[k] for k in ("request", "truth", "persistence")},
                "prediction": case["members"][0]["prediction"],
            }
            for case in cases()[:2]
        ]
        out = score_records(records)
        self.assertAlmostEqual(out["mean_skill"], -0.25)
        summary = out["by_metric"]["crps"]
        self.assertAlmostEqual(summary["mean_skill"], -0.25)
        self.assertAlmostEqual(summary["aggregate_skill"], 1 - 52 / 101)
        self.assertEqual(summary["aggregate_skill_kind"], "loss_weighted_legacy")
        self.assertEqual(out["by_group"][0]["mean_skill"], summary["mean_skill"])

    def test_perfect_null_ranking_has_zero_skill_even_if_prediction_wrong(self):
        record = {
            "request": request(
                target_type="ranking_list",
                ranking_spec={"length": 2, "items": ["A", "B"]},
                unit="rank",
                metric="attention ranking",
            ),
            "prediction": {"ranking": ["B", "A"], "scores": {"B": 1, "A": 0}},
            "persistence": {"ranking": ["A", "B"], "scores": {"A": 1, "B": 0}},
            "truth": ["A", "B"],
        }
        out = score_records([record])
        self.assertEqual(out["records"][0]["loss"], 1)
        self.assertEqual(out["records"][0]["persistence_loss"], 0)
        self.assertEqual(out["records"][0]["skill"], 0)
        self.assertEqual(out["mean_skill"], 0)
        self.assertEqual(out["paired_count"], 1)
        self.assertIsNone(out["by_metric"]["kendall"]["aggregate_skill"])

    def test_missing_null_does_not_dilute_official_skill(self):
        item = cases()[0]
        record = {
            **{key: item[key] for key in ("request", "truth", "persistence")},
            "prediction": item["members"][0]["prediction"],
        }
        missing = {**record, "persistence": None}
        out = score_records([record, missing])
        self.assertEqual(out["paired_count"], 1)
        self.assertAlmostEqual(out["mean_skill"], -1)
        self.assertIsNone(out["records"][1]["skill"])
        self.assertIsNone(score_records([])["mean_skill"])

    def test_raw_losses_are_grouped_by_full_target_definition(self):
        records = []
        definitions = [
            {},
            {"metric": "disapproval"},
            {"unit": "fraction"},
            {"source_name": "another series"},
            {"population": "voters"},
        ]
        for definition in definitions:
            records.append(
                {
                    "request": request(**definition),
                    "truth": 0,
                    "prediction": {"mean": 0, "sd": 1},
                    "persistence": {"mean": 0, "sd": 2},
                }
            )
        out = score_records(records)
        self.assertEqual(len(out["by_group"]), len(definitions))
        self.assertFalse(out["by_metric"]["crps"]["raw_loss_comparable"])
        self.assertTrue(all(group["raw_loss_comparable"] for group in out["by_group"]))
        self.assertTrue(all(group["count"] == 1 for group in out["by_group"]))


class CalibrationObjectiveTests(unittest.TestCase):
    def test_official_objective_can_choose_different_weights_than_mean_loss(self):
        data = cases()
        mean_loss = select(data)
        mean_skill = select(data, objective="mean_skill")
        self.assertEqual(mean_loss["objective"], "mean_loss")
        self.assertEqual(mean_loss["selected"]["weights"]["llm"], 1)
        self.assertAlmostEqual(mean_loss["selected"]["validation_mean_skill"], -0.25)
        self.assertEqual(mean_skill["selected"]["weights"]["persistence"], 1)
        self.assertAlmostEqual(mean_skill["selected"]["validation_mean_skill"], 0)

    def test_official_objective_requires_explicit_null_in_each_partition(self):
        for missing_index in (0, 2):
            with self.subTest(missing_index=missing_index):
                data = cases()
                del data[missing_index]["persistence"]
                # A persistence ensemble member is present, but it must not be
                # substituted for the missing official null.
                with self.assertRaisesRegex(ValueError, "explicit official case.persistence"):
                    select(data, objective="mean_skill")

    def test_explicit_official_test_null_overrides_member_for_both_objectives(self):
        for objective in ("mean_loss", "mean_skill"):
            with self.subTest(objective=objective):
                data = cases()
                data[-1]["persistence"] = {"mean": 0, "sd": 7}
                out = select(data, objective=objective)
                row = out["test"]["records"][0]
                self.assertAlmostEqual(row["persistence_loss"], crps_normal(0, 7, 0))
                self.assertEqual(row["persistence_source"], "case.persistence")

    def test_legacy_member_null_is_labeled_and_explicit_none_is_respected(self):
        data = cases()
        del data[-1]["persistence"]
        out = select(data)
        self.assertIn("not an official null", out["test"]["records"][0]["persistence_source"])
        data[-1]["persistence"] = None
        out = select(data)
        self.assertIsNone(out["test"]["mean_skill"])

    def test_same_scoring_rule_different_target_metrics_cannot_be_calibrated_together(self):
        data = cases()
        data[1]["request"]["metric"] = "disapproval"
        with self.assertRaisesRegex(ValueError, "target metric"):
            select(data)

    def test_test_truth_and_null_cannot_change_selected_settings(self):
        data = cases()
        before = select(data, objective="mean_skill")
        changed = copy.deepcopy(data)
        changed[-1]["truth"] = 10000
        changed[-1]["persistence"] = {"mean": 123, "sd": 1000}
        after = select(changed, objective="mean_skill")
        self.assertEqual(before["selected"], after["selected"])
        self.assertEqual(before["validation_candidates"], after["validation_candidates"])
        self.assertNotEqual(before["test"]["mean_skill"], after["test"]["mean_skill"])

    def test_validation_truth_available_at_test_boundary_is_rejected(self):
        data = cases()
        data[1]["truth_available_at"] = "2026-09-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "validation outcome not known"):
            select(data, objective="mean_skill")

    def test_already_known_truth_at_forecast_cutoff_is_rejected(self):
        data = cases()
        data[-1]["truth_available_at"] = data[-1]["request"]["asof"]
        with self.assertRaisesRegex(ValueError, "truth must be unavailable"):
            select(data, objective="mean_skill")


if __name__ == "__main__":
    unittest.main()
