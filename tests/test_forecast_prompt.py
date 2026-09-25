import json
import unittest

from ssa_harness.prompt import (
    ForecastPromptCompiler,
    InvalidModelOutput,
    PredictionValidator,
)
from ssa_harness.prompt.models import PromptCompilationError


class ForecastPromptTests(unittest.TestCase):
    def setUp(self):
        self.compiler = ForecastPromptCompiler()
        self.validator = PredictionValidator()
        self.base = {
            "round_id": "round-1",
            "question": "What is the next value?",
            "target_type": "continuous_normal",
            "unit": "percent",
            "asof": "2026-09-20T00:00:00Z",
            "forecast_for": "2026-09-27",
            "minimum": 0,
            "maximum": 100,
            "history": [40, 41, 42],
        }
        self.context = {
            "round_id": "round-1",
            "asof": "2026-09-20T00:00:00Z",
            "rendered_text": "Latest value: 42",
            "context_hash": "sha256:test",
        }

    def test_topline_prompt_is_deterministic(self):
        first = self.compiler.compile(self.base, self.context)
        second = self.compiler.compile(self.base, self.context)
        self.assertEqual(first, second)
        self.assertIn("2026-09-20", first.user_message)
        self.assertEqual(first.output_schema["required"], ["mean", "sd"])

    def test_context_asof_mismatch_fails(self):
        context = dict(self.context, asof="2026-09-21T00:00:00Z")
        with self.assertRaises(PromptCompilationError):
            self.compiler.compile(self.base, context)

    def test_profile_schema_requires_all_cells(self):
        request = dict(
            self.base,
            target_type="profile_energy",
            cells=["democrat", "independent", "republican"],
        )
        prompt = self.compiler.compile(request, self.context)
        cells = prompt.output_schema["properties"]["cells"]
        self.assertEqual(cells["required"], request["cells"])
        valid = {"cells": {cell: {"mean": 50, "sd": 2} for cell in request["cells"]}}
        self.assertEqual(
            set(self.validator.parse(valid, request)["cells"]), set(request["cells"])
        )
        del valid["cells"]["democrat"]
        with self.assertRaises(InvalidModelOutput):
            self.validator.parse(valid, request)

    def test_profile_rejects_non_object_cell(self):
        request = dict(
            self.base,
            target_type="profile_energy",
            cells=["a", "b"],
        )
        invalid = {
            "cells": {
                "a": {"mean": 10, "sd": 1},
                "b": 20,
            }
        }
        with self.assertRaises(InvalidModelOutput):
            self.validator.parse(invalid, request)

    def test_ranking_is_rebuilt_from_scores(self):
        request = dict(
            self.base,
            target_type="ranking_list",
            candidates=["A", "B", "C"],
        )
        result = self.validator.parse(
            {
                "scores": {"A": 1, "B": 3, "C": 2},
                "ranking": ["A", "B", "C"],
            },
            request,
        )
        self.assertEqual(result["ranking"], ["B", "C", "A"])

    def test_json_fence_is_accepted_but_extra_fields_are_not(self):
        parsed = self.validator.parse(
            '```json\n{"mean": 42, "sd": 1.5}\n```', self.base
        )
        self.assertEqual(parsed, {"mean": 42.0, "sd": 1.5})
        with self.assertRaises(InvalidModelOutput):
            self.validator.parse(
                json.dumps({"mean": 42, "sd": 1, "note": "x"}), self.base
            )


if __name__ == "__main__":
    unittest.main()
