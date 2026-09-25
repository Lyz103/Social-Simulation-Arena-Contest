from __future__ import annotations

from typing import Any

from ._compat import field, string_list


def _bounded_number(request: Any) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "number"}
    minimum = field(request, "minimum", None)
    maximum = field(request, "maximum", None)
    if minimum is not None:
        schema["minimum"] = float(minimum)
    if maximum is not None:
        schema["maximum"] = float(maximum)
    return schema


def _normal_schema(request: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "mean": _bounded_number(request),
            "sd": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000000000},
        },
        "required": ["mean", "sd"],
        "additionalProperties": False,
    }


def topline_schema(request: Any) -> dict[str, Any]:
    return _normal_schema(request)


def profile_schema(request: Any) -> dict[str, Any]:
    cells = string_list(field(request, "cells"), "cells")
    return {
        "type": "object",
        "properties": {
            "cells": {
                "type": "object",
                "properties": {cell: _normal_schema(request) for cell in cells},
                "required": cells,
                "additionalProperties": False,
            }
        },
        "required": ["cells"],
        "additionalProperties": False,
    }


def ranking_schema(request: Any) -> dict[str, Any]:
    spec = field(request, "ranking_spec", {})
    if spec and not spec.get("items"):
        return {
            "type": "object",
            "properties": {
                "ranking": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 256},
                    "minItems": spec["length"],
                    "maxItems": spec["length"],
                    "uniqueItems": True,
                }
            },
            "required": ["ranking"],
            "additionalProperties": False,
        }
    candidates = string_list(field(request, "candidates"), "candidates")
    score_properties = {candidate: {"type": "number"} for candidate in candidates}
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": score_properties,
                "required": candidates,
                "additionalProperties": False,
            },
            "ranking": {
                "type": "array",
                "items": {"type": "string", "enum": candidates},
                "minItems": len(candidates),
                "maxItems": len(candidates),
                "uniqueItems": True,
            },
        },
        "required": ["scores", "ranking"],
        "additionalProperties": False,
    }
