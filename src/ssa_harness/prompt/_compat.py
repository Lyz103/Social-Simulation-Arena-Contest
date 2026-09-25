from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import PromptCompilationError


_MISSING = object()


def field(obj: Any, name: str, default: Any = _MISSING) -> Any:
    """Read a field from either a mapping or an attribute-based object."""
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
    elif hasattr(obj, name):
        return getattr(obj, name)

    if default is not _MISSING:
        return default
    raise PromptCompilationError(f"missing required field: {name}")


def target_type(request: Any) -> str:
    raw = str(field(request, "target_type")).strip().lower()
    aliases = {
        "topline": "continuous_normal",
        "number": "continuous_normal",
        "profile": "profile_energy",
        "ranking": "ranking_list",
    }
    return aliases.get(raw, raw)


def string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise PromptCompilationError(f"{name} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise PromptCompilationError(f"{name} must contain nonempty strings")
    output = list(value)
    if len(output) != len(set(output)):
        raise PromptCompilationError(f"{name} contains duplicates")
    return output
