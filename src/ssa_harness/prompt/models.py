from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ForecastError(Exception):
    """Base exception for forecast compilation and validation."""


class PromptCompilationError(ForecastError):
    """Raised when a request/context pair cannot be compiled safely."""


class UnsupportedTargetType(PromptCompilationError):
    """Raised for target types not supported by the harness."""


class InvalidModelOutput(ForecastError):
    """Raised when a model response violates the output contract."""


@dataclass(frozen=True)
class CompiledPrompt:
    round_id: str
    target_type: str
    system_message: str
    user_message: str
    output_schema: dict[str, Any]
    model_params: dict[str, Any]
    prompt_version: str
    context_hash: str
    prompt_hash: str

    def messages(self) -> list[dict[str, str]]:
        """Return provider-neutral chat messages."""
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.user_message},
        ]
