"""Frozen dataclass for prompt templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Immutable container for a three-turn conversation template.

    Attributes:
        system_prompt: Content of the system message.
        user_template: Python format string; placeholders filled from example fields.
        response_field: Name of the example field used as the assistant turn.

    Warning: user_template must use {field_name} placeholders only. Literal { or }
    in the template will raise KeyError/ValueError. GRPO loaders for math/code data
    should NOT use render_user() — build message dicts directly in format_example().
    """

    system_prompt: str
    user_template: str
    response_field: str = "answer"

    def render_user(self, **fields: Any) -> str:
        """Render the user turn by formatting the template with example fields.

        Args:
            **fields: Keyword arguments matching the placeholder names in
                ``user_template``. Unused keys are silently ignored; missing
                keys raise ``KeyError``.

        Returns:
            The formatted user message string.
        """
        return self.user_template.format(**fields)

    def to_messages(self, **fields: Any) -> list[dict[str, str]]:
        """Build a two-message (system/user) chat turn list for an example.

        Args:
            **fields: Keyword arguments forwarded to ``render_user()``. Must
                include all placeholder names used in ``user_template``.

        Returns:
            A list of two dicts: ``[{"role": "system", "content": ...},
            {"role": "user", "content": ...}]``. Suitable for passing directly
            to ``tokenizer.apply_chat_template``.
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.render_user(**fields)},
        ]
