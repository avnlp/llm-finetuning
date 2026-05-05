"""Dataset loading and formatting for math reasoning GRPO training."""

from __future__ import annotations

import re
from typing import Any

from llm_finetuning.core import BaseDatasetLoader, DatasetConfig


MATH_REASONING_SYSTEM_PROMPT = (
    "You are a math expert. Solve the problem step by step inside "
    "<reasoning> tags. Then give your final numeric answer inside <answer> tags."
)


class GSM8KLoader(BaseDatasetLoader):
    """Load and format GSM8K examples for GRPO training."""

    CONFIG = DatasetConfig(dataset_id="openai/gsm8k", subset="main")
    SYSTEM_PROMPT = MATH_REASONING_SYSTEM_PROMPT

    def __init__(self, config: DatasetConfig | None = None) -> None:
        """Create a GSM8K loader with the default dataset config."""
        super().__init__(config or self.CONFIG)

    def format_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a GSM8K row to `{'prompt': ..., 'answer': ...}` for GRPO."""
        return {
            "prompt": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": self._extract_answer(example["answer"]),
        }

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extract the final answer from a GSM8K solution string.

        Args:
            text: Full GSM8K answer text, typically ending with a line
                prefixed by ``####``.

        Returns:
            The stripped answer after the ``####`` marker when present;
            otherwise the stripped original text.
        """
        match = re.search(r"####\s*(.+)", text)
        return match.group(1).strip() if match else text.strip()
