"""Reward for presence of both <reasoning> and <answer> structural tags."""

from __future__ import annotations

import re
from typing import Any

from llm_finetuning.core import BaseReward, RewardConfig


class StructureValidationReward(BaseReward):
    """Partial credit for having both <reasoning> and <answer> blocks."""

    def __init__(self) -> None:
        """Create the reward with a stable TRL logging name."""
        super().__init__(RewardConfig(name="structure_validation"))

    def __call__(self, prompts: list, completions: list, **kwargs: Any) -> list[float]:
        """Score completions for containing both `<reasoning>` and `<answer>` blocks."""
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            has_reasoning = bool(
                re.search(r"<reasoning>[\s\S]*?</reasoning>", response)
            )
            has_answer = bool(re.search(r"<answer>[\s\S]*?</answer>", response))
            scores.append((has_reasoning + has_answer) / 2.0)
        return scores
