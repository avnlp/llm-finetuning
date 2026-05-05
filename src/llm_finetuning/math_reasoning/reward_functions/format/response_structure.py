"""Format reward: reasoning → answer structure compliance."""

from __future__ import annotations

import re
from typing import Any

from llm_finetuning.core import BaseReward, RewardConfig


class ResponseStructureReward(BaseReward):
    """Partial credit for reasoning → answer structure."""

    def __init__(self) -> None:
        """Create the reward with a stable TRL logging name."""
        super().__init__(RewardConfig(name="response_structure"))

    def __call__(self, prompts: list, completions: list, **kwargs: Any) -> list[float]:
        """Score completions for containing `<reasoning>` and `<answer>` blocks."""
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            has_reasoning = bool(
                re.search(r"<reasoning>[\s\S]*?</reasoning>", response)
            )
            has_answer = bool(re.search(r"<answer>[\s\S]*?</answer>", response))
            scores.append((has_reasoning + has_answer) / 2.0)
        return scores
