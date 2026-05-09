"""Reward for presence of <reasoning> tags in model output."""

from __future__ import annotations

import re
from typing import Any

from llm_finetuning.core import BaseReward, RewardConfig


class ReasoningTagsReward(BaseReward):
    """Reward presence of exactly one <reasoning>...</reasoning> block."""

    def __init__(self) -> None:
        """Create the reward with a stable TRL logging name."""
        super().__init__(RewardConfig(name="reasoning_tags"))

    def __call__(self, prompts: list, completions: list, **kwargs: Any) -> list[float]:
        """Score completions for containing a complete `<reasoning>` block."""
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            has_tag = bool(re.search(r"<reasoning>[\s\S]*?</reasoning>", response))
            scores.append(1.0 if has_tag else 0.0)
        return scores
