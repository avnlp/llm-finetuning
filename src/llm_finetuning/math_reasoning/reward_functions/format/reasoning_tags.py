"""Format reward: presence of <reasoning> and <answer> XML tags."""

from __future__ import annotations

from typing import Any

from llm_finetuning.core import BaseReward, RewardConfig


class ReasoningTagsReward(BaseReward):
    """Reward presence of <reasoning> and <answer> tags."""

    def __init__(self) -> None:
        """Create the reward with a stable TRL logging name."""
        super().__init__(RewardConfig(name="reasoning_tags"))

    def __call__(self, prompts: list, completions: list, **kwargs: Any) -> list[float]:
        """Score completions based on exact counts of required XML tags."""
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            score = 0.0
            score += 0.5 if response.count("<reasoning>") == 1 else -0.5
            score += 0.5 if response.count("</reasoning>") == 1 else -0.5
            score += 0.5 if response.count("<answer>") == 1 else -0.5
            score += 0.5 if response.count("</answer>") == 1 else -0.5
            scores.append(score)
        return scores
