"""Format reward: responses with sufficient non-empty lines."""

from __future__ import annotations

from typing import Any

from llm_finetuning.core import BaseReward, RewardConfig


class MultilineComplianceReward(BaseReward):
    """Reward responses with at least 5 non-empty lines."""

    def __init__(self) -> None:
        """Create the reward with a stable TRL logging name."""
        super().__init__(RewardConfig(name="multiline_compliance"))

    def __call__(self, prompts: list, completions: list, **kwargs: Any) -> list[float]:
        """Score completions by the number of non-empty lines (min 5)."""
        min_lines = 5
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            lines = [line for line in response.split("\n") if line.strip()]
            score = 1.0 if len(lines) >= min_lines else len(lines) / min_lines
            scores.append(score)
        return scores
