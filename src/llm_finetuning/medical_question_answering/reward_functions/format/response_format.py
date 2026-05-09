"""Reward for response length falling within an acceptable range."""

from __future__ import annotations

from typing import Any

from llm_finetuning.core import BaseReward, RewardConfig


class ResponseFormatReward(BaseReward):
    """Reward responses within a reasonable length range."""

    def __init__(self) -> None:
        """Create the reward with a stable TRL logging name."""
        super().__init__(RewardConfig(name="response_format"))

    def __call__(self, prompts: list, completions: list, **kwargs: Any) -> list[float]:
        """Score completions based on response length (target 50–2000 chars)."""
        min_len, max_len = 50, 2000
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            length = len(response)
            if min_len <= length <= max_len:
                scores.append(1.0)
            elif length < min_len:
                scores.append(length / min_len * 0.5)
            else:
                scores.append(max_len / length * 0.5)
        return scores
