"""Format reward: numbered or bulleted reasoning steps."""

from __future__ import annotations

import re
from typing import Any

from llm_finetuning.core import BaseReward, RewardConfig


class StepFormatReward(BaseReward):
    """Reward numbered or bulleted steps (min 3)."""

    def __init__(self) -> None:
        """Create the reward with a stable TRL logging name."""
        super().__init__(RewardConfig(name="step_format"))

    def __call__(self, prompts: list, completions: list, **kwargs: Any) -> list[float]:
        """Score completions by counting step-like lines (min 3)."""
        min_steps = 3
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            steps = re.findall(
                r"^(?:Step\s*\d+|\d+\.|[-*•]\s*)", response, re.MULTILINE
            )
            score = 1.0 if len(steps) >= min_steps else len(steps) / min_steps
            scores.append(score)
        return scores
