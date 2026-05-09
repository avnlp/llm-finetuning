"""Correctness reward: numeric answer extracted from <answer> tags."""

from __future__ import annotations

import re
from typing import Any

from llm_finetuning.core import BaseReward, RewardConfig


class AnswerCorrectnessReward(BaseReward):
    """Reward correct numeric answer extracted from <answer> tags."""

    def __init__(self) -> None:
        """Create the reward with a stable TRL logging name."""
        super().__init__(RewardConfig(name="answer_correctness"))

    def __call__(self, prompts: list, completions: list, **kwargs: Any) -> list[float]:
        """Score completions by numeric match against `kwargs['answer']`."""
        answers = kwargs["answer"]
        scores = []
        for completion, true_answer in zip(completions, answers):
            response = completion[0]["content"]
            guess = self._extract_answer(response)
            if guess is None:
                scores.append(0.0)
                continue
            if guess.strip() == true_answer.strip():
                scores.append(3.0)
                continue
            try:
                ratio = float(guess) / float(true_answer)
                if 0.9 <= ratio <= 1.1:
                    scores.append(0.5)
                elif 0.8 <= ratio <= 1.2:
                    scores.append(0.25)
                else:
                    scores.append(-1.0)
            except (ValueError, ZeroDivisionError):
                scores.append(-0.5)
        return scores

    @staticmethod
    def _extract_answer(text: str) -> str | None:
        """Extract an answer string from `<answer>` tags or `####` fallback."""
        match = re.search(r"<answer>([\s\S]*?)</answer>", text)
        if match:
            return match.group(1).strip()
        match = re.search(r"####\s*(.+)", text)
        if match:
            return match.group(1).strip()
        return None
