"""Base classes shared by all async LLM-judge reward implementations.

This module defines the ``JudgeInput`` value object and ``BaseLLMJudgeReward``,
which centralise the one piece of domain knowledge that every reward shares:
how to extract a question and a response (and optionally a reference answer)
from the TRL message-list format that GRPOTrainer passes to reward functions.

Keeping this extraction in one class means a future change to the message
schema — e.g. multi-turn prompts or a different content key — only requires
a single edit here rather than a change in every scoring method.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from tenacity import (
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from llm_finetuning.core.reward import BaseReward, RewardConfig


# OpenAI-compatible clients surface rate limits as HTTP 429. Some libraries
# (DeepEval, Evidently) expose the status code on the exception; others only
# embed it in the message string. We check both to maximise coverage.


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like an LLM provider rate-limit error.

    Args:
        exc: Any exception raised during an LLM API call.

    Returns:
        ``True`` when the exception carries HTTP status 429 or its message
        contains a well-known rate-limit phrase; ``False`` otherwise.
    """
    if getattr(exc, "status_code", None) == 429:
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "too many requests" in msg


# Shared retry policy applied by both DeepEval and Evidently wrappers.
# Exponential back-off with jitter avoids thundering-herd on burst failures;
# six attempts give roughly two minutes of total retry budget at max wait=30 s.
_RETRY_KWARGS = {
    "retry": retry_if_exception(_is_rate_limit_error),
    "wait": wait_random_exponential(multiplier=1, min=1, max=30),
    "stop": stop_after_attempt(6),
    "reraise": True,
}


@dataclass(frozen=True, slots=True)
class JudgeInput:
    """Structured inputs extracted from TRL's message-list format.

    GRPOTrainer passes prompts as ``list[list[dict]]`` where each inner list
    is a conversation and each dict has ``"role"`` and ``"content"`` keys.
    Completions follow the same shape but always contain a single assistant
    message.  This dataclass holds the already-extracted strings so that
    scoring methods work with plain values rather than indexing into nested
    structures.

    Attributes:
        question: The text of the last user turn in the prompt.
        response: The text of the assistant completion.
        reference: An optional ground-truth answer used by reference-based
            metrics.  ``None`` when the reward does not require one.
    """

    question: str
    response: str
    reference: str | None = None


class BaseLLMJudgeReward(BaseReward):
    """Abstract base for async LLM-judge rewards.

    Extends ``BaseReward`` with:

    * A ``DEFAULT_MAX_CONCURRENCY`` constant that caps simultaneous LLM calls
      to avoid saturating provider rate limits.
    * ``_build_judge_input`` — a single method that encodes the TRL
      message-list convention so subclasses never index into raw lists.

    Subclasses still own their own ``_score_one``, ``_score_batch``, and
    ``__call__`` implementations because the signatures differ (some require
    a reference answer, others do not).
    """

    DEFAULT_MAX_CONCURRENCY: int = 8

    def __init__(self, config: RewardConfig | None = None) -> None:
        """Initialise the reward and forward configuration to ``BaseReward``.

        Args:
            config: Optional ``RewardConfig`` naming this reward for TRL
                logging.  Defaults to the ``BaseReward`` default when omitted.
        """
        super().__init__(config)

    def _build_judge_input(
        self,
        prompt: list,
        completion: list,
        answer: str | None = None,
    ) -> JudgeInput:
        """Extract scoring inputs from TRL's message-list structures.

        GRPOTrainer passes prompts as a list of message dicts; the question
        is the last message (the most recent user turn).  Completions are
        always a single-element list containing the assistant reply.

        Args:
            prompt: A prompt message list as provided by GRPOTrainer.  Each
                element is a dict with ``"role"`` and ``"content"`` keys.
                The question is taken from ``prompt[-1]["content"]``.
            completion: A completion message list as provided by GRPOTrainer.
                Always contains exactly one dict; the response text is at
                ``completion[0]["content"]``.
            answer: An optional ground-truth string for reference-based
                metrics.  Pass ``None`` (default) for metrics that do not
                require a reference.

        Returns:
            A ``JudgeInput`` populated with the extracted strings.

        Raises:
            IndexError: If ``prompt`` is empty or ``completion`` is empty.
        """
        return JudgeInput(
            question=prompt[-1]["content"],
            response=completion[0]["content"],
            reference=answer,
        )

    def _make_semaphore(self) -> asyncio.Semaphore:
        """Create a semaphore capped at ``DEFAULT_MAX_CONCURRENCY``.

        Returns:
            An ``asyncio.Semaphore`` that limits the number of concurrent LLM
            calls.  The semaphore must be created inside the running event loop
            (i.e. inside a coroutine), not at class construction time.
        """
        return asyncio.Semaphore(self.DEFAULT_MAX_CONCURRENCY)
