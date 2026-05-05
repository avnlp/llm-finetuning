"""Evidently-backed LLM-judge reward class.

All classes here depend on the ``evidently`` package.  Importing this module
without ``evidently`` installed will raise ``ImportError``.  Downstream code
that does not use Evidently rewards should import from
``llm_finetuning.core.llm_judges.deepeval`` or the backend-agnostic
``llm_finetuning.core`` namespace instead, keeping its import graph clean.
"""

from __future__ import annotations

import asyncio
from typing import cast

from evidently.descriptors import LLMEval  # type: ignore[import-untyped]
from evidently.features.llm_judge import (  # type: ignore[import-untyped]
    BinaryClassificationPromptTemplate,
)
from tenacity import retry

from llm_finetuning.core.llm_judges.base import _RETRY_KWARGS, BaseLLMJudgeReward
from llm_finetuning.core.reward import RewardConfig


# This template is module-level because it is identical for all domains —
# there is no domain-specific wording needed for binary correctness.
_CORRECTNESS_TEMPLATE = BinaryClassificationPromptTemplate(
    criteria=(
        "Is the actual output a correct and complete answer to the question, "
        "consistent with the expected output?"
    ),
    target_category="correct",
    non_target_category="incorrect",
)


@retry(**_RETRY_KWARGS)
async def _evaluate_with_retry(evaluator: LLMEval) -> dict:  # type: ignore[type-arg]
    """Call the Evidently evaluator with automatic retry on rate-limit errors.

    Evidently's ``LLMEval.generate_features`` is synchronous, so it is
    dispatched to a thread pool via ``asyncio.to_thread`` to avoid blocking
    the event loop during the LLM call.

    The retry policy (exponential back-off, up to 6 attempts) is shared with
    the DeepEval wrapper via ``_RETRY_KWARGS`` in ``base``.

    Args:
        evaluator: A configured ``LLMEval`` instance ready to score one row.

    Returns:
        A dict containing at minimum a ``"category"`` key whose value is
        ``"correct"`` or ``"incorrect"``.

    Raises:
        Exception: Re-raises the last exception after all retry attempts are
            exhausted.
    """
    return await asyncio.to_thread(evaluator.generate_features, None)


class EvidentlyCorrectnessLLMReward(BaseLLMJudgeReward):
    """Reward factual correctness via Evidently's LLM binary classification judge.

    Uses ``BinaryClassificationPromptTemplate`` to ask the judge whether the
    model's response is correct and complete relative to the reference answer.
    Returns ``1.0`` for ``"correct"`` and ``0.0`` for ``"incorrect"``.

    Unlike DeepEval metrics this reward produces a hard binary score rather
    than a continuous value, which can produce higher-variance GRPO advantages
    when the model is already near-correct.  Consider pairing with a softer
    metric for fine-grained signal.
    """

    def __init__(self) -> None:
        """Create the reward with a stable TRL logging name."""
        super().__init__(RewardConfig(name="evidently_correctness_llm"))

    async def _score_one(
        self,
        semaphore: asyncio.Semaphore,
        prompt: list,
        completion: list,
        answer: str,
    ) -> float:
        """Evaluate one generated response against its reference answer.

        Args:
            semaphore: Shared concurrency limiter acquired before the judge
                call.
            prompt: Conversation history for one training example.
            completion: Generated assistant messages for ``prompt``.
            answer: Reference answer for the binary correctness judge.

        Returns:
            ``1.0`` if the judge labels the response ``"correct"``;
            ``0.0`` otherwise.
        """
        ji = self._build_judge_input(prompt, completion, answer)
        evaluator = LLMEval(
            template=_CORRECTNESS_TEMPLATE,
            input=ji.question,
            output=ji.response,
            reference=ji.reference,
        )
        async with semaphore:
            result = await _evaluate_with_retry(evaluator)
        return 1.0 if result.get("category") == "correct" else 0.0

    async def _score_batch(
        self,
        prompts: list,
        completions: list,
        answers: list[str],
    ) -> list[float]:
        """Evaluate a batch of responses against reference answers.

        Args:
            prompts: Prompt histories from ``GRPOTrainer``.
            completions: Generated assistant responses aligned with
                ``prompts``.
            answers: Reference answers aligned one-to-one with the batch.

        Returns:
            Binary scores (``1.0`` or ``0.0``) in input order.
        """
        semaphore = self._make_semaphore()
        tasks = [
            self._score_one(semaphore, p, c, a)
            for p, c, a in zip(prompts, completions, answers)
        ]
        return list(await asyncio.gather(*tasks))

    def __call__(
        self,
        prompts: list,
        completions: list,
        **kwargs: object,
    ) -> list[float]:
        """Score a GRPO batch with Evidently binary correctness using reference answers.

        Args:
            prompts: Prompt histories from ``GRPOTrainer``.
            completions: Generated assistant responses to evaluate.
            **kwargs: Extra dataset columns. Must include ``"answer"`` with
                a ``list[str]`` of reference answers.

        Returns:
            Per-completion scores aligned with ``completions``, each either
            ``1.0`` (correct) or ``0.0`` (incorrect).

        Raises:
            KeyError: If ``kwargs`` does not contain ``"answer"``.
        """
        answers = cast(list[str], kwargs["answer"])
        return self._run_batch(self._score_batch(prompts, completions, answers))
