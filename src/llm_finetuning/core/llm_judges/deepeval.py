"""DeepEval-backed LLM-judge reward classes.

All classes here depend on the ``deepeval`` package.  Importing this module
without ``deepeval`` installed will raise ``ImportError``.  Downstream code
that does not use DeepEval rewards should import from
``llm_finetuning.core.llm_judges.evidently`` or the backend-agnostic
``llm_finetuning.core`` namespace instead, keeping its import graph free of
this dependency.

The class hierarchy is:

    BaseLLMJudgeReward                     (core/llm_judges/base.py)
    └── DeepEvalSimpleReward               (abstract — no reference answer)
        ├── DeepEvalAnswerRelevancyReward  (concrete)
        └── DeepEvalSummarizationReward    (concrete)
    └── AbstractDeepEvalGEvalRAGReward     (abstract — requires reference answer)
        └── (domain subclasses set rag_criteria)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

from deepeval.metrics import AnswerRelevancyMetric, GEval, SummarizationMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from tenacity import retry

from llm_finetuning.core.llm_judges.base import (
    _RETRY_KWARGS,
    BaseLLMJudgeReward,
)
from llm_finetuning.core.reward import RewardConfig


@runtime_checkable
class _DeepEvalMetric(Protocol):
    """Structural interface satisfied by any DeepEval metric with async scoring.

    DeepEval's concrete metric classes (``AnswerRelevancyMetric``,
    ``SummarizationMetric``, ``GEval``, …) all expose ``a_measure`` but share
    no common ABC in the public API.  This Protocol lets us type the shared
    retry wrapper without importing every concrete class or accepting ``Any``.
    """

    async def a_measure(
        self,
        test_case: LLMTestCase,
        *args: Any,
        **kwargs: Any,
    ) -> float | None:
        """Score a single test case asynchronously.

        Args:
            test_case: The populated ``LLMTestCase`` to evaluate.
            *args: Forwarded positional arguments (metric-specific).
            **kwargs: Forwarded keyword arguments (metric-specific).

        Returns:
            A numeric score, or ``None`` when the metric cannot produce one.
        """
        ...


@retry(**_RETRY_KWARGS)
async def _measure_with_retry(
    metric: _DeepEvalMetric,
    test_case: LLMTestCase,
) -> float:
    """Call ``metric.a_measure`` with automatic retry on rate-limit errors.

    The retry policy (exponential back-off, up to 6 attempts) is defined
    centrally in ``base._RETRY_KWARGS`` so that all backends share identical
    back-off behaviour.

    Args:
        metric: Any object satisfying the ``_DeepEvalMetric`` Protocol.
        test_case: The populated ``LLMTestCase`` to evaluate.

    Returns:
        The metric score as a ``float``.  ``None`` results are coerced to
        ``0.0`` rather than propagated, since ``None`` indicates a scoring
        failure rather than a meaningful zero.

    Raises:
        Exception: Re-raises the last exception after all retry attempts are
            exhausted.
    """
    result = await metric.a_measure(test_case)
    return float(result or 0.0)


class DeepEvalSimpleReward(BaseLLMJudgeReward):
    """Abstract base for DeepEval rewards that score without a reference answer.

    Subclasses implement ``_build_metric`` to supply the specific DeepEval
    metric instance.  All async fan-out, semaphore management, and the
    synchronous boundary are handled here so concrete subclasses stay minimal.

    Example:
        Concrete subclasses only need to provide a metric factory::

            class MyMetricReward(DeepEvalSimpleReward):
                def __init__(self) -> None:
                    super().__init__(RewardConfig(name="my_metric"))

                def _build_metric(self) -> MyMetric:
                    return MyMetric(threshold=0.5)
    """

    @abstractmethod
    def _build_metric(self) -> _DeepEvalMetric:
        """Construct and return the DeepEval metric for a single scoring call.

        A fresh metric instance is created per row because DeepEval metrics
        accumulate internal state (scores, reasons) on each ``a_measure``
        call; reusing an instance across rows would corrupt results.

        Returns:
            A configured metric satisfying the ``_DeepEvalMetric`` Protocol.
        """
        raise NotImplementedError

    async def _score_one(
        self,
        semaphore: asyncio.Semaphore,
        prompt: list,
        completion: list,
    ) -> float:
        """Evaluate one generated response for a single prompt.

        Args:
            semaphore: Shared concurrency limiter that bounds simultaneous
                judge calls.
            prompt: Conversation history for one training example.
            completion: Generated assistant messages for ``prompt``.

        Returns:
            A reward score in ``[0.0, 1.0]``.
        """
        ji = self._build_judge_input(prompt, completion)
        test_case = LLMTestCase(input=ji.question, actual_output=ji.response)
        metric = self._build_metric()
        async with semaphore:
            return await _measure_with_retry(metric, test_case)

    async def _score_batch(
        self,
        prompts: list,
        completions: list,
    ) -> list[float]:
        """Evaluate a batch of prompt/response pairs with bounded concurrency.

        Args:
            prompts: Prompt histories from ``GRPOTrainer``.
            completions: Generated assistant responses aligned with
                ``prompts``.

        Returns:
            Scores in input order, one per completion.
        """
        semaphore = self._make_semaphore()
        tasks = [self._score_one(semaphore, p, c) for p, c in zip(prompts, completions)]
        return list(await asyncio.gather(*tasks))

    def __call__(
        self,
        prompts: list,
        completions: list,
        **kwargs: object,
    ) -> list[float]:
        """Return DeepEval scores for a GRPO batch.

        Args:
            prompts: Prompt histories from ``GRPOTrainer``.
            completions: Generated assistant responses to score.
            **kwargs: Extra dataset columns forwarded by ``GRPOTrainer``.
                Unused by this reward.

        Returns:
            Per-completion reward values aligned with ``completions``.
        """
        return self._run_batch(self._score_batch(prompts, completions))


class DeepEvalAnswerRelevancyReward(DeepEvalSimpleReward):
    """Reward answer relevancy via DeepEval's ``AnswerRelevancyMetric``.

    Scores how well the model's response addresses the question, without
    requiring a reference answer.  Uses DeepEval's built-in LLM judge.
    """

    def __init__(self) -> None:
        """Create the reward with a stable TRL logging name."""
        super().__init__(RewardConfig(name="deepeval_answer_relevancy"))

    def _build_metric(self) -> AnswerRelevancyMetric:
        """Construct an ``AnswerRelevancyMetric`` scoped to input and output.

        Returns:
            A fresh ``AnswerRelevancyMetric`` configured to evaluate only
            ``INPUT`` and ``ACTUAL_OUTPUT``, which is the minimal set needed
            for relevancy scoring without a reference.
        """
        return AnswerRelevancyMetric(
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
        )


class DeepEvalSummarizationReward(DeepEvalSimpleReward):
    """Reward summarization quality via DeepEval's ``SummarizationMetric``.

    Evaluates whether the model's response is a faithful and concise summary
    of the input, without requiring an external reference answer.
    """

    def __init__(self) -> None:
        """Create the reward with a stable TRL logging name."""
        super().__init__(RewardConfig(name="deepeval_summarization"))

    def _build_metric(self) -> SummarizationMetric:
        """Construct a ``SummarizationMetric`` scoped to input and output.

        Returns:
            A fresh ``SummarizationMetric`` configured to evaluate
            ``INPUT`` and ``ACTUAL_OUTPUT`` only.
        """
        return SummarizationMetric(
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
        )


class AbstractDeepEvalGEvalRAGReward(BaseLLMJudgeReward, ABC):
    """Abstract base for GEval RAG rewards with domain-specific evaluation criteria.

    Concrete subclasses must declare a ``rag_criteria`` class variable that
    provides the domain-specific GEval prompt.  The enforcement happens at
    class-definition time via ``__init_subclass__``, so a subclass without
    a valid ``rag_criteria`` raises ``TypeError`` immediately — not when the
    first batch is scored.

    Example:
        Domain subclasses are intentionally minimal::

            class DeepEvalGEvalRAGReward(AbstractDeepEvalGEvalRAGReward):
                rag_criteria: ClassVar[str] = (
                    "Evaluate if the answer correctly addresses the medical question. "
                    "Award points for: (1) factual accuracy, "
                    "(2) use of supporting evidence, (3) completeness."
                )

                def __init__(self) -> None:
                    super().__init__(RewardConfig(name="deepeval_gevalrag"))

    Attributes:
        rag_criteria: Domain-specific GEval evaluation prompt.  Must be set
            as a non-empty string class variable on every concrete subclass.
    """

    rag_criteria: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Enforce that every concrete subclass declares a valid ``rag_criteria``.

        This hook runs at class-definition time (when the ``class`` statement
        is executed), not at instantiation time, so misconfigured subclasses
        are caught as early as possible — ideally at import time during
        development.

        Args:
            **kwargs: Forwarded to ``super().__init_subclass__``.

        Raises:
            TypeError: If the subclass is concrete (not itself abstract) and
                either omits ``rag_criteria`` or sets it to a non-string or
                blank string value.
        """
        super().__init_subclass__(**kwargs)

        # Skip the check for abstract intermediaries — only enforce on
        # classes that will actually be instantiated and used for scoring.
        if getattr(cls, "__abstractmethods__", None):
            return

        criteria = getattr(cls, "rag_criteria", None)
        if not isinstance(criteria, str) or not criteria.strip():
            raise TypeError(
                f"{cls.__name__} must define a non-empty string 'rag_criteria' class "
                "variable. Set it to the domain-specific GEval evaluation prompt "
                "before defining the class."
            )

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
            answer: Reference answer used for GEval scoring.

        Returns:
            A GEval score in ``[0.0, 1.0]``.
        """
        ji = self._build_judge_input(prompt, completion, answer)
        test_case = LLMTestCase(
            input=ji.question,
            actual_output=ji.response,
            expected_output=ji.reference,
        )
        # GEval metric is constructed per-row because it embeds the criteria
        # string and accumulates per-call reasoning state internally.
        metric = GEval(
            name="RAG_GEval",
            criteria=self.rag_criteria,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
        )
        async with semaphore:
            return await _measure_with_retry(metric, test_case)

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
            GEval scores in input order.
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
        """Score a GRPO batch with DeepEval GEval using reference answers.

        Args:
            prompts: Prompt histories from ``GRPOTrainer``.
            completions: Generated assistant responses to evaluate.
            **kwargs: Extra dataset columns. Must include ``"answer"`` with
                a ``list[str]`` of reference answers.

        Returns:
            Per-completion GEval scores aligned with ``completions``.

        Raises:
            KeyError: If ``kwargs`` does not contain ``"answer"``.
        """
        answers = cast(list[str], kwargs["answer"])
        return self._run_batch(self._score_batch(prompts, completions, answers))
