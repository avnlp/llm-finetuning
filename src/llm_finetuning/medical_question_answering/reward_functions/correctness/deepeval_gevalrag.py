"""GEval RAG correctness reward for medical question answering."""

from __future__ import annotations

from typing import ClassVar

from llm_finetuning.core.llm_judges import AbstractDeepEvalGEvalRAGReward
from llm_finetuning.core.reward import RewardConfig


class DeepEvalGEvalRAGReward(AbstractDeepEvalGEvalRAGReward):
    """Reward medical QA correctness via DeepEval GEval with RAG criteria.

    The criteria string reflects the medical domain: it asks the judge to
    evaluate factual accuracy, use of evidence, and completeness specifically
    in the context of medical questions.
    """

    rag_criteria: ClassVar[str] = (
        "Evaluate if the answer correctly addresses the medical question. "
        "Award points for: (1) factual accuracy, (2) use of supporting evidence, "
        "(3) completeness of the answer."
    )

    def __init__(self) -> None:
        """Create the reward with a stable TRL logging name."""
        super().__init__(RewardConfig(name="deepeval_gevalrag"))
