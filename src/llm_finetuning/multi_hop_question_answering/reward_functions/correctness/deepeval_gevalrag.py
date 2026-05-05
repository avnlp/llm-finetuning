"""GEval RAG correctness reward for multi-hop question answering."""

from __future__ import annotations

from typing import ClassVar

from llm_finetuning.core.llm_judges import AbstractDeepEvalGEvalRAGReward
from llm_finetuning.core.reward import RewardConfig


class DeepEvalGEvalRAGReward(AbstractDeepEvalGEvalRAGReward):
    """Reward multi-hop QA correctness via DeepEval GEval with RAG criteria.

    The criteria string reflects the multi-hop domain: it emphasises the
    need to correctly chain evidence across multiple reasoning steps.
    """

    rag_criteria: ClassVar[str] = (
        "Evaluate if the answer correctly addresses the multi-hop question. "
        "Award points for: (1) factual accuracy, (2) use of supporting evidence, "
        "(3) completeness of the answer."
    )

    def __init__(self) -> None:
        """Create the reward with a stable TRL logging name."""
        super().__init__(RewardConfig(name="deepeval_gevalrag"))
