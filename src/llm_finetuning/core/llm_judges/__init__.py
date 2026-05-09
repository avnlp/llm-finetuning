"""Public surface of the ``llm_judges`` package.

Import from here when you need the concrete reward classes.  The backend
modules (``deepeval``, ``evidently``) are intentionally not imported from
``llm_finetuning.core.__init__`` to avoid pulling heavyweight evaluation
dependencies into unrelated code paths (dataset loaders, training scripts,
prompt templates, etc.).

Typical usage in a domain reward file::

    from llm_finetuning.core.llm_judges import DeepEvalAnswerRelevancyReward
"""

from __future__ import annotations

from llm_finetuning.core.llm_judges.base import BaseLLMJudgeReward, JudgeInput
from llm_finetuning.core.llm_judges.deepeval import (
    AbstractDeepEvalGEvalRAGReward,
    DeepEvalAnswerRelevancyReward,
    DeepEvalSimpleReward,
    DeepEvalSummarizationReward,
)
from llm_finetuning.core.llm_judges.evidently import EvidentlyCorrectnessLLMReward


__all__ = [
    "JudgeInput",
    "BaseLLMJudgeReward",
    "DeepEvalSimpleReward",
    "DeepEvalAnswerRelevancyReward",
    "DeepEvalSummarizationReward",
    "AbstractDeepEvalGEvalRAGReward",
    "EvidentlyCorrectnessLLMReward",
]
