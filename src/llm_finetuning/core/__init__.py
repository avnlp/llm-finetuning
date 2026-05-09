"""Core abstractions shared across pipelines.

This namespace is intentionally backend-agnostic.  Evaluation-specific reward
classes (DeepEval, Evidently) live in ``llm_finetuning.core.llm_judges`` so
that importing ``BaseReward`` or ``BaseDatasetLoader`` does not transitively
pull in heavyweight evaluation dependencies.
"""

from llm_finetuning.core.dataset_loader import BaseDatasetLoader, DatasetConfig
from llm_finetuning.core.prompt_template import PromptTemplate
from llm_finetuning.core.reward import BaseReward, RewardConfig


__all__ = [
    "BaseDatasetLoader",
    "DatasetConfig",
    "PromptTemplate",
    "BaseReward",
    "RewardConfig",
]
