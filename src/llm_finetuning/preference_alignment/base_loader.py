"""Base class for preference alignment dataset loaders."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from llm_finetuning.core import BaseDatasetLoader, DatasetConfig


class PreferenceDatasetLoader(BaseDatasetLoader):
    """Base loader for preference alignment datasets (DPO, ORPO, PPO, KTO)."""

    def __init__(
        self, config: DatasetConfig, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        """Create a loader bound to a dataset config and tokenizer."""
        super().__init__(config)
        self.tokenizer = tokenizer
