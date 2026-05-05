"""Dataset loading for KTO training on kto-mix-14k."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from llm_finetuning.core import BaseDatasetLoader, DatasetConfig


class KTOMixLoader(BaseDatasetLoader):
    """Load the kto-mix-14k dataset for KTO training."""

    CONFIG = DatasetConfig(dataset_id="trl-lib/kto-mix-14k")

    def __init__(self, config: DatasetConfig | None = None) -> None:
        """Create a KTO Mix loader with the default dataset config."""
        super().__init__(config or self.CONFIG)

    def load(self, split: str = "train") -> Dataset:
        """Load the dataset split without mapping (already in KTO schema)."""
        # Dataset is already in KTO format (prompt/completion/label); no mapping needed.
        return load_dataset(self.config.dataset_id, split=split)

    def format_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Return the example unchanged (unused; mapping is skipped)."""
        return example
