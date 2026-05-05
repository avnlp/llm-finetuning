"""Dataset loading for PPO training on WebGPT comparisons."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from llm_finetuning.core import BaseDatasetLoader, DatasetConfig


class WebGPTPPOLoader(BaseDatasetLoader):
    """Load and format WebGPT comparisons for PPO training."""

    CONFIG = DatasetConfig(dataset_id="openai/webgpt_comparisons")

    def __init__(self, config: DatasetConfig | None = None) -> None:
        """Create a WebGPT PPO loader with the default dataset config."""
        super().__init__(config or self.CONFIG)

    def load(self, split: str = "train") -> Dataset:
        """Load and filter the split, then map to PPO `prompt` rows."""
        ds = load_dataset(self.config.dataset_id, split=split)
        ds = ds.filter(lambda x: x["preference"] != 0)
        return self._format_dataset(ds)

    def format_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a WebGPT row to a single PPO `prompt` string."""
        return {"prompt": example["question"]["full_text"]}
