"""Dataset loading and formatting for multi-hop question answering GRPO training."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from llm_finetuning.core import BaseDatasetLoader, DatasetConfig


class MultiHopLoader(BaseDatasetLoader):
    """Base loader for multi-hop GRPO datasets with shared formatting."""

    SYSTEM_PROMPT = (
        "You are an expert at multi-hop reasoning. "
        "Think step by step inside <reasoning> tags. "
        "Provide your final answer inside <answer> tags."
    )

    def format_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a multi-hop QA row to `{'prompt': ..., 'answer': ...}` for GRPO."""
        return {
            "prompt": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["answer"],
        }


class HotpotQALoader(MultiHopLoader):
    """Load and format HotpotQA for GRPO training."""

    CONFIG = DatasetConfig(dataset_id="hotpotqa/hotpot_qa", subset="distractor")

    def __init__(self, config: DatasetConfig | None = None) -> None:
        """Create a HotpotQA loader with the default dataset config."""
        super().__init__(config or self.CONFIG)


class FreshQALoader(MultiHopLoader):
    """Load and format FreshQA (SealQA longseal) for GRPO training."""

    CONFIG = DatasetConfig(dataset_id="vtllms/sealqa", subset="longseal")

    def __init__(self, config: DatasetConfig | None = None) -> None:
        """Create a FreshQA loader with the default dataset config."""
        super().__init__(config or self.CONFIG)


class MuSiQueLoader(MultiHopLoader):
    """Load and format MuSiQue for GRPO training (answerable-only)."""

    CONFIG = DatasetConfig(dataset_id="dgslibisey/MuSiQue")

    def __init__(self, config: DatasetConfig | None = None) -> None:
        """Create a MuSiQue loader with the default dataset config."""
        super().__init__(config or self.CONFIG)

    def load(self, split: str = "train") -> Dataset:
        """Load and filter MuSiQue to answerable examples, then map formatting."""
        ds = load_dataset(
            self.config.dataset_id,
            self.config.subset,
            split=split,
            cache_dir=self.config.cache_dir,
            **self.config.extra_load_kwargs,
        )
        ds = ds.filter(lambda x: x["answerable"])
        return self._format_dataset(ds)
