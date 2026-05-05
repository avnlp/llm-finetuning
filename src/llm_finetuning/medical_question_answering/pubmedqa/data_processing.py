"""Dataset loading and formatting for PubMedQA GRPO training."""

from __future__ import annotations

from typing import Any

from llm_finetuning.core import BaseDatasetLoader, DatasetConfig


class PubMedQALoader(BaseDatasetLoader):
    """Load and format PubMedQA examples for GRPO training."""

    CONFIG = DatasetConfig(dataset_id="qiaojin/PubMedQA", subset="pqa_artificial")
    SYSTEM_PROMPT = (
        "You are a biomedical research expert. Reason through the research question step by step "
        "inside <reasoning> tags. Provide your final answer inside <answer> tags."
    )

    def __init__(self, config: DatasetConfig | None = None) -> None:
        """Create a PubMedQA loader with the default dataset config."""
        super().__init__(config or self.CONFIG)

    def format_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a PubMedQA row to `{'prompt': ..., 'answer': ...}` for GRPO."""
        return {
            "prompt": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["long_answer"],
        }
