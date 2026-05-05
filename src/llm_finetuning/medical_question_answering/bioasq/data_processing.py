"""Dataset loading and formatting for BioASQ GRPO training."""

from __future__ import annotations

from typing import Any

from llm_finetuning.core import BaseDatasetLoader, DatasetConfig


class BioASQLoader(BaseDatasetLoader):
    """Load and format BioASQ examples for GRPO training."""

    CONFIG = DatasetConfig(
        dataset_id="enelpol/rag-mini-bioasq", subset="question-answer-passages"
    )
    SYSTEM_PROMPT = (
        "You are a biomedical expert. Reason through the question step by step "
        "inside <reasoning> tags. Provide your final answer inside <answer> tags."
    )

    def __init__(self, config: DatasetConfig | None = None) -> None:
        """Create a BioASQ loader with the default dataset config."""
        super().__init__(config or self.CONFIG)

    def format_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a BioASQ row to `{'prompt': ..., 'answer': ...}` for GRPO."""
        return {
            "prompt": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["answer"],
        }
