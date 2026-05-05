"""Dataset loading and formatting for MedQA GRPO training."""

from __future__ import annotations

from typing import Any

from llm_finetuning.core import BaseDatasetLoader, DatasetConfig


class MedQALoader(BaseDatasetLoader):
    """Load and format MedQA examples for GRPO training."""

    CONFIG = DatasetConfig(
        dataset_id="bigbio/med_qa", subset="med_qa_en_4options_bigbio_qa"
    )
    SYSTEM_PROMPT = (
        "You are a medical expert. Reason through the clinical question step by step "
        "inside <reasoning> tags. Provide your final answer inside <answer> tags."
    )

    def __init__(self, config: DatasetConfig | None = None) -> None:
        """Create a MedQA loader with the default dataset config."""
        super().__init__(config or self.CONFIG)

    def format_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a MedQA row to `{'prompt': ..., 'answer': ...}` for GRPO."""
        choices = "\n".join(
            f"{label}. {text}"
            for label, text in zip(
                example["choices"]["key"], example["choices"]["value"]
            )
        )
        question = f"{example['question']}\n\nChoices:\n{choices}"
        return {
            "prompt": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "answer": example["answer"],
        }
