"""Dataset loading and formatting for DPO training on WebGPT comparisons."""

from __future__ import annotations

import re
from typing import Any

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from llm_finetuning.core import DatasetConfig
from llm_finetuning.preference_alignment.base_loader import PreferenceDatasetLoader


class WebGPTDPOLoader(PreferenceDatasetLoader):
    """Load and format WebGPT comparisons for DPO training."""

    CONFIG = DatasetConfig(dataset_id="openai/webgpt_comparisons")

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, config: DatasetConfig | None = None
    ) -> None:
        """Create a WebGPT DPO loader bound to a tokenizer."""
        super().__init__(config or self.CONFIG, tokenizer)

    def load(self, split: str = "train") -> Dataset:
        """Load and filter the split, then map to DPO `prompt/chosen/rejected`."""
        ds = load_dataset(self.config.dataset_id, split=split)
        ds = ds.filter(lambda x: x["preference"] != 0)
        return self._format_dataset(ds)

    def format_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a WebGPT row to DPO columns using `apply_chat_template`."""
        assistant_prefix = "<|assistant|>\n"
        question = example["question"]["full_text"]
        if example["preference"] > 0:
            chosen_text = example["answer_0"]["text"]
            rejected_text = example["answer_1"]["text"]
        else:
            chosen_text = example["answer_1"]["text"]
            rejected_text = example["answer_0"]["text"]

        prompt_msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": question},
        ]
        chosen_msgs = [{"role": "assistant", "content": chosen_text}]
        rejected_msgs = [{"role": "assistant", "content": rejected_text}]

        text_chosen = re.sub(
            f"^{re.escape(assistant_prefix)}",
            "",
            self.tokenizer.apply_chat_template(chosen_msgs, tokenize=False),
        )
        text_rejected = re.sub(
            f"^{re.escape(assistant_prefix)}",
            "",
            self.tokenizer.apply_chat_template(rejected_msgs, tokenize=False),
        )
        return {
            "prompt": self.tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            ),
            "chosen": text_chosen,
            "rejected": text_rejected,
        }
