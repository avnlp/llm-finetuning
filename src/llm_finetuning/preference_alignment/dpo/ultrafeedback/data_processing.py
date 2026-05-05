"""Dataset loading and formatting for DPO training on UltraFeedback."""

from __future__ import annotations

import re
from typing import Any

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from llm_finetuning.core import DatasetConfig
from llm_finetuning.preference_alignment.base_loader import PreferenceDatasetLoader


class UltraFeedbackDPOLoader(PreferenceDatasetLoader):
    """Load and format UltraFeedback binarized for DPO training."""

    CONFIG = DatasetConfig(dataset_id="HuggingFaceH4/ultrafeedback_binarized")

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        fraction: float = 1.0,
        config: DatasetConfig | None = None,
    ) -> None:
        """Create an UltraFeedback DPO loader with optional subsampling."""
        super().__init__(config or self.CONFIG, tokenizer)
        self.fraction = fraction

    def load(self, split: str = "train_prefs") -> Dataset:
        """Load a fraction of the split and map to DPO `prompt/chosen/rejected`."""
        split_str = f"{split}[:{int(self.fraction * 100)}%]"
        ds = load_dataset(self.config.dataset_id, split=split_str)
        return self._format_dataset(ds)

    def format_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map an UltraFeedback row to DPO columns using `apply_chat_template`."""
        assistant_prefix = "<|assistant|>\n"
        prompt_msgs = [{"role": "system", "content": ""}] + [
            msg for msg in example["chosen"] if msg["role"] == "user"
        ][:1]
        chosen_msgs = example["chosen"][1:]
        rejected_msgs = example["rejected"][1:]
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
