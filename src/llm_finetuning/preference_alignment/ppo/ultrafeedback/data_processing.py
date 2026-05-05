"""Dataset loading for PPO training on UltraFeedback."""

from __future__ import annotations

from typing import Any

from llm_finetuning.core import BaseDatasetLoader, DatasetConfig


class UltraFeedbackPPOLoader(BaseDatasetLoader):
    """Load and format UltraFeedback binarized for PPO training."""

    CONFIG = DatasetConfig(dataset_id="HuggingFaceH4/ultrafeedback_binarized")

    def __init__(self, config: DatasetConfig | None = None) -> None:
        """Create an UltraFeedback PPO loader with the default dataset config."""
        super().__init__(config or self.CONFIG)

    def format_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map an UltraFeedback row to a single PPO `prompt` string."""
        user_msgs = [msg for msg in example["chosen"] if msg["role"] == "user"]
        return {"prompt": user_msgs[0]["content"] if user_msgs else ""}
