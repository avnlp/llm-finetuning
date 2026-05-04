"""Dataloader for preparing datasets with prompts for SFT training."""

from typing import Any, Optional

from datasets import load_dataset


class Dataloader:
    """Loads and formats datasets for SFT training with system and user prompts."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        system_message_inputs: Optional[dict[str, Any]] = None,
        user_message_inputs: Optional[dict[str, Any]] = None,
    ):
        """Initialize the Dataloader with dataset and prompts.

        Args:
            dataset_name: Name of the dataset to load.
            split: Dataset split (e.g., 'train', 'test').
            system_prompt: Template for the system message.
            user_prompt: Template for the user message.
            system_message_inputs: Input values for system prompt formatting.
            user_message_inputs: Input values for user prompt formatting.
        """
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = load_dataset(dataset_name, split=split)
        self.system_prompt = system_prompt or "Default system prompt."
        self.user_prompt = user_prompt or "Default user prompt: {question}."
        self.system_message_inputs = system_message_inputs or {}
        self.user_message_inputs = user_message_inputs or {}

    def _create_conversation(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Create a conversation structure from a sample.

        Args:
            sample: A single sample from the dataset.

        Returns:
            A dictionary representing the conversation.
        """
        try:
            system_message = self.system_prompt.format(**self.system_message_inputs)
        except KeyError as e:
            msg = f"Missing key in system_message_inputs: {e}"
            raise ValueError(msg) from e

        try:
            user_message = self.user_prompt.format(
                **{**self.user_message_inputs, **sample}
            )
        except KeyError as e:
            msg = f"Missing key in user_message_inputs or sample: {e}"
            raise ValueError(msg) from e

        return {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": sample.get("answer", "")},
            ]
        }
