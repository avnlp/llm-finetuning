from typing import Any, Dict, Optional

from datasets import load_dataset


class Dataloader:
    def __init__(
        self,
        dataset_name: str,
        split: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        system_message_inputs: Optional[dict[str, Any]] = None,
        user_message_inputs: Optional[dict[str, Any]] = None,
    ):
        """Initializes the Dataloader class with dataset and prompts.

        Args:
            dataset_name (str): Name of the dataset to load.
            split (str): Dataset split (e.g., 'train', 'test').
            system_prompt (str): Template for the system message.
            user_prompt (str): Template for the user message.
            system_message_inputs (Dict[str, Any]): Input values for system prompt formatting.
            user_message_inputs (Dict[str, Any]): Input values for user prompt formatting.
        """
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = load_dataset(dataset_name, split=split)
        self.system_prompt = system_prompt or "Default system prompt."
        self.user_prompt = user_prompt or "Default user prompt: {question}."
        self.system_message_inputs = system_message_inputs or {}
        self.user_message_inputs = user_message_inputs or {}

    def _create_conversation(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Creates a conversation structure based on the sample and initialized prompts.

        Args:
            sample (Dict[str, Any]): A single sample from the dataset.

        Returns:
            Dict[str, Any]: A dictionary representing the conversation.
        """
        # Ensure system and user prompts are correctly formatted
        try:
            system_message = self.system_prompt.format(**self.system_message_inputs)
        except KeyError as e:
            msg = f"Missing key in system_message_inputs: {e}"
            raise ValueError(msg)

        try:
            user_message = self.user_prompt.format(**{**self.user_message_inputs, **sample})
        except KeyError as e:
            msg = f"Missing key in user_message_inputs or sample: {e}"
            raise ValueError(msg)

        return {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": sample.get("answer", "")},
            ]
        }
