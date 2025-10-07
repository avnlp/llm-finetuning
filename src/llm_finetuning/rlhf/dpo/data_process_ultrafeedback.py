"""Data processing utilities for Direct Preference Optimization (DPO) with UltraFeedback dataset.

This module provides functions to load, process, and prepare datasets for DPO training,
specifically tailored for the UltraFeedback dataset format. It handles dataset mixing,
chat template application, and proper formatting for preference learning tasks.
"""

import os
import re
from typing import Any, Optional, Union

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import PreTrainedTokenizer


def get_datasets(
    data_config: Union[dict[str, float], Any],
    splits: Optional[list[str]] = None,
    shuffle: bool = True,
) -> DatasetDict:
    """Load and optionally mix multiple datasets with specified proportions.

    This function serves as the main entry point for loading datasets for DPO training.
    It supports both local and remote (Hugging Face Hub) datasets and can handle
    weighted mixing of multiple datasets.

    Example:
        ```python
        # Example usage with dataset mixing
        data_config = {
            "dataset1": 0.5,  # 50% of training data
            "dataset2": 0.3,  # 30% of training data
            "dataset3": 0.2,  # 20% of training data
        }
        datasets = get_datasets(data_config, splits=["train", "test"])
        ```

    Args:
        data_config: Either a dictionary mapping dataset names to their proportions
            or a DataArguments object containing dataset configuration.
        splits: List of dataset splits to load (e.g., ['train', 'test', 'validation']).
            Defaults to ['train', 'test'].
        shuffle: Whether to shuffle the datasets after loading and concatenation.

    Returns:
        DatasetDict: A dictionary containing the loaded and mixed datasets, keyed by split.

    Raises:
        ValueError: If the data configuration format is invalid or datasets cannot be loaded.
    """
    if splits is None:
        splits = ["train", "test"]
    if type(data_config) is dict:
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset2": 0.3,
        #             "dataset3": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        msg = f"Data config {data_config} not recognized."
        raise ValueError(msg)

    return mix_datasets(dataset_mixer, splits=splits, shuffle=shuffle)


def mix_datasets(
    dataset_mixer: dict[str, float],
    splits: Optional[list[str]] = None,
    shuffle: bool = True,
) -> DatasetDict:
    """Load and mix multiple datasets based on specified proportions.

    This function handles the core logic of loading datasets (either from local disk or
    Hugging Face Hub) and combining them according to the specified proportions.
    Training sets are subsampled based on the proportions, while test/validation sets
    are kept at full size for fair evaluation.

    Args:
        dataset_mixer: Dictionary where keys are dataset identifiers (either paths or
            Hugging Face dataset names) and values are the proportions to use.
        splits: List of dataset splits to load (e.g., ['train', 'test']).
            Defaults to ['train', 'test'] if None.
        shuffle: Whether to shuffle the datasets after loading and concatenation.

    Returns:
        DatasetDict: A dictionary containing the mixed datasets, keyed by split.

    Raises:
        ValueError: If no valid datasets are found or if any proportion is negative.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for ds, frac in dataset_mixer.items():
        fracs.append(frac)
        for split in splits:
            try:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, split=split)
            except DatasetGenerationError:
                # If not, check local dataset
                dataset = load_from_disk(os.path.join(ds, split))

            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                msg = f"Split type {split} not recognized as one of test or train."
                raise ValueError(msg)

    if any(frac < 0 for frac in fracs):
        msg = "Dataset fractions cannot be negative."
        raise ValueError(msg)

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)

    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(
                seed=42
            )
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        msg = f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
        raise ValueError(msg)

    return raw_datasets


def _strip_prefix(s: str, pattern: str) -> str:
    r"""Remove the specified prefix from a string if present.

    This is a helper function used to clean up model outputs by removing
    assistant prefixes that might be added during template processing.

    Args:
        s: The input string to process.
        pattern: The prefix pattern to remove from the start of the string.

    Returns:
        str: The input string with the prefix removed if it was present.
    """
    return re.sub(f"^{re.escape(pattern)}", "", s)


def apply_chat_template_example(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    assistant_prefix: str = "<|assistant|>\n",
) -> dict[str, str]:
    r"""Apply chat template formatting to a preference learning example.

    This function processes a single example from a preference dataset, formatting the
    chosen and rejected responses according to the tokenizer's chat template. It handles
    the special case of UltraFeedback-style examples that include chosen and rejected
    responses for the same prompt.

    The input example should contain 'chosen' and 'rejected' keys, each containing a
    list of message dictionaries with 'role' and 'content' fields.

    Example Input:
        ```python
        {
            # Prompt
            "prompt": "how can i develop a habit of drawing daily",
            # Chosen (preferred) response
            "chosen": [
                {
                    "content": "how can i develop a habit of drawing daily",
                    "role": "user",
                },
                {"content": "Developing a daily drawing habit...", "role": "assistant"},
            ],
            # Rejected response
            "rejected": [
                {
                    "content": "how can i develop a habit of drawing daily",
                    "role": "user",
                },
                {
                    "content": "As an AI, I can't provide personal advice...",
                    "role": "assistant",
                },
            ],
        }
        ```

    Args:
        example: A dictionary containing 'chosen' and 'rejected' conversation examples.
        tokenizer: A Hugging Face tokenizer with a chat template defined.
        assistant_prefix: The prefix to strip from the beginning of assistant responses.
            Defaults to "<|assistant|>\n".

    Returns:
        Dict[str, str]: The processed example with 'text_chosen', 'text_rejected',
            and 'text_prompt' fields added.

    Raises:
        ValueError: If the example doesn't contain the expected 'chosen' and 'rejected' keys,
            or if the conversation format is invalid.
    """
    if all(k in example for k in ("chosen", "rejected")):
        # We filter out the prompt, so the text is everything after the last assistant token
        prompt_messages = [
            next(msg for msg in example["chosen"] if msg["role"] == "user")
        ]

        # Insert empty system message if not present
        if example["chosen"][0]["role"] != "system":
            prompt_messages.insert(0, {"role": "system", "content": ""})
        else:
            # Get system message from Chosen messages
            prompt_messages.insert(0, example["chosen"][0])

        chosen_messages = example["chosen"][1:]
        rejected_messages = example["rejected"][1:]

        example["text_chosen"] = tokenizer.apply_chat_template(
            chosen_messages, tokenize=False
        )
        example["text_rejected"] = tokenizer.apply_chat_template(
            rejected_messages, tokenize=False
        )
        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
        example["text_rejected"] = _strip_prefix(
            example["text_rejected"], assistant_prefix
        )

    else:
        msg = f"Could not format example as dialogue for preference training! Require `[chosen, rejected]` keys but found {list(example.keys())}"
        raise ValueError(msg)
    return example
