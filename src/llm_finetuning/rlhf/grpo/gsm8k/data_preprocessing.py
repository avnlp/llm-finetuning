"""GSM8K dataset preprocessing utilities for GRPO training."""

from typing import Any


def extract_hash_answer(text: str) -> str | None:
    """Extract answer from GSM8K format with '####' separator."""
    return text.split("####")[1].strip() if "####" in text else None


def format_gsm8k_dataset(dataset: Any, system_prompt: str) -> dict:
    """Format GSM8K dataset with system prompt and chat template."""
    return dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )


def get_tokenized_lengths(dataset: Any, tokenizer: Any) -> list:
    """Get tokenized lengths for dataset prompts."""
    return dataset.map(
        lambda x: {
            "tokens": tokenizer.apply_chat_template(
                x["prompt"], add_generation_prompt=True, tokenize=True
            )
        },
        batched=False,
    )["tokens"]


def get_max_prompt_length(tokenized_lengths: list) -> int:
    """Get maximum prompt length from tokenized lengths."""
    return max(len(tokens) for tokens in tokenized_lengths) + 1
