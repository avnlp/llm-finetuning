"""Dataset loading and formatting for OpenR1-Math SFT format-priming."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from llm_finetuning.core import BaseDatasetLoader, DatasetConfig
from llm_finetuning.math_reasoning.data_processing import MATH_REASONING_SYSTEM_PROMPT


class OpenR1MathSFTLoader(BaseDatasetLoader):
    """Load and format OpenR1-Math-220k for SFT format-priming.

    Filters to numeric-answer examples, converts ``<think>`` traces to
    ``<reasoning>/<answer>`` format, and applies a token-length budget.
    Output schema: ``{"text": str}`` for ``SFTTrainer``.
    """

    CONFIG = DatasetConfig(
        dataset_id="open-r1/OpenR1-Math-220k",
        subset="default",
        remove_source_columns=True,
    )

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: DatasetConfig | None = None,
        max_example_tokens: int = 1024,
        max_samples: int | None = None,
        require_reasoning_complete: bool = True,
    ) -> None:
        """Create an OpenR1-Math loader.

        Args:
            tokenizer: Tokenizer used for chat-template formatting and length
                filtering.
            config: Optional dataset config override; defaults to
                ``OpenR1MathSFTLoader.CONFIG``.
            max_example_tokens: Maximum number of tokens allowed per formatted
                example (length budget).
            max_samples: Optional cap on the number of examples returned after
                all filtering. ``None`` means no cap.
            require_reasoning_complete: When ``True``, drop examples where no
                reasoning trace is marked complete.
        """
        super().__init__(config or self.CONFIG)
        self.tokenizer = tokenizer
        self.max_example_tokens = max_example_tokens
        self.max_samples = max_samples
        self.require_reasoning_complete = require_reasoning_complete

    def load(self, split: str) -> Dataset:
        """Load, filter, format, and length-budget the dataset.

        Args:
            split: HuggingFace dataset split name, e.g. ``"train"``.

        Returns:
            A ``Dataset`` with a single ``"text"`` column ready for
            ``SFTTrainer``.
        """
        raw = load_dataset(
            self.config.dataset_id,
            self.config.subset,
            split=split,
            cache_dir=self.config.cache_dir,
            **self.config.extra_load_kwargs,
        )
        if not isinstance(raw, Dataset):
            raise TypeError(f"Expected a Dataset, got {type(raw)}")
        ds = raw
        ds = ds.filter(self._has_numeric_answer)
        if self.require_reasoning_complete:
            ds = ds.filter(self._is_reasoning_complete)
        ds = self._format_dataset(ds)
        ds = ds.filter(self._fits_length_budget)
        if self.max_samples is not None:
            ds = ds.shuffle(seed=3407).select(range(min(self.max_samples, len(ds))))
        return ds

    def format_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert an OpenR1-Math row to ``{"text": str}`` for SFT.

        Args:
            example: A single raw dataset row with ``"problem"``,
                ``"solution"``, and ``"answer"`` fields.

        Returns:
            A dict with a single ``"text"`` key containing the chat-templated
            string.
        """
        assistant_content = self._build_assistant_response(
            example["solution"], example["answer"]
        )
        messages = [
            {"role": "system", "content": MATH_REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": assistant_content},
        ]
        return {
            "text": self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        }

    @staticmethod
    def _build_assistant_response(solution: str, answer: str) -> str:
        """Convert ``<think>`` traces to ``<reasoning>/<answer>`` format.

        Args:
            solution: The raw solution string, optionally wrapped in
                ``<think>`` / ``</think>`` tags.
            answer: The numeric answer string.

        Returns:
            A formatted string with ``<reasoning>`` and ``<answer>`` tags.
        """
        reasoning = solution.strip()
        reasoning = reasoning.removeprefix("<think>").removesuffix("</think>").strip()
        return (
            f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>{answer.strip()}</answer>"
        )

    @staticmethod
    def _has_numeric_answer(example: dict[str, Any]) -> bool:
        """Return ``True`` if the answer can be parsed as a number."""
        try:
            float(str(example["answer"]).strip().replace(",", ""))
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _is_reasoning_complete(example: dict[str, Any]) -> bool:
        """Return ``True`` if at least one reasoning trace is marked complete."""
        flags = example.get("is_reasoning_complete", [])
        return any(flags) if flags else True

    def _fits_length_budget(self, example: dict[str, Any]) -> bool:
        """Return ``True`` if the formatted text fits the token budget."""
        token_count = len(
            self.tokenizer.encode(example["text"], add_special_tokens=False)
        )
        return token_count <= self.max_example_tokens
