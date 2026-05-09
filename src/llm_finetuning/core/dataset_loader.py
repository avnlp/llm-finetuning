"""Abstract base class for all dataset loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Configuration for loading and mapping a HuggingFace dataset."""

    dataset_id: str
    subset: str | None = None
    cache_dir: str | None = None
    remove_source_columns: bool = True
    num_proc: int = 1
    extra_load_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra keyword arguments forwarded verbatim to ``load_dataset``."""


class BaseDatasetLoader(ABC):
    """Loads a HuggingFace dataset split and formats it for training.

    Output column schema by trainer type:
      - SFT  (SFTTrainer):   {"text": str}
      - GRPO (GRPOTrainer):  {"prompt": list[dict], "answer": str}
      - DPO/ORPO:            {"prompt": str, "chosen": str, "rejected": str}
      - PPO:                 {"prompt": str} → tokenized to {"input_ids": list[int]}
      - KTO:                 {"prompt": str, "completion": str, "label": bool}
    """

    def __init__(self, config: DatasetConfig) -> None:
        """Create a loader bound to a concrete dataset configuration."""
        self.config = config

    def load(self, split: str) -> Dataset:
        """Load a dataset split and map it to the trainer's expected schema.

        Args:
            split: The HuggingFace dataset split name, e.g. ``"train"`` or
                ``"test"``.

        Returns:
            A ``Dataset`` whose columns match the output schema of
            ``format_example`` for this loader.
        """
        ds = load_dataset(
            self.config.dataset_id,
            self.config.subset,
            split=split,
            cache_dir=self.config.cache_dir,
            **self.config.extra_load_kwargs,
        )
        return self._format_dataset(ds)

    def _format_dataset(self, ds: Dataset) -> Dataset:
        """Map ``format_example`` over a dataset and optionally remove source columns.

        Args:
            ds: The raw ``Dataset`` returned by ``load_dataset``.

        Returns:
            A new ``Dataset`` with columns produced by ``format_example``.
            Source columns are removed when ``config.remove_source_columns``
            is ``True``.
        """
        remove_cols = (
            list(ds.column_names) if self.config.remove_source_columns else None
        )
        return ds.map(
            self.format_example,
            remove_columns=remove_cols,
            num_proc=self.config.num_proc,
        )

    def tokenize(
        self,
        ds: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        text_column: str = "prompt",
    ) -> Dataset:
        """Pre-tokenize a text column into ``input_ids`` for trainers that need it.

        Used by PPO, where ``PPOTrainer`` expects a pre-tokenized
        ``input_ids`` column rather than raw strings.

        Args:
            ds: A formatted ``Dataset`` containing a text column.
            tokenizer: The tokenizer to encode the text.
            text_column: Name of the column containing the text to tokenize.

        Returns:
            A new ``Dataset`` with a single ``input_ids`` column.
        """

        def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
            return {
                "input_ids": tokenizer(batch[text_column], padding=False)["input_ids"]
            }

        return ds.map(
            _tokenize,
            batched=True,
            remove_columns=ds.column_names,
            num_proc=self.config.num_proc,
        )

    @abstractmethod
    def format_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw example into the schema expected by the trainer.

        Args:
            example: A single row from the raw HuggingFace dataset, as a
                ``dict`` mapping column names to values.

        Returns:
            A ``dict`` whose keys are the columns expected by the trainer.
            The required schema depends on trainer type — see the class
            docstring for the full mapping.
        """
        raise NotImplementedError
