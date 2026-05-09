"""Class-based SFT dataset loaders."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedTokenizerBase

from llm_finetuning.core import BaseDatasetLoader, DatasetConfig, PromptTemplate
from llm_finetuning.supervised_finetuning.data_preparation.arc_prompt import (
    TEMPLATE as ARC_TEMPLATE,
)
from llm_finetuning.supervised_finetuning.data_preparation.arc_prompt import (
    format_choices,
)
from llm_finetuning.supervised_finetuning.data_preparation.earnings_call_prompt import (
    TEMPLATE as EARNINGS_CALL_TEMPLATE,
)
from llm_finetuning.supervised_finetuning.data_preparation.earnings_call_prompt import (
    format_earnings_example,
)
from llm_finetuning.supervised_finetuning.data_preparation.factscore_prompt import (
    TEMPLATE as FACTSCORE_TEMPLATE,
)
from llm_finetuning.supervised_finetuning.data_preparation.popqa_prompt import (
    TEMPLATE as POPQA_TEMPLATE,
)
from llm_finetuning.supervised_finetuning.data_preparation.triviaqa_prompt import (
    TEMPLATE as TRIVIAQA_TEMPLATE,
)
from llm_finetuning.supervised_finetuning.data_preparation.triviaqa_prompt import (
    format_triviaqa_example,
)


class SFTDatasetLoader(BaseDatasetLoader):
    """Base SFT loader: applies chat template, returns single 'text' column."""

    def __init__(
        self,
        config: DatasetConfig,
        template: PromptTemplate,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """Store dataset config, prompt template, and tokenizer for SFT formatting."""
        super().__init__(config)
        self.template = template
        self.tokenizer = tokenizer

    def format_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map a raw example to a single ``text`` field for ``SFTTrainer``.

        The pipeline is: ``preprocess`` → ``render_user`` → assemble three-turn
        conversation → ``apply_chat_template`` → ``{"text": str}``.

        Args:
            example: A single raw dataset row as a ``dict``.

        Returns:
            A ``dict`` with one key, ``"text"``, whose value is the full
            conversation serialised by the tokenizer's chat template.
        """
        example = self.preprocess(example)
        messages = [
            {"role": "system", "content": self.template.system_prompt},
            {"role": "user", "content": self.template.render_user(**example)},
            {
                "role": "assistant",
                "content": str(example[self.template.response_field]),
            },
        ]
        return {
            "text": self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        }

    def preprocess(self, example: dict[str, Any]) -> dict[str, Any]:
        """Normalize raw fields before template substitution.

        Override in subclasses for dataset-specific transformations
        (e.g. renaming ``transcript`` to ``context`` or flattening nested
        fields). The default implementation returns ``example`` unchanged.

        Args:
            example: A single raw dataset row as a ``dict``.

        Returns:
            Example data ready for ``PromptTemplate.render_user``.
        """
        return example


class ARCLoader(SFTDatasetLoader):
    """SFT loader for ARC-Challenge."""

    CONFIG = DatasetConfig(dataset_id="allenai/ai2_arc", subset="ARC-Challenge")

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, config: DatasetConfig | None = None
    ) -> None:
        """Initialize ARC-Challenge formatting with ``tokenizer``."""
        super().__init__(config or self.CONFIG, ARC_TEMPLATE, tokenizer)

    def preprocess(self, example: dict[str, Any]) -> dict[str, Any]:
        """Render the ARC choices dict as a printable multiple-choice block."""
        example["choices"] = format_choices(
            example["choices"]["label"], example["choices"]["text"]
        )
        return example


class TriviaQALoader(SFTDatasetLoader):
    """SFT loader for TriviaQA (rc config)."""

    CONFIG = DatasetConfig(dataset_id="mandarjoshi/trivia_qa", subset="rc")

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, config: DatasetConfig | None = None
    ) -> None:
        """Configure the TriviaQA RC loader for ``tokenizer``."""
        super().__init__(config or self.CONFIG, TRIVIAQA_TEMPLATE, tokenizer)

    def preprocess(self, example: dict[str, Any]) -> dict[str, Any]:
        """Extract the flattened answer and context fields expected by the template."""
        return format_triviaqa_example(example)


class FactScoreLoader(SFTDatasetLoader):
    """SFT loader for the FactScore biography dataset."""

    CONFIG = DatasetConfig(
        dataset_id="awinml/factscore_unlabelled_alpaca_13b_retrieval"
    )

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, config: DatasetConfig | None = None
    ) -> None:
        """Prepare FactScore examples for serialization with ``tokenizer``."""
        super().__init__(config or self.CONFIG, FACTSCORE_TEMPLATE, tokenizer)


class PopQALoader(SFTDatasetLoader):
    """SFT loader for PopQA."""

    CONFIG = DatasetConfig(dataset_id="akariasai/PopQA")

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, config: DatasetConfig | None = None
    ) -> None:
        """Bind the PopQA prompt template to ``tokenizer``."""
        super().__init__(config or self.CONFIG, POPQA_TEMPLATE, tokenizer)


class EarningsCallLoader(SFTDatasetLoader):
    """SFT loader for the Earnings Calls QA dataset."""

    CONFIG = DatasetConfig(dataset_id="lamini/earnings-calls-qa")

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, config: DatasetConfig | None = None
    ) -> None:
        """Set up Earnings Calls QA formatting with ``tokenizer``."""
        super().__init__(config or self.CONFIG, EARNINGS_CALL_TEMPLATE, tokenizer)

    def preprocess(self, example: dict[str, Any]) -> dict[str, Any]:
        """Copy ``transcript`` to ``context`` for prompt rendering."""
        return format_earnings_example(example)
