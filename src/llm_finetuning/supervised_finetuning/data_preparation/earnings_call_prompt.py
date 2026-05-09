"""Prompt constants and formatting utilities for the Earnings Calls QA dataset.

This module defines the prompts used to fine-tune a model on the Lamini Earnings Calls
QA dataset, which contains question-answer pairs grounded in earnings call transcripts.
The formatting step renames the raw ``transcript`` field to ``context`` so it matches
the ``{context}`` placeholder in ``USER_PROMPT_TEMPLATE``.

HuggingFace ID: ``lamini/earnings-calls-qa``, split ``train``.
"""

from llm_finetuning.core import PromptTemplate


TEMPLATE = PromptTemplate(
    system_prompt="You are a financial analyst. Answer questions accurately based on earnings call transcripts.",
    user_template="Context: {context}\n\nQuestion: {question}",
    response_field="answer",
)


def format_earnings_example(example: dict) -> dict:
    """Pre-process a raw Earnings Calls example for chat-template formatting.

    The raw dataset exposes the transcript under the key ``transcript``, but
    ``TEMPLATE.user_template`` expects the key ``context``. This function copies
    the value so template substitution works without modifying the template.

    Args:
        example: A single Earnings Calls dataset example containing at minimum
            ``transcript`` (str), ``question`` (str), and ``answer`` (str) fields.

    Returns:
        The same example dict with a new ``context`` key set to the value of
        ``transcript``. The original ``transcript`` key is preserved.
    """
    example["context"] = example["transcript"]
    return example
