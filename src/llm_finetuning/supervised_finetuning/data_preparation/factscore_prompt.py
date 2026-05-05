"""Prompt constants for the FactScore biography generation dataset.

This module defines the prompts used to fine-tune a model on the FactScore dataset,
which contains biographical summaries paired with source prompts.
The raw ``input`` field holds the biography request (e.g. "Tell me a bio of ...") and
``output`` holds the generated biography text. No additional field transformations are
required — the fields map directly into ``TEMPLATE.user_template`` and
``TEMPLATE.response_field``.

HuggingFace ID: ``awinml/factscore_unlabelled_alpaca_13b_retrieval``, split ``train``.
"""

from llm_finetuning.core import PromptTemplate


TEMPLATE = PromptTemplate(
    system_prompt="You are a factual assistant. When asked about a person, generate an accurate biographical summary. Only include facts you are confident are true.",
    user_template="{input}",
    response_field="output",
)
