"""Prompt constants for the PopQA factual question answering dataset.

PopQA is an entity-centric QA dataset covering a wide range of factual knowledge.
Each example provides a ``question`` (str) and ``possible_answers`` (list of str).
Because there are multiple valid answers, the full list is passed as the training
target; the dataloader converts it to a string via ``str()``.

No field transformation is required — ``question`` maps directly into
``TEMPLATE.user_template`` and ``possible_answers`` is used as the response field.

Note:
    The dataset's canonical split is ``test`` (no dedicated train split exists).
    All 14k test examples are used for fine-tuning.

HuggingFace ID: ``akariasai/PopQA``, split ``test``.
"""

from llm_finetuning.core import PromptTemplate


TEMPLATE = PromptTemplate(
    system_prompt="You are a knowledgeable assistant. Answer factual questions concisely.",
    user_template="Question: {question}",
    response_field="possible_answers",
)
