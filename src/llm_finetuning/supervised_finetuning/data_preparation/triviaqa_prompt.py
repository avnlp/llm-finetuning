"""Prompt constants and formatting utilities for the TriviaQA dataset.

TriviaQA (reading comprehension config) pairs trivia questions with web search results
that may contain the answer. Each example exposes a nested ``answer`` dict and a
``search_results`` dict with multiple retrieved passages.

The formatting step:

1. Extracts the canonical answer string from ``answer["value"]``.
2. Extracts the first ``search_context`` passage as a ``context`` field injected
   into the user prompt, giving the model evidence to ground its answer.

HuggingFace ID: ``mandarjoshi/trivia_qa``, config ``rc``, split ``train``.
"""

from llm_finetuning.core import PromptTemplate


TEMPLATE = PromptTemplate(
    system_prompt="You are a trivia expert. Answer questions accurately and concisely using the provided context.",
    user_template="Question: {question}\nContext: {context}",
    response_field="answer",
)


def format_triviaqa_example(example: dict) -> dict:
    """Pre-process a raw TriviaQA example for chat-template formatting.

    The raw TriviaQA dataset stores the answer as a nested dict (``answer["value"]``
    for the canonical string) and search results as a dict of parallel lists. This
    function flattens both structures so they can be substituted into
    ``USER_PROMPT_TEMPLATE`` without further processing.

    Args:
        example: A single TriviaQA dataset example containing:

            - ``answer`` (dict): with a ``value`` (str) key for the canonical answer.
            - ``search_results`` (dict): with a ``search_context`` (list of str) key
              holding retrieved passage texts. May be empty.
            - ``question`` (str): the trivia question.

    Returns:
        The same example dict with:

        - ``answer`` replaced by the flat canonical answer string.
        - A new ``context`` key set to the first search passage, or an empty
          string if no passages are available.
    """
    example["answer"] = example["answer"]["value"]
    search_contexts = example["search_results"].get("search_context", [])
    example["context"] = search_contexts[0] if search_contexts else ""
    return example
