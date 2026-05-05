"""Prompt constants and formatting utilities for the ARC dataset.

ARC stands for AI2 Reasoning Challenge.

ARC-Challenge is a dataset of grade-school science multiple-choice questions. Each
example contains a question, a set of answer choices stored as parallel ``label`` and
``text`` lists, and the correct answer key.

HuggingFace ID: ``allenai/ai2_arc``, config ``ARC-Challenge``, split ``train``.
"""

from llm_finetuning.core import PromptTemplate


TEMPLATE = PromptTemplate(
    system_prompt="You are a science reasoning expert. Analyze the question, evaluate each choice, and provide the correct answer.",
    user_template="Question: {question}\nChoices:\n{choices}\n\nThink step by step and provide the correct answer letter.",
    response_field="answerKey",
)


def format_choices(labels: list[str], texts: list[str]) -> str:
    """Format parallel ARC choice labels and texts into a single labelled string.

    Converts the raw parallel lists from the ARC dataset (e.g. ``labels=["A", "B"]``
    and ``texts=["text1", "text2"]``) into a human-readable block suitable for
    template substitution into ``USER_PROMPT_TEMPLATE``.

    Args:
        labels: List of answer-choice labels, e.g. ``["A", "B", "C", "D"]``.
        texts: List of answer-choice texts corresponding to each label.

    Returns:
        A newline-separated string of ``"<label>. <text>"`` entries, e.g.::

            "A. Plant cells have a wall\\nB. Animal cells have a wall\\n..."
    """
    return "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
