"""Prompts for MedQA training."""

SYSTEM_PROMPT = """You are given a <Question>, its <Context>.
1. Think through the question and context; show your reasoning within <think>…</think>.
2. Provide your final answer within <answer>…</answer>.

Respond exactly in this format:

<think>
…your step-by-step reasoning…
</think>

<answer>
…your concise answer…
</answer>
"""

USER_PROMPT = """Answer the question: {question} using the context: {context}"""
