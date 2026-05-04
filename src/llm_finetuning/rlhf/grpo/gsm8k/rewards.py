"""Reward functions for GRPO training on GSM8K dataset."""

import re
from typing import Any


def extract_xml_answer(text: str) -> str:
    """Extract content between <answer> tags."""
    if "<answer>" not in text:
        return ""
    answer = text.rsplit("<answer>", maxsplit=1)[-1]
    return answer.split("</answer>")[0].strip()


def count_xml(text: str) -> float:
    """Count XML structure points for response."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.rsplit("\n</answer>\n", maxsplit=1)[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.rsplit("\n</answer>", maxsplit=1)[-1]) - 1) * 0.001
    return count


def correctness_reward_func(
    prompts: Any, completions: Any, answer: Any, **kwargs: Any
) -> list[float]:
    """Reward correct final answers."""
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions: Any, **kwargs: Any) -> list[float]:
    """Reward integer answers."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions: Any, **kwargs: Any) -> list[float]:
    """Strict XML format validation."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions: Any, **kwargs: Any) -> list[float]:
    """Lenient XML format validation."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def xmlcount_reward_func(completions: Any, **kwargs: Any) -> list[float]:
    """XML structure scoring."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# ================================
# mistral reward functions
# ================================
def correctness_reward_func_mistral(
    prompts: Any, completions: Any, answer: Any, **kwargs: Any
) -> list[float]:
    """Reward correct answers for Mistral."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func_mistral(completions: Any, **kwargs: Any) -> list[float]:
    """Reward integer answers for Mistral."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func_mistral(completions: Any, **kwargs: Any) -> list[float]:
    """Strict format validation for Mistral."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]


def soft_format_reward_func_mistral(completions: Any, **kwargs: Any) -> list[float]:
    """Soft format validation for Mistral."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]


def xmlcount_reward_func_mistral(completions: Any, **kwargs: Any) -> list[float]:
    """XML count for Mistral."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# ================================
# phi-4 reward functions
# ================================
def correctness_reward_func_phi(
    prompts: Any, completions: Any, answer: Any, **kwargs: Any
) -> list[float]:
    """Reward correct answers for Phi-4."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func_phi(completions: Any, **kwargs: Any) -> list[float]:
    """Reward integer answers for Phi-4."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func_phi(completions: Any, **kwargs: Any) -> list[float]:
    """Strict format for Phi-4."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]


def soft_format_reward_func_phi(completions: Any, **kwargs: Any) -> list[float]:
    """Soft format for Phi-4."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]


def xmlcount_reward_func_phi(completions: Any, **kwargs: Any) -> list[float]:
    """XML count for Phi-4."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# ================================
# gemma3 reward functions
# ================================
def correctness_reward_func_gemma(
    prompts: Any, completions: Any, answer: Any, **kwargs: Any
) -> list[float]:
    """Reward correct answers for Gemma3."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func_gemma(completions: Any, **kwargs: Any) -> list[float]:
    """Reward integer answers for Gemma3."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func_gemma(completions: Any, **kwargs: Any) -> list[float]:
    """Strict format for Gemma3."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]


def soft_format_reward_func_gemma(completions: Any, **kwargs: Any) -> list[float]:
    """Soft format for Gemma3."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]


def xmlcount_reward_func_gemma(completions: Any, **kwargs: Any) -> list[float]:
    """XML count for Gemma3."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
