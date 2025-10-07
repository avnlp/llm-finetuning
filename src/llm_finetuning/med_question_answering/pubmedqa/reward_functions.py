"""Reward Functions for PubMedQA.

We use the following reward functions for training PubMedQA:
- Correctness reward function: Compares model's predicted Yes/No/Maybe with ground truth.
- XML structure reward function: Ensures proper <answer>...</answer> formatting.
"""

import re


class RewardManager:
    """Reward functions for PubMedQA."""

    @staticmethod
    def extract_answer(text: str) -> str:
        """Extract answer from <answer> tag."""
        match = re.search(r"<answer>\s*(Yes|No|Maybe)\s*</answer>", text, re.IGNORECASE)
        return match.group(1).strip().lower() if match else ""

    @staticmethod
    def evaluate_xml_structure(generation: str) -> dict[str, float]:
        """Check that <answer>...</answer> tag exists and is properly paired."""
        open_count = generation.count("<answer>")
        close_count = generation.count("</answer>")
        valid = open_count == 1 and close_count == 1
        return {"answer_structure": 1.0 if valid else 0.0}

    @staticmethod
    def evaluate_answer_correctness(response: str, expected_answer: str) -> float:
        """Check if model's <answer> matches ground truth label (yes/no/maybe)."""
        pred = RewardManager.extract_answer(response)
        gold = expected_answer.strip().lower()
        return 1.0 if pred == gold else 0.0

    @staticmethod
    def correctness_reward_func(
        completions, instruction, answer, **kwargs
    ) -> list[float]:
        """Reward correctness: +20 if prediction matches gold, else 0."""
        responses = [completion[0]["content"] for completion in completions]
        scores = []
        for response in responses:
            base_score = RewardManager.evaluate_answer_correctness(
                response, answer
            )  # 0.0 or 1.0
            scores.append(base_score * 20)  # scale to [0, 20]
        return scores

    @staticmethod
    def xml_structure_reward_func(completions, **kwargs) -> list[float]:
        """Reward XML compliance: +10 if <answer>...</answer> exists, else 0."""
        responses = [completion[0]["content"] for completion in completions]
        scores = []
        for response in responses:
            structure_scores = RewardManager.evaluate_xml_structure(response)
            scores.append(structure_scores["answer_structure"] * 10)  # scale to [0, 10]
        return scores
