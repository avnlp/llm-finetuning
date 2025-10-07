"""Reward Functions for BioASQ.

Reward functions for training on BioASQ dataset:
- Factoid/list answer correctness: Compares model's predicted answers with ground truth
- XML structure reward: Ensures proper <answer> and <item> formatting
"""

import re
from typing import List

from rapidfuzz import fuzz


class BioASQRewardManager:
    """Reward functions for BioASQ."""

    @staticmethod
    def extract_answers(text: str) -> List[str]:
        """Extract answers from <item> tags."""
        matches = re.findall(r"<item>\s*(.*?)\s*</item>", text, re.IGNORECASE)
        return [match.strip() for match in matches]

    @staticmethod
    def evaluate_xml_structure(generation: str) -> dict[str, float]:
        """Check that <answer> and <item> tags exist and are properly formatted."""
        open_answer_count = generation.count("<answer>")
        close_answer_count = generation.count("</answer>")
        item_count = generation.count("<item>")

        valid_structure = (
            open_answer_count == 1 and close_answer_count == 1 and item_count >= 1
        )
        return {"answer_structure": 1.0 if valid_structure else 0.0}

    @staticmethod
    def evaluate_answer_correctness(
        response: str, expected_answers: List[str]
    ) -> float:
        """Check if model's answers match ground truth using fuzzy matching."""
        pred_answers = BioASQRewardManager.extract_answers(response)
        if not pred_answers:
            return 0.0

        total_score = 0.0
        for pred in pred_answers:
            best_match = max(
                [fuzz.ratio(pred.lower(), exp.lower()) for exp in expected_answers]
            )
            total_score += best_match / 100.0  # Normalize to 0-1

        return total_score / len(pred_answers)  # Average score

    @staticmethod
    def correctness_reward_func(
        completions, instruction, answers, **kwargs
    ) -> list[float]:
        """Reward correctness: Scale based on answer quality."""
        responses = [completion[0]["content"] for completion in completions]
        scores = []
        for response in responses:
            base_score = BioASQRewardManager.evaluate_answer_correctness(
                response, answers
            )
            scores.append(base_score * 25)  # scale to [0, 25]
        return scores

    @staticmethod
    def xml_structure_reward_func(completions, **kwargs) -> list[float]:
        """Reward XML compliance: +5 if properly structured."""
        responses = [completion[0]["content"] for completion in completions]
        scores = []
        for response in responses:
            structure_scores = BioASQRewardManager.evaluate_xml_structure(response)
            scores.append(structure_scores["answer_structure"] * 5)  # scale to [0, 5]
        return scores
