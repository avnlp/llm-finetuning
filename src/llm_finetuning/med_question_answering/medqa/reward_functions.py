"""Reward Functions for MedQA.

We use 2 reward functions for training MedQA:
- Correctness reward function: Measures the correctness of the model's answer.
- XML structure reward function: Checks presence of all new tags in the model's answer.
"""

import re

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from tenacity import retry, wait_random_exponential


class RewardManager:
    """Reward functions for MedQA."""

    @staticmethod
    def extract_tag_content(text: str, tag: str) -> str:
        """Extract content from XML tags using regex pattern matching.

        Args:
            text: Input text containing XML tags
            tag: Specific XML tag to extract content from

        Returns:
            Extracted content as string, or empty string if not found

        Example:
            >>> extract_tag_content("<think>Reasoning</think>", "think")
            'Reasoning'
        """
        # Use non-greedy matching with re.DOTALL to capture multiline content
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
        match = pattern.search(text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def evaluate_xml_structure(generation: str) -> dict[str, float]:
        """Evaluate presence and proper pairing of required XML tags.

        Required tags: think, answer

        Args:
            generation: Full text of model response

        Returns:
            Dictionary containing:
            - Individual tag scores (1.0 if properly formed, else 0.0)
            - Overall structure score (average of individual scores)
        """
        required_tags = ["think", "answer"]
        scores = {}

        for tag in required_tags:
            # Count opening and closing tags
            open_count = generation.count(f"<{tag}>")
            close_count = generation.count(f"</{tag}>")
            # Valid if exactly one pair exists
            scores[f"{tag}_structure"] = (
                1.0 if open_count == 1 and close_count == 1 else 0.0
            )

        # Calculate composite score
        scores["overall_structure_score"] = sum(scores.values()) / len(required_tags)
        return scores

    @staticmethod
    @retry(wait=wait_random_exponential(multiplier=1, max=60))
    def evaluate_answer_correctness(
        question: str, answer: str, expected_answer: str
    ) -> float:
        """Evaluate answer correctness using DeepEval LLM-as-a-Judge metric.

        Utilizes GEval with criteria: "Determine if the actual output is factually correct
        based on expected output and context". Implements exponential backoff for retries.

        Args:
            question: User query/instruction given to model
            answer: Model-generated response
            expected_answer: Ground truth reference answer

        Returns:
            Normalized correctness score between 0.0 (incorrect) and 1.0 (perfect)
        """
        test_case = LLMTestCase(
            input=question, actual_output=answer, expected_output=expected_answer
        )

        correctness_metric = GEval(
            name="Answer Correctness",
            criteria="Determine if the actual output is factually correct based on expected output and context",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
        )
        correctness_metric.measure(test_case)
        return correctness_metric.score

    @staticmethod
    def correctness_reward_func(
        completions, instruction, answer, **kwargs
    ) -> list[float]:
        """Reward function that evaluates answer correctness.

        Args:
            completions: List of model completion objects
            instruction: User query/instruction
            answer: Expected answer (ground truth)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            List of scaled reward scores (range: 1-20) for each completion
        """
        responses = [completion[0]["content"] for completion in completions]
        scores = []

        for response in responses:
            # Get base correctness score (0.0-1.0)
            base_score = RewardManager.evaluate_answer_correctness(
                question=instruction, answer=response, expected_answer=answer
            )
            # Scale to reward range [1, 20]
            scaled_score = base_score * 20
            scores.append(scaled_score)

        return scores

    @staticmethod
    def xml_structure_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that evaluates XML structure compliance.

        Args:
            completions: List of model completion objects
            **kwargs: Additional keyword arguments (unused)

        Returns:
            List of scaled reward scores (range: 1-10) for each completion
        """
        responses = [completion[0]["content"] for completion in completions]
        scores = []

        for response in responses:
            structure_scores = RewardManager.evaluate_xml_structure(response)
            # Scale overall structure score to [1, 10]
            scaled_score = structure_scores["overall_structure_score"] * 10
            scores.append(scaled_score)

        return scores
