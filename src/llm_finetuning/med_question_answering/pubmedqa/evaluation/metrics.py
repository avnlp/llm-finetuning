"""Evaluation Metrics for PubMedQA generations.

This module provides evaluation framework for PubMedQA outputs that typically include
yes/no/maybe answers with supporting explanations.
"""

import re
from typing import Any, Dict, List

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from rapidfuzz import fuzz
from tenacity import retry, wait_random_exponential
from tqdm import tqdm


class PubMedQAEvaluator:
    """Evaluator for PubMedQA generation outputs.

    This class provides methods to evaluate the answer accuracy and explanation quality
    of PubMedQA outputs that typically include yes/no/maybe answers with explanations.
    """

    LLM_AS_A_JUDGE_MODEL = "llama-3.3-70b-versatile"

    @staticmethod
    def extract_answer(text: str) -> str:
        """Extract yes/no/maybe answer from text using regex patterns.

        Args:
            text: Input text containing answer

        Returns:
            Extracted answer (yes/no/maybe) or empty string if not found
        """
        patterns = [
            r"answer:\s*(yes|no|maybe)",
            r"answer\s*is\s*:\s*(yes|no|maybe)",
            r"^.*?\b(yes|no|maybe)\b.*?$",
        ]

        text_lower = text.lower().strip()
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        return ""

    @staticmethod
    def evaluate_answer_accuracy(
        predicted_answer: str, expected_answer: str
    ) -> Dict[str, float]:
        """Evaluate answer accuracy using exact match and fuzzy matching.

        Args:
            predicted_answer: Model's extracted answer
            expected_answer: Ground truth answer

        Returns:
            Dictionary containing accuracy scores
        """
        exact_match = 1.0 if predicted_answer == expected_answer else 0.0

        # Fuzzy matching for partial credit
        fuzzy_score = fuzz.ratio(predicted_answer, expected_answer) / 100

        return {
            "exact_match": exact_match,
            "fuzzy_accuracy": fuzzy_score,
            "answer_accuracy": (exact_match + fuzzy_score) / 2,
        }

    @retry(wait=wait_random_exponential(multiplier=1, max=60))
    def evaluate_explanation_quality(
        self, question: str, explanation: str, expected_explanation: str
    ) -> float:
        """Evaluate explanation quality using LLM-as-judge.

        Args:
            question: Original user question
            explanation: Generated explanation from model
            expected_explanation: Ground truth explanation

        Returns:
            Explanation quality score (0.0-1.0)
        """
        test_case = LLMTestCase(
            input=question,
            actual_output=explanation,
            expected_output=expected_explanation,
        )

        # Configure evaluation criteria for PubMedQA explanations
        explanation_metric = GEval(
            name="PubMedQA Explanation Quality",
            criteria="Evaluate the medical accuracy, completeness, and relevance of the explanation. Consider if it properly supports the answer and uses appropriate medical reasoning.",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=self.LLM_AS_A_JUDGE_MODEL,
        )
        explanation_metric.measure(test_case)
        return explanation_metric.score

    def evaluate_generation(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Run full evaluation pipeline on a single PubMedQA generation.

        Evaluation steps:
        1. Answer extraction
        2. Answer accuracy evaluation
        3. Explanation quality evaluation
        4. Composite scoring

        Args:
            data: Dictionary containing:
                - question: User question
                - generation: Model output
                - answer: Expected answer (yes/no/maybe)
                - explanation: Expected explanation

        Returns:
            Dictionary containing all evaluation metrics
        """
        generation = data.get("generation", "")
        metrics = {}

        # Answer Extraction
        predicted_answer = self.extract_answer(generation)
        expected_answer = data.get("answer", "").lower()

        # Answer Accuracy Evaluation
        accuracy_metrics = self.evaluate_answer_accuracy(
            predicted_answer, expected_answer
        )
        metrics.update(accuracy_metrics)

        # Explanation Quality Evaluation
        metrics["explanation_quality"] = self.evaluate_explanation_quality(
            question=data["question"],
            explanation=generation,
            expected_explanation=data.get("explanation", ""),
        )

        # Composite Scoring
        metrics["overall_score"] = (
            metrics["answer_accuracy"] * 0.6 + metrics["explanation_quality"] * 0.4
        )

        return metrics

    def evaluate_generations(
        self, generations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate a batch of PubMedQA generations and returns aggregate scores.

        Args:
            generations: List of data dictionaries for evaluation

        Returns:
            Dictionary of average scores across all metrics
        """
        # Initialize score accumulator
        score_accumulator = {
            "exact_match": 0.0,
            "fuzzy_accuracy": 0.0,
            "answer_accuracy": 0.0,
            "explanation_quality": 0.0,
            "overall_score": 0.0,
        }

        # Process each generation with progress tracking
        for data in tqdm(generations, desc="Evaluating PubMedQA generations"):
            metrics = self.evaluate_generation(data)

            # Aggregate scores
            for metric in score_accumulator:
                if metric in metrics:
                    score_accumulator[metric] += metrics[metric]

        # Calculate averages and round to 4 decimals
        return {
            metric: round(total / len(generations), 4)
            for metric, total in score_accumulator.items()
        }
