"""Evaluation Metrics for BioASQ generations.

This module provides evaluation framework for BioASQ outputs that follow a specific XML structure
with answer and item tags for factoid/list questions.

The evaluation includes:
1. XML structural integrity checks
2. Answer correctness using fuzzy matching and LLM-as-a-judge metrics
3. Traditional information retrieval metrics (F1, Precision, Recall)

The generations contain these required XML tags:
<answer>, <item>
"""

import re
from typing import List, Dict, Any
from rapidfuzz import fuzz, process

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from tenacity import retry, wait_random_exponential
from tqdm import tqdm


class BioASQEvaluator:
    """Evaluator for BioASQ generation outputs.

    This class provides methods to evaluate the structural integrity and correctness
    of BioASQ outputs that follow a specific XML format with answer and item tags.
    """

    LLM_AS_A_JUDGE_MODEL = "llama-3.3-70b-versatile"

    @staticmethod
    def extract_items(text: str) -> List[str]:
        """Extract items from <item> tags using regex.

        Handles multi-line content and missing tags gracefully.

        Args:
            text: Input text containing XML tags

        Returns:
            List of extracted items, or empty list if not found
        """
        pattern = re.compile(r"<item>\s*(.*?)\s*</item>", re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(text)
        return [match.strip() for match in matches] if matches else []

    @staticmethod
    def evaluate_xml_structure(generation: str) -> Dict[str, float]:
        """Evaluate XML structural integrity by checking required tags.

        Required tags: answer, item.
        Scores based on proper nesting and presence of required tags.

        Args:
            generation: XML-formatted generation text

        Returns:
            Dictionary containing structure scores
        """
        scores = {}
        
        # Check answer tags
        answer_open_count = generation.count("<answer>")
        answer_close_count = generation.count("</answer>")
        scores["answer_structure"] = 1.0 if answer_open_count == 1 and answer_close_count == 1 else 0.0
        
        # Check item tags
        item_open_count = generation.count("<item>")
        item_close_count = generation.count("</item>")
        item_valid = item_open_count >= 1 and item_open_count == item_close_count
        scores["item_structure"] = 1.0 if item_valid else 0.0
        
        # Calculate overall structural integrity score
        scores["overall_structure_score"] = sum(scores.values()) / len(scores)
        return scores

    @staticmethod
    def calculate_f1(precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def evaluate_list_correctness(self, predicted_items: List[str], expected_items: List[str]) -> Dict[str, float]:
        """Evaluate correctness of predicted items against expected items using fuzzy matching.
        
        Uses F1, precision, and recall metrics adapted for list answers.
        """
        if not expected_items:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        
        if not predicted_items:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        
        # Calculate matches using fuzzy matching
        matched_predicted = set()
        matched_expected = set()
        
        # For each expected item, find the best matching predicted item
        for exp_item in expected_items:
            best_match, score, idx = process.extractOne(
                exp_item, predicted_items, scorer=fuzz.token_sort_ratio
            )
            if score >= 70:  # Threshold for considering a match
                matched_expected.add(exp_item)
                matched_predicted.add(best_match)
        
        # Calculate precision, recall, and F1
        precision = len(matched_predicted) / len(predicted_items) if predicted_items else 0
        recall = len(matched_expected) / len(expected_items) if expected_items else 0
        f1 = self.calculate_f1(precision, recall)
        
        return {
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    @retry(wait=wait_random_exponential(multiplier=1, max=60))
    def evaluate_answer_correctness_llm(self, question: str, answer: str, expected_answer: str) -> float:
        """Evaluate factual correctness using LLM-as-judge for BioASQ answers.
        
        Args:
            question: Original user question
            answer: Generated answer from model (formatted string)
            expected_answer: Ground truth answer (formatted string)

        Returns:
            Correctness score (0.0-1.0)
        """
        test_case = LLMTestCase(
            input=question, 
            actual_output=answer, 
            expected_output=expected_answer
        )

        # Configure evaluation criteria for BioASQ
        correctness_metric = GEval(
            name="BioASQ Answer Correctness",
            criteria="Determine if the actual output contains factually correct answers based on the expected output. Consider partial matches and the completeness of the answer list.",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=self.LLM_AS_A_JUDGE_MODEL
        )
        correctness_metric.measure(test_case)
        return correctness_metric.score

    def evaluate_generation(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Run full evaluation pipeline on a single BioASQ generation.
        
        Evaluation steps:
        1. XML structure validation
        2. Content extraction from XML tags
        3. Traditional IR metrics (F1, precision, recall)
        4. LLM-as-judge metrics evaluation
        5. Composite scoring

        Args:
            data: Dictionary containing:
                - question: User question
                - context: Source context
                - answers: List of expected answers
                - generation: XML-formatted output

        Returns:
            Dictionary containing all evaluation metrics
        """
        generation = data.get("generation", "")
        metrics = {}

        # Structural Evaluation
        structure_metrics = self.evaluate_xml_structure(generation)
        metrics.update(structure_metrics)

        # Content Extraction
        predicted_items = self.extract_items(generation)
        expected_items = data.get("answers", [])
        
        # Traditional IR Metrics
        ir_metrics = self.evaluate_list_correctness(predicted_items, expected_items)
        metrics.update(ir_metrics)
        
        # Format answers for LLM evaluation
        predicted_str = ", ".join(predicted_items) if predicted_items else "No answer provided"
        expected_str = ", ".join(expected_items) if expected_items else "No expected answers"

        # LLM-as-Judge Evaluation
        metrics["llm_correctness"] = self.evaluate_answer_correctness_llm(
            question=data["question"],
            answer=predicted_str,
            expected_answer=expected_str,
        )

        # Composite Scoring
        content_keys = ["f1", "llm_correctness"]
        metrics["content_score"] = sum(metrics[k] for k in content_keys) / len(content_keys)

        # Final overall score averages structure and content
        metrics["overall_score"] = (metrics["overall_structure_score"] + metrics["content_score"]) / 2

        return metrics

    def evaluate_generations(self, generations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate a batch of BioASQ generations and returns aggregate scores.
        
        Args:
            generations: List of data dictionaries for evaluation

        Returns:
            Dictionary of average scores across all metrics
        """
        # Initialize score accumulator
        score_accumulator = {
            "answer_structure": 0.0,
            "item_structure": 0.0,
            "overall_structure_score": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "llm_correctness": 0.0,
            "content_score": 0.0,
            "overall_score": 0.0,
        }

        # Process each generation with progress tracking
        for idx, data in enumerate(tqdm(generations, desc="Evaluating BioASQ generations")):
            metrics = self.evaluate_generation(data)

            # Aggregate scores
            for metric in score_accumulator.keys():
                if metric in metrics:
                    score_accumulator[metric] += metrics[metric]

        # Calculate averages and round to 4 decimals
        return {metric: round(total / len(generations), 4) for metric, total in score_accumulator.items()}
