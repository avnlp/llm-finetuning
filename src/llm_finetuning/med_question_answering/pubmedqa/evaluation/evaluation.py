"""Evaluate PubMedQA generations."""

import json

from .metrics_pubmedqa import PubMedQAEvaluator


def main(generations_file: str, scores_file: str):
    """Run evaluation on PubMedQA generations."""
    # Load generations
    with open(generations_file) as f:
        generations = [json.loads(line) for line in f]

    # Initialize evaluator
    evaluator = PubMedQAEvaluator()

    # Evaluate generations
    scores = evaluator.evaluate_generations(generations)
    print(scores)

    # Save scores
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main("generations_pubmedqa.jsonl", "scores_pubmedqa.json")
