"""Evaluate BioASQ generations."""

import json

from .metrics_bioasq import BioASQEvaluator


def main(generations_file: str, scores_file: str):
    """Run evaluation on BioASQ generations."""
    # Load generations
    with open(generations_file) as f:
        generations = [json.loads(line) for line in f]

    # Initialize evaluator
    evaluator = BioASQEvaluator()

    # Evaluate generations
    scores = evaluator.evaluate_generations(generations)
    print(scores)

    # Save scores
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main("generations_bioasq.jsonl", "scores_bioasq.json")
