"""Evaluate generations."""

import json

from .metrics import Evaluator


def main(generations_file: str, scores_file: str):
    """Run evaluation on generations."""
    # Load generations
    with open(generations_file) as f:
        generations = [json.loads(line) for line in f]

    # Initialize evaluator
    evaluator = Evaluator()

    # Evaluate generations
    scores = evaluator.evaluate_generations(generations)
    print(scores)

    # Save scores
    with open(scores_file, "w") as f:
        json.dump(scores, f)


if __name__ == "__main__":
    main("generations.jsonl", "scores.json")
