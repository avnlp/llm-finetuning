"""Data processing functions for HotpotQA and MuSiQUE datasets."""

from typing import Any, Dict, List


def format_hotpot_context(example: Dict[str, Any]) -> str:
    """Format HotpotQA context into a single string."""
    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]
    return " ".join(
        f"{title}: {' '.join(sents)}" for title, sents in zip(titles, sentences)
    )


def preprocess_hotpot(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Preprocess HotpotQA examples for GRPO training."""
    inputs = [  # type: ignore[no-untyped-call]
        f"Question: {q}\nContext: {format_hotpot_context(ex)}\nAnswer:"  # type: ignore[arg-type]
        for q, ex in zip(examples["question"], examples)
    ]
    groups = [
        {"answer": ans, "supporting_facts": sf}
        for ans, sf in zip(examples["answer"], examples["supporting_facts"])
    ]
    return {"query": inputs, "answer": examples["answer"], "group": groups}


def format_musique_context(example: Dict[str, Any]) -> str:
    """Format MuSiQUE context from paragraphs."""
    return "\n".join(example["paragraphs"])


def preprocess_musique(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Preprocess MuSiQUE examples for GRPO training."""
    inputs = [  # type: ignore[no-untyped-call]
        f"Question: {q}\nContext: {format_musique_context(ex)}\nAnswer:"  # type: ignore[arg-type]
        for q, ex in zip(examples["question"], examples)
    ]
    groups = [
        {"answer": ans, "aliases": aliases, "decomposition": decomp}
        for ans, aliases, decomp in zip(
            examples["answer"],
            examples["answer_aliases"],
            examples["question_decomposition"],
        )
    ]
    return {"query": inputs, "answer": examples["answer"], "group": groups}
