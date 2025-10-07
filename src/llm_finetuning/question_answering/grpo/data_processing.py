def format_hotpot_context(example):
    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]
    return " ".join(
        f"{title}: {' '.join(sents)}" for title, sents in zip(titles, sentences)
    )


def preprocess_hotpot(examples):
    inputs = [
        f"Question: {q}\nContext: {format_hotpot_context(ex)}\nAnswer:"
        for q, ex in zip(examples["question"], examples)
    ]
    groups = [
        {"answer": ans, "supporting_facts": sf}
        for ans, sf in zip(examples["answer"], examples["supporting_facts"])
    ]
    return {"query": inputs, "answer": examples["answer"], "group": groups}


def format_musique_context(example):
    return "\n".join(example["paragraphs"])


def preprocess_musique(examples):
    inputs = [
        f"Question: {q}\nContext: {format_musique_context(ex)}\nAnswer:"
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
