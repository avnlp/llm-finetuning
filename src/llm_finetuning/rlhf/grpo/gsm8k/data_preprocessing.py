from datasets import load_dataset


def extract_hash_answer(text: str) -> str | None:
    return text.split("####")[1].strip() if "####" in text else None


def format_gsm8k_dataset(dataset, system_prompt: str) -> dict:
    return dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )


def get_tokenized_lengths(dataset, tokenizer) -> list:
    return dataset.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
        batched=False,
    )["tokens"]


def get_max_prompt_length(tokenized_lengths: list) -> int:
    return max(len(tokens) for tokens in tokenized_lengths) + 1
