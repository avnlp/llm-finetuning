import groq
import pandas as pd
import argparse
import json
from collections import Counter
from tqdm import tqdm
import backoff
import jsonlines
import random

# Initialize Groq client
client = groq.Client()


@backoff.on_exception(backoff.expo, groq.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


KNOWLEDGE_INSTRUCTIONS = {
    "earnings_calls_data": "Answer the question based on the given earnings call transcript."
}

PROMPT_DICT = {
    "multi": (
        "You'll be provided with an instruction, along with evidence and possibly some preceding sentences. "
        "When there are preceding sentences, your focus should be on the sentence that comes after them. "
        "Your job is to determine if the evidence is relevant to the initial instruction and the preceding context, and provides useful information to complete the task described in the instruction. "
        "If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].\n\n"
        "###\nInstruction: Given four answer options, A, B, C, and D, choose the best answer.\n\n"
        "Input: Earth rotating causes\n"
        "A: the cycling of AM and PM\nB: the creation of volcanic eruptions\nC: the cycling of the tides\nD: the creation of gravity\n\n"
        "Evidence: Rotation causes the day-night cycle which also creates a corresponding cycle of temperature and humidity. Sea level rises and falls twice a day as the earth rotates.\n\n"
        "Rating: [Relevant]\n"
        "Explanation: The evidence explicitly mentions that the rotation causes a day-night cycle, as described in the answer option A.\n\n"
        "###\nInstruction: {instruction}\n\n"
        "Evidence: {evidence}\n\n"
        "Rating:"
    )
}


def load_jsonlines(file):
    with jsonlines.open(file, "r") as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def postprocess(results):
    raw_output = results["content"]
    print(raw_output)
    if "\nExplanation:" in raw_output:
        explanation = raw_output.split("\nExplanation:")[1].strip()
        score_string = raw_output.split("\nExplanation:")[0]
        score = None
        for i in range(1, 6):
            if str(i) in score_string:
                score = int(i)
        return score, explanation
    else:
        return "", ""


def process_input(example):
    return PROMPT_DICT["multi"].format_map(example)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs="+")
    parser.add_argument("--output_file_name", type=str)
    parser.add_argument("--multi_retrieval", action="store_true")
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--model_name", type=str, default="groq-model")
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    client.api_key = args.api_key

    examples = []
    for input_file in args.input_files:
        if input_file.endswith(".json"):
            examples += json.load(open(input_file))
        else:
            examples += load_jsonlines(input_file)

    result_list = []
    if args.n is not None and len(examples) > args.n:
        examples = random.sample(examples, k=args.n)

    task_types = Counter([item.get("dataset_name", "unknown") for item in examples])
    print(task_types)

    for idx, example in tqdm(enumerate(examples)):
        if "instruction" not in example and "question" in example:
            example["instruction"] = (
                KNOWLEDGE_INSTRUCTIONS.get(example.get("dataset_name", ""), "")
                + example["question"]
            )

        input_prompt = process_input(example)
        try:
            results = completions_with_backoff(
                model=args.model_name,
                messages=[{"role": "user", "content": input_prompt}],
                max_tokens=200,
                temperature=0,
            )
            score, explanation = postprocess(results)
            result_list.append(
                {
                    "input": example,
                    "score": score,
                    "explanation": explanation,
                    "raw_output": results["content"],
                }
            )

            if idx % 20 == 0:
                print(f"Input: {example['instruction']}")
                print(f"Output: {example.get('output', '')}")
                print(f"Evidence: {example.get('evidence', '')}")
                print(f"Score: {score} ({explanation})")

        except Exception as e:
            print(f"Error processing example {idx}: {e}")

    with open(args.output_file_name, "w") as outfile:
        json.dump(result_list, outfile)


if __name__ == "__main__":
    main()
