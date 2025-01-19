import pandas as pd
import argparse
import json
from collections import Counter
from tqdm import tqdm
import backoff
import groq
import jsonlines
import random


PROMPT_DICT = {
    "context": (
        "Given an instruction, please make a judgment on whether finding some external documents from the web (e.g., Wikipedia) helps to generate a better response. Please answer [Yes] or [No] and write an explanation.\n\n"
        "##\nInstruction: Give three tips for staying healthy.\n"
        "Need retrieval?: [Yes]\n"
        "Explanation: There might be some online sources listing three tips for staying healthy or some reliable sources to explain the effects of different behaviors on health. So retrieving documents is helpful to improve the response to this query.\n\n"
        "##\nInstruction: Describe a time when you had to make a difficult decision.\n"
        "Need retrieval?: [No]\n"
        "Explanation: This instruction is asking about some personal experience and thus it does not require one to find some external documents.\n\n"
        "##\nInstruction: Write a short story in third person narration about a protagonist who has to make an important career decision.\n"
        "Need retrieval?: [No]\n"
        "Explanation: This instruction asks us to write a short story, which does not require external evidence to verify.\n\n"
        "##\nInstruction: What is the capital of France?\n"
        "Need retrieval?: [Yes]\n"
        "Explanation: While the instruction simply asks us to answer the capital of France, which is a widely known fact, retrieving web documents for this question can still help.\n\n"
        "##\nInstruction: Find the area of a circle given its radius. Radius = 4\n"
        "Need retrieval?: [No]\n"
        "Explanation: This is a math question and although we may be able to find some documents describing a formula, it is unlikely to find a document exactly mentioning the answer.\n\n"
        "##\nInstruction: Arrange the words in the given sentence to form a grammatically correct sentence. quickly the brown fox jumped\n"
        "Need retrieval?: [No]\n"
        "Explanation: This task doesn't require any external evidence, as it is a simple grammatical question.\n\n"
        "##\nInstruction: Explain the process of cellular respiration in plants."
        "Need retrieval?: [Yes]\n"
        "Explanation: This instruction asks for a detailed description of a scientific concept, and is highly likely that we can find a reliable and useful document to support the response.\n\n"
        "##\nInstruction:{instruction}\n"
        "Need retrieval?: "
    ),
}


def load_jsonlines(file):
    with jsonlines.open(file, "r") as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def postprocess(results):
    raw_output = results["choices"][0]["message"]["content"]
    if "\nExplanation:" in raw_output:
        explanation = raw_output.split("\nExplanation:")[1]
        if explanation[0] == " ":
            explanation = explanation[1:]
        decision_token = raw_output.split("\nExplanation:")[0]
        if decision_token is None:
            return "", explanation
        else:
            return decision_token, explanation
    else:
        return "", ""


def process_input(example):
    return PROMPT_DICT["context"].format_map(example)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs="+")
    parser.add_argument("--output_file_name", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--org_name", type=str)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="groq-8b-model")
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    # Set Groq API credentials
    with open(args.api_key) as f:
        groq.api_key = f.read().strip()
    groq.organization = args.org_name

    examples = []
    for input_file in args.input_files:
        if input_file.endswith(".json"):
            examples += json.load(open(input_file))
        else:
            examples += load_jsonlines(input_file)

    result_list = []
    if args.n is not None:
        if args.random:
            examples = random.sample(examples, args.n)
        else:
            examples = examples[: args.n]

    for idx, example in tqdm(enumerate(examples)):
        input = process_input(example)
        if idx % 5 == 0:
            print(input)
        try:
            results = completions_with_backoff(
                model=args.model_name,
                messages=[
                    {"role": "user", "content": input},
                ],
                request_timeout=60,
                max_tokens=200,
            )
            decision_token, explanation = postprocess(results)
            result_list.append(
                {
                    "input": example,
                    "decision_token": decision_token,
                    "explanation": explanation,
                    "raw_output": results["choices"][0]["message"]["content"],
                }
            )
            if idx % 5 == 0:
                print("Input: {}".format(example["instruction"]))
                print("Explanation: {}".format(explanation))
                print("Decision Token: {}".format(decision_token))

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
        if idx % 100 == 0:
            with open(args.output_file_name + "_tmp", "w") as outfile:
                json.dump(result_list, outfile)

    with open(args.output_file_name, "w") as outfile:
        json.dump(result_list, outfile)


if __name__ == "__main__":
    main()

