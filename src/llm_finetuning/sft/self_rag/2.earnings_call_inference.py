import groq
import pandas as pd
import argparse
import json
from collections import Counter
from tqdm import tqdm
import backoff
import jsonlines
import random
from sacrebleu.metrics import BLEU

bleu = BLEU()


def load_jsonlines(file):
    with jsonlines.open(file, "r") as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def postprocess(results):
    raw_output = results["choices"][0]["message"]["content"]
    return raw_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, required=True, help="Input JSONL file with examples"
    )
    parser.add_argument(
        "--output_file_name", type=str, required=True, help="Output JSON file name"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="Path to file containing Groq API key",
    )
    parser.add_argument(
        "--org_name", type=str, required=True, help="Groq organization name"
    )
    parser.add_argument(
        "--n", type=int, default=None, help="Number of examples to process"
    )
    args = parser.parse_args()

    # Set Groq API credentials
    with open(args.api_key) as f:
        groq.api_key = f.read().strip()
    groq.organization = args.org_name

    # Load input examples
    examples = load_jsonlines(args.input_file)
    result_list = []

    # Sample examples if a limit is provided
    if args.n is not None:
        examples = random.sample(examples, args.n)

    for idx, example in tqdm(enumerate(examples)):
        try:
            # Call Groq API for completions
            results = completions_with_backoff(
                model="llama3-8b-8192",   
                messages=[
                    {"role": "user", "content": example["instruction"]},
                ],
                request_timeout=60,
                max_tokens=200,
            )
            pred = postprocess(results)
            metric_result = bleu.corpus_score([pred], [[example["answers"]]]).score

            # Append result to the list
            result_list.append(
                {
                    "input": example,
                    "pred": pred,
                    "bleu": metric_result,
                    "raw_output": results["choices"][0]["message"]["content"],
                }
            )

            if idx % 20 == 0:
                print("Input: {}".format(example["instruction"]))
                print("Gold: {}".format(example["answers"]))
                print("Prediction: {}".format(pred))
                print("BLEU Score: {}".format(metric_result))

        except Exception as e:
            print(f"Error processing example {idx}: {e}")

        # Save interim results every 100 examples
        if idx % 100 == 0:
            with open(args.output_file_name + "_tmp", "w") as outfile:
                json.dump(result_list, outfile)

    # Save final results
    with open(args.output_file_name, "w") as outfile:
        json.dump(result_list, outfile)


if __name__ == "__main__":
    main()
