import json
import os

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import GenerationConfig
from unsloth import FastLanguageModel


# Configuration
MODEL_PATH = "./unsloth_sft_musique"
DATASET_NAME = "your-hf-username/musique-processed"
TEST_SPLIT = "test"
MAX_SEQ_LENGTH = 4096
OUTPUT_DIR = "./evaluation_results"
EVAL_RESULTS_PATH = os.path.join(OUTPUT_DIR, "eval_results.json")
SAMPLE_PREDS_PATH = os.path.join(OUTPUT_DIR, "sample_predictions.txt")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model and tokenizer
print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    load_in_4bit=True,
)

# Set special tokens
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens(
    {
        "additional_special_tokens": [
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
        ]
    }
)
model.eval()

# Load test dataset
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)
test_dataset = dataset[TEST_SPLIT]


# Evaluation function
def evaluate_qa(model, tokenizer, dataset, max_new_tokens=100):
    """Evaluate QA performance on Musique dataset."""
    # Load metrics
    exact_match_metric = evaluate.load("exact_match")
    evaluate.load("f1")

    # Generation config
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
    )

    # Formatting template for evaluation
    def format_eval_prompt(example):
        return (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Answer the question based on the context below. Keep your response concise.\n\n"
            f"Question: {example['question']}\n\n"
            f"Context: {example['context']}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    # Run evaluation
    predictions = []
    references = []
    generation_times = []

    for example in tqdm(test_dataset, desc="Evaluating"):
        # Format input
        input_text = format_eval_prompt(example)
        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH
        ).to("cuda")

        # Generate prediction with timing
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)
        gen_time = time.time() - start_time
        generation_times.append(gen_time)

        # Decode prediction
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract assistant's response
        try:
            start_idx = pred_text.find("<|start_header_id|>assistant<|end_header_id|>")
            if start_idx == -1:
                prediction = pred_text.split("assistant<|end_header_id|>")[-1].strip()
            else:
                prediction = pred_text[start_idx:].split("\n\n", 1)[1].strip()

            prediction = prediction.split("<|eot_id|>")[0].strip()
        except Exception as e:
            print(f"Error extracting prediction: {e}")
            prediction = ""

        predictions.append(prediction)
        references.append(example["answer"])

    # Compute metrics
    exact_match = exact_match_metric.compute(
        predictions=predictions,
        references=references,
        ignore_case=True,
        ignore_punctuation=True,
    )["exact_match"]

    # Calculate F1 scores
    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_norm = pred.lower().strip()
        ref_norm = ref.lower().strip()

        pred_tokens = pred_norm.split()
        ref_tokens = ref_norm.split()

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1_scores.append(0.0)
            continue

        common_tokens = set(pred_tokens) & set(ref_tokens)
        precision = len(common_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = len(common_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)

    avg_f1 = np.mean(f1_scores) * 100
    avg_gen_time = np.mean(generation_times)

    # Create detailed results
    results = {
        "exact_match": exact_match,
        "f1": avg_f1,
        "avg_generation_time": avg_gen_time,
        "predictions": predictions,
        "references": references,
        "samples": [],
    }

    # Save sample predictions
    for i in range(min(10, len(test_dataset))):
        results["samples"].append(
            {
                "question_id": test_dataset[i]["question_id"],
                "question": test_dataset[i]["question"],
                "context": test_dataset[i]["context"][:500] + "...",
                "prediction": predictions[i],
                "reference": references[i],
            }
        )

    return results


if __name__ == "__main__":
    import time

    start_time = time.time()

    print("\nStarting evaluation...")
    eval_results = evaluate_qa(model, tokenizer, test_dataset)

    # Print results
    print("\n" + "=" * 50)
    print(f"Evaluation Results ({len(test_dataset)} samples)")
    print("=" * 50)
    print(f"Exact Match (EM): {eval_results['exact_match']:.2f}%")
    print(f"Average F1 Score: {eval_results['f1']:.2f}%")
    print(f"Avg Generation Time: {eval_results['avg_generation_time']:.2f}s")
    print("=" * 50)

    # Save results
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Full results saved to {EVAL_RESULTS_PATH}")

    # Save sample predictions
    with open(SAMPLE_PREDS_PATH, "w") as f:
        f.write(f"Musique Evaluation Results ({len(test_dataset)} samples)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Exact Match (EM): {eval_results['exact_match']:.2f}%\n")
        f.write(f"Average F1 Score: {eval_results['f1']:.2f}%\n")
        f.write(f"Avg Generation Time: {eval_results['avg_generation_time']:.2f}s\n\n")

        f.write("Sample Predictions:\n")
        f.write("=" * 80 + "\n")
        for i, sample in enumerate(eval_results["samples"]):
            f.write(f"\nSample {i + 1} (ID: {sample['question_id']})\n")
            f.write("-" * 80 + "\n")
            f.write(f"Question: {sample['question']}\n\n")
            f.write(f"Context: {sample['context']}\n\n")
            f.write(f"Reference Answer: {sample['reference']}\n")
            f.write(f"Model Prediction: {sample['prediction']}\n")
            f.write("\n" + "=" * 80 + "\n\n")

    print(f"Sample predictions saved to {SAMPLE_PREDS_PATH}")
    print(f"Total evaluation time: {time.time() - start_time:.2f} seconds")
