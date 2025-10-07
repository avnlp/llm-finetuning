"""BioASQ Model Inference Pipeline.

This script performs batch inference on the BioASQ dataset.

The pipeline:
1. Loads a quantized model optimized with Unsloth
2. Processes a validation dataset with biomedical questions
3. Generates responses using a structured prompt template
4. Captures full input/output metadata
5. Saves results in JSON Lines format for evaluation
"""

import json
from pathlib import Path

import yaml
from datasets import load_dataset
from unsloth import FastLanguageModel
from vllm import SamplingParams

from .inference_config_bioasq import BioASQInferenceConfig
from .prompts_bioasq import BIOASQ_USER_PROMPT, SYSTEM_PROMPT


def main(config_path: str):
    """Run the BioASQ inference process."""
    # Load configuration from YAML file
    config_path = Path(config_path)
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config = BioASQInferenceConfig(**config_dict)

    # Initialize model with optimized 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=config.model.lora_rank,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
    )

    # Load validation dataset
    dataset = load_dataset(config.dataset.name, split=config.dataset.split)

    # Initialize results container
    results = []

    # Configure generation parameters (fixed for all samples)
    sampling_params = SamplingParams(
        temperature=config.generation.temperature,
        top_k=config.generation.top_k,
        top_p=config.generation.top_p,
        min_p=config.generation.min_p,
        max_tokens=config.generation.max_new_tokens,
    )

    print(f"Starting inference on {len(dataset)} validation examples...")

    # Process each example in the dataset
    for i, row in enumerate(dataset):
        # Extract question and context from dataset row
        question = row["question"]
        context = row["context"]
        row["answers"]  # List of expected answers

        # Format messages using predefined prompt templates
        messages = [
            # System prompt defines model behavior constraints
            {"role": "system", "content": SYSTEM_PROMPT},
            # User prompt incorporates question and context
            {
                "role": "user",
                "content": BIOASQ_USER_PROMPT.format(
                    question=question, context=context
                ),
            },
        ]

        # Apply chat template to create model-ready input
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate response using optimized inference
        generation = (
            model.fast_generate(
                text,
                sampling_params=sampling_params,
            )[0]
            .outputs[0]
            .text
        )

        # Preserve original data and add inference artifacts
        result = dict(row)
        result.update(
            {
                "index": i,
                "messages": messages,
                "formatted_text": text,
                "generation": generation,
            }
        )
        results.append(result)

        # Periodic progress updates with samples
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processed {i + 1}/{len(dataset)} examples")
            print(
                f"Sample input: {question[:100]}{'...' if len(question) > 100 else ''}"
            )
            print(
                f"Sample output: {generation[:200]}{'...' if len(generation) > 200 else ''}"
            )
            print("-" * 50)

    print(f"\nCompleted inference on {len(results)} examples!")
    print(f"Results contain keys: {list(results[0].keys())}")

    # Save results to JSON Lines file
    output_file = config.output.file_name

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Results saved to: {output_file}")
    print(f"File contains {len(results)} records in JSONL format")


if __name__ == "__main__":
    main("config_bioasq.yaml")
