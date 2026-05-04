"""BioASQ Model Fine-Tuning Pipeline.

Fine-tuning LLMs using GRPO for BioASQ dataset.
Includes specialized reward functions for factoid/list question answering.
"""

import yaml
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel  # type: ignore[import-untyped]
from vllm import SamplingParams  # type: ignore[import-not-found]

from .prompts_bioasq import (  # type: ignore[import-untyped]
    BIOASQ_USER_PROMPT,
    SYSTEM_PROMPT,
)
from .reward_functions_bioasq import BioASQRewardManager  # type: ignore[import-untyped]
from .train_config_bioasq import BioASQTrainConfig  # type: ignore[import-untyped]


def main(config_path: str) -> None:
    """Fine-tune LLM using GRPO for BioASQ."""
    # Load configuration
    with open(config_path) as f:
        config = BioASQTrainConfig(**yaml.safe_load(f))

    # Initialize model (same structure as PubMedQA but with BioASQ config)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=config.lora_rank,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=config.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Load and preprocess BioASQ dataset
    dataset = load_dataset(config.dataset_name, split=config.split)

    # Format prompts
    dataset = dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": BIOASQ_USER_PROMPT.format(
                        question=x["question"], context=x["context"]
                    ),
                },
            ],
            "answers": x["answers"],  # List of expected answers
        }
    )

    # Configure sampling parameters
    vllm_sampling_params = SamplingParams(
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        min_p=config.min_p,
        max_tokens=config.max_new_tokens,
    )

    # GRPO training configuration
    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        learning_rate=config.learning_rate,
        # ... (rest of training arguments similar to PubMedQA)
    )

    # Initialize trainer with BioASQ reward functions
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[
            BioASQRewardManager.correctness_reward_func,
            BioASQRewardManager.xml_structure_reward_func,
        ],
    )

    # Start training
    trainer.train()

    # Save model
    trainer.save_model(config.output_dir)
