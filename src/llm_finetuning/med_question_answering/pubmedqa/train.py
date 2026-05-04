"""PubMedQA Model Fine-Tuning Pipeline.

Fine-tune LLMs using GRPO for PubMedQA dataset.
Includes model initialization, dataset loading, prompt formatting,
GRPO training configuration, and multi-objective reward functions.

The fine-tuning evaluates:
1. Answer correctness (Yes/No/Maybe)
2. XML structure compliance

Key Components:
- Unsloth for accelerated LoRA fine-tuning
- vLLM for efficient generation
- TRL's GRPOTrainer for policy optimization
- Custom reward functions for domain-specific evaluation
"""

import yaml
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel  # type: ignore[import-untyped]
from vllm import SamplingParams  # type: ignore[import-not-found]

from .prompts import PUBMEDQA_USER_PROMPT, SYSTEM_PROMPT
from .reward_functions import RewardManager
from .train_config import PubMedQATrainConfig


def main(config_path: str) -> None:
    """Fine-tune LLM using GRPO for PubMedQA.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    # Load configuration if path is provided
    if config_path:
        with open(config_path) as f:
            config = PubMedQATrainConfig(**yaml.safe_load(f))

        model_name = config.model_name
        max_seq_length = config.max_seq_length
        lora_rank = config.lora_rank
        gpu_memory_utilization = config.gpu_memory_utilization
        temperature = config.temperature
        top_k = config.top_k
        top_p = config.top_p
        min_p = config.min_p
        max_new_tokens = config.max_new_tokens
        learning_rate = config.learning_rate
        weight_decay = config.weight_decay
        warmup_ratio = config.warmup_ratio
        lr_scheduler_type = config.lr_scheduler_type
        optim = config.optim
        logging_steps = config.logging_steps
        per_device_train_batch_size = config.per_device_train_batch_size
        gradient_accumulation_steps = config.gradient_accumulation_steps
        num_generations = config.num_generations
        max_steps = config.max_steps
        save_steps = config.save_steps
        report_to = config.report_to
        output_dir = config.output_dir
        dataset_name = config.dataset_name
        split = config.split

    # Initialize model with 4-bit quantization and memory optimization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # Apply Parameter-Efficient Fine-Tuning (LoRA)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Load and preprocess PubMedQA dataset
    dataset = load_dataset(dataset_name, split=split)

    # Format prompts with system and user instructions
    dataset = dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": PUBMEDQA_USER_PROMPT.format(
                        question=x["question"], context=x["context"]
                    ),
                },
            ],
            "label": x["final_decision"],  # Yes / No / Maybe
        }
    )

    # Configure sampling parameters
    vllm_sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        max_tokens=max_new_tokens,
    )

    # GRPO training configuration
    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        logging_steps=logging_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_completion_length=max_new_tokens,
        max_steps=max_steps,
        save_steps=save_steps,
        report_to=report_to,
        output_dir=output_dir,
    )

    # Initialize GRPO trainer with multi-objective rewards
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            RewardManager.correctness_reward_func,
            RewardManager.xml_structure_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # Execute training process
    trainer.train()
