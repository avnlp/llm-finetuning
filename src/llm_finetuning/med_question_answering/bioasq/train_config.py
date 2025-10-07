"""Pydantic configuration model for BioASQ training."""

from pydantic import BaseModel


class BioASQTrainConfig(BaseModel):
    """Configuration for BioASQ model fine-tuning."""

    # Model parameters
    model_name: str = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    max_seq_length: int = 8192  # BioASQ contexts can be longer
    lora_rank: int = 64
    gpu_memory_utilization: float = 0.8

    # Sampling parameters
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    min_p: float = 0.1
    max_new_tokens: int = 1024

    # Training parameters
    learning_rate: float = 4e-6
    weight_decay: float = 0.02
    warmup_ratio: float = 0.15
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    logging_steps: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    num_generations: int = 4
    max_steps: int = 200
    save_steps: int = 10
    report_to: str = "wandb"
    output_dir: str = "bioasq_outputs"

    # Dataset parameters
    dataset_name: str = "qiaojin/BioASQ"
    split: str = "train"
