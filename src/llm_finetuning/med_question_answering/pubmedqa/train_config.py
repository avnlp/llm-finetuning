"""Pydantic configuration model for PubMedQA training."""

from pydantic import BaseModel


class PubMedQATrainConfig(BaseModel):
    """Configuration for PubMedQA model fine-tuning."""

    # Model parameters
    model_name: str = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
    max_seq_length: int = (
        4096  # PubMed abstracts are shorter than MedQA, so 4k is usually enough
    )
    lora_rank: int = 32
    gpu_memory_utilization: float = 0.7

    # Sampling parameters
    temperature: float = 0.6
    top_k: int = 20
    top_p: float = 0.95
    min_p: float = 0.0
    max_new_tokens: int = 512

    # Training parameters
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    logging_steps: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 4
    max_steps: int = 100
    save_steps: int = 5
    report_to: str = "wandb"
    output_dir: str = "outputs"

    # Dataset parameters
    dataset_name: str = "qiaojin/PubMedQA"
    split: str = "train"
