"""DPO training script using UltraFeedback dataset.

Based on Unsloth's Zephyr DPO implementation, modified for Qwen3-4B and YAML config.
"""
# type: ignore[import-not-found, misc]

# Based on: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb
# Modified to work with Qwen3-4B, load args from YAML, and support any model.

import yaml
from dpo_process_ultrafeedback import (  # type: ignore[import-not-found]
    apply_chat_template_example,
    get_datasets,
)
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer  # type: ignore[import-untyped]


# Load configuration from YAML
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Patch DPO Trainer for Unsloth Optimizations
PatchDPOTrainer()

# Initialize model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["model"]["name"],
    max_seq_length=config["model"]["max_seq_length"],
    dtype=config["model"]["dtype"],
    load_in_4bit=config["model"]["load_in_4bit"],
)

# Load and preprocess dataset
raw_datasets = get_datasets(
    {config["dataset"]["name"]: config["dataset"]["sample_percentage"]},
    splits=config["dataset"]["splets"],
)
column_names = list(raw_datasets["train"].features)

raw_datasets = raw_datasets.map(
    apply_chat_template_example,
    fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
    num_proc=config["dataset"]["num_proc"],
    remove_columns=column_names,
    desc="Formatting comparisons with prompt template",
)

# Rename dataset columns
for split in ["train", "test"]:
    raw_datasets[split] = raw_datasets[split].rename_columns(
        {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
    )

# Apply LoRA configuration
model = FastLanguageModel.get_peft_model(
    model,
    r=config["lora"]["r"],
    target_modules=config["lora"]["target_modules"],
    lora_alpha=config["lora"]["lora_alpha"],
    lora_dropout=config["lora"]["lora_dropout"],
    bias=config["lora"]["bias"],
    use_gradient_checkpointing=config["lora"]["use_gradient_checkpointing"],
    random_state=config["lora"]["random_state"],
    use_rslora=config["lora"]["use_rslora"],
    loftq_config=config["lora"]["loftq_config"],
)

# Configure DPO training
dpo_args = DPOConfig(
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    warmup_ratio=config["training"]["warmup_ratio"],
    num_train_epochs=config["training"]["num_train_epochs"],
    learning_rate=config["training"]["learning_rate"],
    logging_steps=config["training"]["logging_steps"],
    optim=config["training"]["optim"],
    weight_decay=config["training"]["weight_decay"],
    lr_scheduler_type=config["training"]["lr_scheduler_type"],
    seed=config["training"]["seed"],
    output_dir=config["training"]["output_dir"],
    report_to=config["training"]["report_to"],
)

# Initialize DPOTrainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_args,
    beta=config["dpo"]["beta"],
    train_dataset=raw_datasets["train"],
    eval_dataset=raw_datasets["test"],
    tokenizer=tokenizer,
    max_length=config["dpo"]["max_length"],
    max_prompt_length=config["dpo"]["max_prompt_length"],
)

# Start training
dpo_trainer.train()
