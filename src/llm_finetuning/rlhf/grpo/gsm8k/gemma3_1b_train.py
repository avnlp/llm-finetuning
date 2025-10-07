# Code taken from https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(1B)-GRPO.ipynb
# Dataset preprocessing for GSM8K is imported from the data_preprocessing.py file, and reward functions are imported from the rewards.py file.

import yaml
from data_preprocessing import (
    format_gsm8k_dataset,
    get_max_prompt_length,
    get_tokenized_lengths,
)
from datasets import load_dataset
from rewards import (
    check_answer,
    check_numbers,
    match_format_approximately,
    match_format_exactly,
)
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastModel, is_bfloat16_supported


# Load configuration from YAML file
with open("gemma3_1b_train.yaml") as f:
    config = yaml.safe_load(f)

# Extract configuration values
model_config = config["model"]
dataset_config = config["dataset"]
training_config = config["training"]["config"]
generation_config = config["training"]["generation"]
reward_config = config["rewards"]
output_config = config["output"]

# Set special tokens from configuration
system_prompt = dataset_config["preprocessing"]["system_prompt"]
reasoning_start = dataset_config["preprocessing"]["special_tokens"]["reasoning_start"]
reasoning_end = dataset_config["preprocessing"]["special_tokens"]["reasoning_end"]
solution_start = dataset_config["preprocessing"]["special_tokens"]["solution_start"]
solution_end = dataset_config["preprocessing"]["special_tokens"]["solution_end"]

# Load model with configuration
model, tokenizer = FastModel.from_pretrained(
    model_name=model_config["name"],
    max_seq_length=model_config["max_seq_length"],
    load_in_4bit=model_config["quantization"]["load_in_4bit"],
    load_in_8bit=model_config["quantization"]["load_in_8bit"],
    full_finetuning=False,
)

# Configure LoRA
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=model_config["lora"]["r"],
    lora_alpha=model_config["lora"]["alpha"],
    lora_dropout=model_config["lora"]["dropout"],
    bias="none",
    random_state=3407,
)

# Load and prepare dataset
dataset = load_dataset(
    dataset_config["name"], dataset_config["config"], split=dataset_config["split"]
)
dataset = format_gsm8k_dataset(
    dataset, system_prompt, reasoning_start, reasoning_end, solution_start, solution_end
)

# Calculate tokenized lengths
tokenized_lengths = get_tokenized_lengths(dataset, tokenizer)
max_prompt_length = get_max_prompt_length(tokenized_lengths)

# Override generation config with calculated values if needed
if generation_config["max_prompt_length"] == "auto":
    generation_config["max_prompt_length"] = max_prompt_length
if generation_config["max_completion_length"] == "auto":
    generation_config["max_completion_length"] = (
        model_config["max_seq_length"] - max_prompt_length
    )

# Map reward functions from names to actual functions
REWARD_FUNCTIONS = {
    "match_format_exactly": match_format_exactly,
    "match_format_approximately": match_format_approximately,
    "check_answer": check_answer,
    "check_numbers": check_numbers,
}

reward_funcs = [REWARD_FUNCTIONS[name] for name in reward_config["functions"]]

# Create GRPO configuration
training_args = GRPOConfig(
    learning_rate=training_config["learning_rate"],
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=training_config["weight_decay"],
    warmup_ratio=training_config["warmup_ratio"],
    lr_scheduler_type=training_config["lr_scheduler"],
    optim=training_config["optim"],
    logging_steps=training_config["logging_steps"],
    per_device_train_batch_size=training_config["batch_size"],
    bf16=is_bfloat16_supported() and training_config.get("bf16", False),
    fp16=not is_bfloat16_supported() and training_config.get("fp16", True),
    gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
    num_generations=generation_config["num_generations"],
    max_prompt_length=generation_config["max_prompt_length"],
    max_completion_length=generation_config["max_completion_length"],
    max_steps=training_config["max_steps"],
    save_steps=training_config["save_steps"],
    max_grad_norm=training_config["max_grad_norm"],
    report_to=training_config["report_to"],
    output_dir=output_config["dir"],
    # Generation parameters
    temperature=generation_config["temperature"],
    top_p=generation_config["top_p"],
    max_new_tokens=generation_config["max_new_tokens"],
)

# Create trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=dataset,
)

print("Starting GRPO training with configuration:")
print(f"Model: {model_config['name']}")
print(f"Dataset: {dataset_config['name']} ({dataset_config['split']} split)")
print(f"Batch size: {training_config['batch_size']}")
print(f"Learning rate: {training_config['learning_rate']}")
print(f"Reward functions: {', '.join(reward_config['functions'])}")
print(f"Max steps: {training_config['max_steps']}")
print(f"Output directory: {output_config['dir']}")

trainer.train()
print("Training completed!")
