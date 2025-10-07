# File: train_triviaqa.py
import torch
import yaml
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


# Load configuration from YAML file
with open("train_triviaqa.yaml") as f:
    config = yaml.safe_load(f)

# Extract configuration values
model_config = config["model"]
dataset_config = config["dataset"]
training_config = config["training"]

# Load 4-bit quantized model from Unsloth
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_config["name"],
    max_seq_length=model_config["max_seq_length"],
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    load_in_4bit=model_config["quantization"]["load_in_4bit"],
)

# Set special tokens
tokenizer.pad_token = tokenizer.eos_token
if "special_tokens" in model_config:
    tokenizer.add_special_tokens(
        {"additional_special_tokens": model_config["special_tokens"]}
    )
    print(f"Added special tokens: {model_config['special_tokens']}")

# Load preprocessed dataset
print("Loading dataset...")
dataset = load_dataset(dataset_config["name"])
train_dataset = dataset[dataset_config.get("split", "train")]

# Prepare training arguments
training_args = TrainingArguments(
    output_dir=training_config["output_dir"],
    per_device_train_batch_size=training_config["batch_size"],
    gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
    warmup_ratio=training_config["warmup_ratio"],
    logging_steps=training_config["logging_steps"],
    save_steps=training_config["save_steps"],
    learning_rate=training_config["learning_rate"],
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    weight_decay=training_config["weight_decay"],
    lr_scheduler_type=training_config["lr_scheduler"],
    num_train_epochs=training_config["num_train_epochs"],
    max_steps=training_config["max_steps"],
    report_to="none",
    remove_unused_columns=training_config["remove_unused_columns"],
)

# Create trainer
print("Creating trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field=dataset_config["text_field"],
    max_seq_length=model_config["max_seq_length"],
    args=training_args,
    neftune_noise_alpha=training_config["neftune_noise_alpha"],
    packing=training_config["packing"],
)

# Start training
print("Starting training...")
trainer.train()

# Save final model
output_dir = training_config["output_dir"]
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
