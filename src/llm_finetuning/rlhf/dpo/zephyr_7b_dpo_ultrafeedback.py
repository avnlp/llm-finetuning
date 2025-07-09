# The code is taken from https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb
# The code has been rewritten to work with the UltraFeedback dataset.

from unsloth import FastLanguageModel, PatchDPOTrainer
import torch
import numpy as np
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from transformers import TrainingArguments, TextStreamer

# Apply DPO patch first
PatchDPOTrainer()

# Model configuration
max_seq_length = 4096
dtype = None
load_in_4bit = True

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/zephyr-sft-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


# New function to process UltraFeedback dataset
def process_ultra_feedback(sample):
    instruction = sample["instruction"]
    completions = sample["completions"]

    # Extract ratings and responses
    ratings_responses = []
    for comp in completions:
        helpfulness = comp["annotations"]["helpfulness"][0]
        rating = float(helpfulness["Rating"])
        ratings_responses.append((rating, comp["response"]))

    # Find best and worst responses
    ratings = [r[0] for r in ratings_responses]
    best_idx = np.argmax(ratings)
    worst_idx = np.argmin(ratings)

    return {
        "instruction": instruction,
        "chosen": ratings_responses[best_idx][1],
        "rejected": ratings_responses[worst_idx][1],
    }


# Load and process dataset
dataset = load_dataset("openbmb/UltraFeedback", split="train")
dataset = dataset.map(process_ultra_feedback, remove_columns=dataset.column_names)

# Sample 0.5% of the dataset (similar to original)
dataset = dataset.train_test_split(test_size=0.995, seed=42)["train"]


# Apply Zephyr chat template
def apply_zephyr_template(example):
    # Create message dictionaries
    system_message = {"role": "system", "content": ""}
    user_message = {"role": "user", "content": example["instruction"]}
    chosen_message = {"role": "assistant", "content": example["chosen"]}
    rejected_message = {"role": "assistant", "content": example["rejected"]}

    # Format using Zephyr template
    example["text_prompt"] = tokenizer.apply_chat_template(
        [system_message, user_message], tokenize=False, add_generation_prompt=True
    )
    example["text_chosen"] = tokenizer.apply_chat_template([chosen_message], tokenize=False)
    example["text_rejected"] = tokenizer.apply_chat_template([rejected_message], tokenize=False)

    # Remove assistant prefix for DPO
    assistant_prefix = "<|assistant|>\n"
    example["text_chosen"] = example["text_chosen"].replace(assistant_prefix, "", 1)
    example["text_rejected"] = example["text_rejected"].replace(assistant_prefix, "", 1)

    return example


# Apply formatting
dataset = dataset.map(apply_zephyr_template, remove_columns=["instruction", "chosen", "rejected"])

# Rename columns for DPO Trainer
dataset = dataset.rename_columns({"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"})

# Train DPO model
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=DPOConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        num_train_epochs=3,
        learning_rate=5e-6,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
        report_to="none",
    ),
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_length=1024,
    max_prompt_length=512,
)

dpo_trainer.train()

# Inference example
FastLanguageModel.for_inference(model)
prompt = "Explain quantum computing in simple terms"
inputs = tokenizer([f"<|system|>\n</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"], return_tensors="pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256)
