# Code taken from:
# Dataset preprocessing for GSM8K is imported from the data_preprocessing.py file, and reward functions are imported from the rewards.py file.
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
import re
from data_preprocessing import (
    format_gsm8k_dataset,
    get_tokenized_lengths,
    get_max_prompt_length,
)
from rewards import (
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers,
)

# ================ CONFIGURATION ================
max_seq_length = 2048
lora_rank = 64
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

# ================ MODEL SETUP ================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,
)

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
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ================ DATASET PREPROCESSING ================
dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = format_gsm8k_dataset(dataset, system_prompt)
tokenized_lengths = get_tokenized_lengths(dataset, tokenizer)
max_prompt_length = get_max_prompt_length(tokenized_lengths)

# ================ TRAINING CONFIGURATION ================
training_args = GRPOConfig(
    learning_rate=5e-6,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    gradient_accumulation_steps=4,
    num_generations=4,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=500,
    save_steps=250,
    max_grad_norm=1.0,
    report_to="none",
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    args=training_args,
    train_dataset=dataset,
)

# ================ TRAINING ================
print("Starting training...")
trainer.train()
