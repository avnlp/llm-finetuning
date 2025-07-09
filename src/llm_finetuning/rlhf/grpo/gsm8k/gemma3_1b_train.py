# Code taken from https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(1B)-GRPO.ipynb
# Dataset preprocessing for GSM8K is imported from the data_preprocessing.py file, and reward functions are imported from the rewards.py file.

from unsloth import FastModel
import torch
from datasets import load_dataset
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from transformers import TextStreamer
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

max_seq_length = 1024
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-1b-it",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = format_gsm8k_dataset(dataset, system_prompt)
tokenized_lengths = get_tokenized_lengths(dataset, tokenizer)
max_prompt_length = get_max_prompt_length(tokenized_lengths)

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    logging_steps=1,
    per_device_train_batch_size=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    gradient_accumulation_steps=1,
    num_generations=4,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=50,
    save_steps=50,
    max_grad_norm=0.1,
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

print("Starting GRPO training...")
trainer.train()
