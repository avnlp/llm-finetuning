# ---------------- train_llama3_musique.py ----------------
import os
import re
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments
from trl import GRPOTrainer, GRPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from data_processing import preprocess_musique
from rewards import compute_musique_reward

model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("uva-irlab/musique", "full", split="train")
dataset = dataset.map(preprocess_musique, batched=True, remove_columns=dataset.column_names)


def reward_func(prompts, completions, answers, group, **kwargs):
    return [compute_musique_reward(pred, g) for pred, g in zip(completions, group)]


model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

grpo_config = GRPOConfig(
    batch_size=2,
    mini_batch_size=1,
    learning_rate=1.41e-5,
    adap_kl_ctrl=True,
    init_kl_coef=0.2,
    group_reward_mode="relative",
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.1,
)

training_args = TrainingArguments(
    output_dir="./grpo_musique",
    per_device_train_batch_size=grpo_config.batch_size,
    remove_unused_columns=False,
    num_train_epochs=1,
    logging_steps=1,
    save_steps=10,
    fp16=True,
)

grpo_trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    config=grpo_config,
    dataset=dataset,
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 100,
    "temperature": 0.7,
}

for epoch, batch in enumerate(grpo_trainer.dataloader):
    queries = batch["query"]
    answers = batch["answer"]
    groups = batch["group"]

    inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(
        grpo_trainer.accelerator.device
    )

    response_tensors = grpo_trainer.generate(
        inputs.input_ids, attention_mask=inputs.attention_mask, **generation_kwargs
    )
    completions = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    reward_list = reward_func(prompts=queries, completions=completions, answers=answers, group=groups)
    rewards = [torch.tensor(r, device=grpo_trainer.accelerator.device) for r in reward_list]

    stats = grpo_trainer.step(response_tensors, rewards, groups)
    print(f"Step {epoch} | AvgReward: {sum(reward_list)/len(reward_list):.4f}")

model.save_pretrained("./grpo_musique_final")
tokenizer.save_pretrained("./grpo_musique_final")
