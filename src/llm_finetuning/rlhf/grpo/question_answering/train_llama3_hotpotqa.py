# ---------------- train_llama3_hotpotqa.py ----------------

import os
import re
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments
from trl import GRPOTrainer, GRPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from data_processing import preprocess_hotpot
from rewards import compute_hotpot_reward

model_name = "groq/llama3-70b-8192"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = (
    load_dataset("hotpot_qa", "distractor", split="train")
    .select(range(50))
    .map(preprocess_hotpot, batched=True, remove_columns=dataset.column_names)
)


def reward_func(prompts, completions, answers, group, **kwargs):
    return [compute_hotpot_reward(pred, g) for pred, g in zip(completions, group)]


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
    output_dir="./grpo_hotpotqa",
    per_device_train_batch_size=grpo_config.batch_size,
    remove_unused_columns=False,
    num_train_epochs=1,
    logging_steps=1,
    save_steps=10,
)

grpo_trainer = GRPOTrainer(model=model, ref_model=ref_model, tokenizer=tokenizer, config=grpo_config, dataset=dataset)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 60,
}

for epoch, batch in enumerate(grpo_trainer.dataloader):
    if epoch >= 5:
        break

    queries, answers, groups = batch["query"], batch["answer"], batch["group"]

    inputs = tokenizer(queries, padding=True, truncation=True, return_tensors="pt", max_length=512).to(
        grpo_trainer.accelerator.device
    )

    out = grpo_trainer.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **generation_kwargs)
    completions = tokenizer.batch_decode(out, skip_special_tokens=True)

    reward_list = reward_func(prompts=queries, completions=completions, answers=answers, group=groups)
    rewards = [torch.tensor(r, device=grpo_trainer.accelerator.device) for r in reward_list]
    stats = grpo_trainer.step(out, rewards, groups)

    print(f"Step {epoch} | AvgReward: {sum(reward_list)/len(reward_list):.4f}")

model.save_pretrained("./grpo_hotpotqa_final")
tokenizer.save_pretrained("./grpo_hotpotqa_final")
