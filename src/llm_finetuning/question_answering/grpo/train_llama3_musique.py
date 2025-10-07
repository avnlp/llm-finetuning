import torch
import yaml
from data_processing import preprocess_musique
from datasets import load_dataset
from rewards import compute_musique_reward
from transformers import AutoTokenizer, TrainingArguments
from trl import AutoModelForCausalLMWithValueHead, GRPOConfig, GRPOTrainer


# Load configuration
with open("train_llama_3_musique.yaml") as f:
    config = yaml.safe_load(f)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset(
    config["dataset"]["name"],
    config["dataset"]["config"],
    split=config["dataset"]["split"],
).map(preprocess_musique, batched=True, remove_columns=dataset.column_names)


def reward_func(prompts, completions, answers, group, **kwargs):
    return [compute_musique_reward(pred, g) for pred, g in zip(completions, group)]


# Initialize models
model = AutoModelForCausalLMWithValueHead.from_pretrained(config["model"]["name"])
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config["model"]["name"])

# GRPO Configuration
grpo_config = GRPOConfig(
    batch_size=config["grpo"]["batch_size"],
    mini_batch_size=config["grpo"]["mini_batch_size"],
    learning_rate=config["grpo"]["learning_rate"],
    adap_kl_ctrl=config["grpo"]["adap_kl_ctrl"],
    init_kl_coef=config["grpo"]["init_kl_coef"],
    group_reward_mode=config["grpo"]["group_reward_mode"],
    cliprange=config["grpo"]["cliprange"],
    cliprange_value=config["grpo"]["cliprange_value"],
    vf_coef=config["grpo"]["vf_coef"],
)

# Training arguments
training_args = TrainingArguments(
    output_dir=config["training"]["output_dir"],
    per_device_train_batch_size=grpo_config.batch_size,
    remove_unused_columns=False,
    num_train_epochs=config["training"]["num_train_epochs"],
    logging_steps=config["training"]["logging_steps"],
    save_steps=config["training"]["save_steps"],
    fp16=config["training"]["fp16"],
)

# Initialize GRPO Trainer
grpo_trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    config=grpo_config,
    dataset=dataset,
)

# Generation config
generation_kwargs = {
    "min_length": config["generation"]["min_length"],
    "top_k": config["generation"]["top_k"],
    "top_p": config["generation"]["top_p"],
    "do_sample": config["generation"]["do_sample"],
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": config["generation"]["max_new_tokens"],
    "temperature": config["generation"]["temperature"],
}

# Training loop
for epoch, batch in enumerate(grpo_trainer.dataloader):
    queries = batch["query"]
    answers = batch["answer"]
    groups = batch["group"]

    inputs = tokenizer(
        queries, return_tensors="pt", padding=True, truncation=True, max_length=8192
    ).to(grpo_trainer.accelerator.device)

    response_tensors = grpo_trainer.generate(
        inputs.input_ids, attention_mask=inputs.attention_mask, **generation_kwargs
    )
    completions = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    reward_list = reward_func(
        prompts=queries, completions=completions, answers=answers, group=groups
    )
    rewards = [
        torch.tensor(r, device=grpo_trainer.accelerator.device) for r in reward_list
    ]

    stats = grpo_trainer.step(response_tensors, rewards, groups)
    print(f"Step {epoch} | AvgReward: {sum(reward_list) / len(reward_list):.4f}")

# Save final model
model.save_pretrained(config["training"]["output_dir"] + "_final")
tokenizer.save_pretrained(config["training"]["output_dir"] + "_final")
