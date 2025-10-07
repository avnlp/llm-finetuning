import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TextStreamer, pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from unsloth import FastLanguageModel


# Model configuration
max_seq_length = 4096
dtype = None
load_in_4bit = True

# Load base model and tokenizer
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
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Wrap model for PPO with value head
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

# Load reward model
reward_tokenizer = AutoTokenizer.from_pretrained(
    "OpenAssistant/reward-model-deberta-v3-large-v2"
)
reward_model = pipeline(
    "text-classification",
    model="OpenAssistant/reward-model-deberta-v3-large-v2",
    device=model.pretrained_model.device,
)


# New function to process UltraFeedback dataset for PPO
def process_ultra_feedback_for_ppo(sample):
    instruction = sample["instruction"]
    return {"prompt": instruction}


# Load and process dataset
dataset = load_dataset("openbmb/UltraFeedback", split="train")
dataset = dataset.map(
    process_ultra_feedback_for_ppo, remove_columns=dataset.column_names
)

# Sample 0.5% of the dataset
dataset = dataset.train_test_split(test_size=0.995, seed=42)["train"]


# Apply Zephyr chat template to prompts
def format_prompt(example):
    system_message = {"role": "system", "content": ""}
    user_message = {"role": "user", "content": example["prompt"]}

    formatted_prompt = tokenizer.apply_chat_template(
        [system_message, user_message], tokenize=False, add_generation_prompt=True
    )
    return {"prompt": formatted_prompt}


dataset = dataset.map(format_prompt)


# Reward function using the reward model
def reward_function(samples, responses, **kwargs):
    rewards = []
    for prompt, response in zip(samples, responses):
        # Combine prompt and response
        text = prompt + response

        # Get reward score (convert to float)
        reward_output = reward_model(text, truncation=True, max_length=1024)[0]
        score = reward_output["score"]

        # Convert to reward: positive if label is "1" (good), negative otherwise
        if reward_output["label"] == "1":
            rewards.append(torch.tensor(score))
        else:
            rewards.append(torch.tensor(-score))

    return rewards


# PPO Training Configuration
ppo_config = PPOConfig(
    batch_size=2,
    mini_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1.41e-5,
    optimize_cuda_cache=True,
    log_with="tensorboard",
)

# Initialize PPOTrainer
ppo_trainer = PPOTrainer(
    model=model,
    config=ppo_config,
    dataset=dataset,
    tokenizer=tokenizer,
)

# Training loop
for _epoch in range(3):
    for batch in ppo_trainer.dataloader:
        prompts = batch["prompt"]

        # Generate responses
        response_tensors = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_tensors.append(response[len(prompt) :])  # Remove prompt

        # Compute rewards
        rewards = reward_function(prompts, response_tensors)

        # Train with PPO
        stats = ppo_trainer.step(prompts, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

# Save the trained model
model.save_pretrained("zephyr-ppo-ultrafeedback")

# Inference example
FastLanguageModel.for_inference(model)
prompt = "Explain quantum computing in simple terms"
inputs = tokenizer(
    [f"<|system|>\n</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"], return_tensors="pt"
).to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256)
