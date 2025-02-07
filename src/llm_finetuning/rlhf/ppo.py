import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig

# Define dataset parameters
DATASET_NAME = "your_dataset_name_here"  # Replace with the actual dataset name
DATASET_SPLIT = "train"  # Replace with the desired split (e.g., "train", "test", "validation")

# Load formatted dataset for PPO
dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

# Model ID for the pretrained model
MODEL_ID = "/kaggle/input/llama-3.1/transformers/8b-instruct/2"

# BitsAndBytesConfig for quantization (int-4 configuration)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    use_cache=False,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# LoRA Configuration (used for parameter-efficient fine-tuning)
peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules=["all-linear"],  # Ensure the expected data type is a list
    task_type="CAUSAL_LM",
)

# PPO Configurations
ppo_config = PPOConfig(
    batch_size=16,  # Batch size for each PPO update
    forward_batch_size=4,  # Number of examples processed per forward pass
    learning_rate=1.41e-5,  # Learning rate for PPO optimization
    log_with="wandb",  # Log training metrics with Weights and Biases
    wandb_project="llama-3-ppo",  # WandB project name
    mini_batch_size=8,  # Mini-batch size for optimization
    epochs=3,  # Number of training epochs
)

# Initialize PPOTrainer
trainer = PPOTrainer(
    model=model,
    ref_model=model,  # Reference model for KL divergence calculation
    tokenizer=tokenizer,
    config=ppo_config,
    dataset=dataset,  # Dataset containing prompts and optionally responses
    peft_config=peft_config,
)


# Custom reward function (Example: Reward based on specific criteria)
def reward_function(response_texts):
    """
    Define your custom reward function here.
    Example: Reward responses that contain specific keywords.
    """
    rewards = []
    for response in response_texts:
        reward = 1.0 if "desired_keyword" in response else -1.0
        rewards.append(reward)
    return rewards


# Attach custom reward function to the trainer (if needed)
trainer.reward_function = reward_function

# Train the model using the built-in train method
trainer.train()

# Save the trained model
OUTPUT_DIR = "llama-3-8b-ppo"
trainer.save_model(output_dir=OUTPUT_DIR)

# Load Trained Model
peft_model = AutoPeftModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# Merge LoRA with the base model and save the merged model
merged_model = model.merge_and_unload()
merged_model.save_pretrained(OUTPUT_DIR, safe_serialization=True, max_shard_size="2GB")
