import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import ORPOTrainer, ORPOConfig

# Load formatted dataset for ORPO
# The dataset should contain prompts, responses, and rewards
dataset = load_dataset(args.dataset_name, split=args.dataset_split)

# Model ID for the pretrained model
model_id = "/kaggle/input/llama-3.1/transformers/8b-instruct/2"

# BitsAndBytesConfig for quantization (int-4 configuration)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=bnb_config, use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# LoRA Configuration (used for parameter-efficient fine-tuning)
peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# ORPO Configurations
orpo_config = ORPOConfig(
    batch_size=16,  # Batch size for each ORPO update
    forward_batch_size=4,  # Number of examples processed per forward pass
    learning_rate=1.41e-5,  # Learning rate for ORPO optimization
    log_with="wandb",  # Log training metrics with Weights and Biases
    wandb_project="llama-3-orpo",  # WandB project name
    mini_batch_size=8,  # Mini-batch size for optimization
    epochs=3,  # Number of training epochs
    kl_coeff=0.1,  # KL divergence coefficient
    ref_mean=None,  # Reference model mean for reward normalization
    ref_std=None,  # Reference model std for reward normalization
)

# Initialize ORPOTrainer
trainer = ORPOTrainer(
    model=model,
    ref_model=model,  # Reference model for KL divergence calculation
    tokenizer=tokenizer,
    config=orpo_config,
    dataset=dataset,  # Dataset containing prompts, responses, and rewards
    peft_config=peft_config,
)

# Start ORPO Training
trainer.train()

# Save the trained model
trainer.save_model(output_dir="llama-3-8b-orpo")

# Load Trained Model
peft_model = AutoPeftModelForCausalLM.from_pretrained(
    "llama-3-8b-orpo",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# Merge LoRA with the base model and save the merged model
merged_model = model.merge_and_unload()
merged_model.save_pretrained("llama-3-8b-orpo", safe_serialization=True, max_shard_size="2GB")
