import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer

# Define dataset parameters (replace with your dataset details)
DATASET_NAME = "your_dataset_name_here"  # Replace with the actual dataset name
DATASET_SPLIT = "train"  # Replace with the desired split (e.g., "train", "test", "validation")

# Load formatted dataset for DPO
# The dataset should contain preference pairs or rankings
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

# LoRA Configuration (still using LoRA as part of model)
peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules=["all-linear"],  # Updated to a list as expected by some configurations
    task_type="CAUSAL_LM",
)

# TrainingArguments specific to DPO
training_args = TrainingArguments(
    output_dir="llama-3-8b-dpo",  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=1,  # Batch size per device
    gradient_accumulation_steps=2,  # Steps before backward/update pass
    gradient_checkpointing=True,  # Save memory with gradient checkpointing
    optim="adamw_torch_fused",  # Fused AdamW optimizer
    logging_steps=10,  # Log every 10 steps
    save_strategy="epoch",  # Save checkpoint after each epoch
    learning_rate=1e-4,  # Learning rate for DPO
    bf16=True,  # Use bfloat16 precision
    max_grad_norm=0.3,  # Max gradient norm
    warmup_ratio=0.03,  # Warmup ratio
    lr_scheduler_type="constant",  # Constant learning rate
    push_to_hub=False,  # Set to False if not pushing to the Hub
    report_to="wandb",  # Report metrics to Weights and Biases
)

# Initialize DPOTrainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # The dataset must contain preference pairs
    peft_config=peft_config,
    tokenizer=tokenizer,
    beta=0.1,  # DPO beta parameter for preference weighting
)

# Start DPO Training
trainer.train()

# Save Model
trainer.save_model()

# Load Trained Model
peft_model = AutoPeftModelForCausalLM.from_pretrained(
    training_args.output_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# Merge LoRA with the base model and save the merged model
merged_model = model.merge_and_unload()
merged_model.save_pretrained(training_args.output_dir, safe_serialization=True, max_shard_size="2GB")
