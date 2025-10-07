# The code is taken from https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-ORPO.ipynb
# The code has been rewritten to work with the UltraFeedback dataset.

import numpy as np
from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer


max_seq_length = 4096
dtype = None
load_in_4bit = True

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Define prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token


# New formatting function for UltraFeedback
def format_ultra_feedback(sample):
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

    # Handle cases where all ratings are equal
    if ratings[best_idx] == ratings[worst_idx]:
        return None

    return {
        "prompt": alpaca_prompt.format(instruction, "", ""),
        "chosen": ratings_responses[best_idx][1] + EOS_TOKEN,
        "rejected": ratings_responses[worst_idx][1] + EOS_TOKEN,
    }


# Load and format dataset
dataset = load_dataset("openbmb/UltraFeedback", split="train")
dataset = dataset.map(format_ultra_feedback, remove_columns=dataset.column_names)
dataset = dataset.filter(lambda x: x is not None)  # Remove skipped samples

# Enable reward modeling stats
PatchDPOTrainer()

# Train the model
orpo_trainer = ORPOTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=ORPOConfig(
        max_length=max_seq_length,
        max_prompt_length=max_seq_length // 2,
        max_completion_length=max_seq_length // 2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        beta=0.1,
        logging_steps=1,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        num_train_epochs=1,  # Full training run
        output_dir="outputs",
        report_to="none",
    ),
)

orpo_trainer.train()
