# Code taken from: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/HuggingFace%20Course-Gemma3_(1B)-GRPO.ipynb
## Dataset preprocessing for GSM8K is imported from the data_preprocessing.py file, and reward functions are imported from the rewards.py file.

# Modified to use YAML configuration file
import yaml
from data_preprocessing import format_gsm8k_dataset
from datasets import load_dataset
from rewards import (
    correctness_reward_func,
    int_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    xmlcount_reward_func,
)
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from vllm import SamplingParams


# Load configuration from YAML file
with open("llama3_1_8b_train.yaml") as f:
    config = yaml.safe_load(f)

# Extract configuration values
model_config = config["model"]
dataset_config = config["dataset"]
training_config = config["training"]["config"]
generation_config = config["training"]["generation"]
reward_config = config["rewards"]
inference_config = config["inference"]
saving_config = config["saving"]
output_config = config["output"]

# Set system prompt and answer extraction
SYSTEM_PROMPT = dataset_config["preprocessing"]["system_prompt"]
ANSWER_EXTRACTION = dataset_config["preprocessing"]["answer_extraction"]

# Load model with configuration
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_config["name"],
    max_seq_length=model_config["max_seq_length"],
    load_in_4bit=model_config["quantization"]["load_in_4bit"],
    fast_inference=True,
    max_lora_rank=model_config["lora"]["r"],
    gpu_memory_utilization=0.6,
)

# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=model_config["lora"]["r"],
    target_modules=model_config["lora"]["target_modules"],
    lora_alpha=model_config["lora"]["alpha"],
    use_gradient_checkpointing=model_config["lora"]["use_gradient_checkpointing"],
    random_state=3407,
)


# ================ DATASET PREPROCESSING ================
def get_gsm8k_questions(split=dataset_config["split"]) -> Dataset:
    data = load_dataset(dataset_config["name"], dataset_config["config"])[split]
    return format_gsm8k_dataset(data, SYSTEM_PROMPT)


# Load and process dataset
dataset = get_gsm8k_questions()

# Calculate max prompt length
tokenized_lengths = dataset.map(
    lambda x: {
        "tokens": tokenizer.apply_chat_template(
            x["prompt"], add_generation_prompt=True, tokenize=True
        )
    },
    batched=False,
)["tokens"]
max_prompt_length = max(len(tokens) for tokens in tokenized_lengths)

# Override generation config with calculated values
if generation_config["max_prompt_length"] == "auto":
    generation_config["max_prompt_length"] = max_prompt_length
if generation_config["max_completion_length"] == "auto":
    generation_config["max_completion_length"] = (
        model_config["max_seq_length"] - max_prompt_length
    )

# Map reward functions from names to actual functions
REWARD_FUNCTIONS = {
    "xmlcount_reward_func": xmlcount_reward_func,
    "soft_format_reward_func": soft_format_reward_func,
    "strict_format_reward_func": strict_format_reward_func,
    "int_reward_func": int_reward_func,
    "correctness_reward_func": correctness_reward_func,
}
reward_funcs = [REWARD_FUNCTIONS[name] for name in reward_config["functions"]]

# ================ TRAINING CONFIGURATION ================
training_args = GRPOConfig(
    learning_rate=training_config["learning_rate"],
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=training_config["weight_decay"],
    warmup_ratio=training_config["warmup_ratio"],
    lr_scheduler_type=training_config["lr_scheduler"],
    optim=training_config["optim"],
    logging_steps=training_config["logging_steps"],
    per_device_train_batch_size=training_config["batch_size"],
    bf16=is_bfloat16_supported() and training_config.get("bf16", False),
    fp16=not is_bfloat16_supported() and training_config.get("fp16", True),
    gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
    num_generations=generation_config["num_generations"],
    max_prompt_length=generation_config["max_prompt_length"],
    max_completion_length=generation_config["max_completion_length"],
    max_steps=training_config["max_steps"],
    save_steps=training_config["save_steps"],
    max_grad_norm=training_config["max_grad_norm"],
    report_to=training_config["report_to"],
    output_dir=output_config["dir"],
)

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=dataset,
)

# Start training
print("Starting GRPO training with configuration:")
print(f"Model: {model_config['name']}")
print(f"Dataset: {dataset_config['name']} ({dataset_config['split']} split)")
print(f"Batch size: {training_config['batch_size']}")
print(f"Learning rate: {training_config['learning_rate']}")
print(f"Reward functions: {', '.join(reward_config['functions'])}")
print(f"Max steps: {training_config['max_steps']}")
print(f"Output directory: {output_config['dir']}")

trainer.train()
print("Training completed!")


# ================ INFERENCE ================
def generate_response(question, use_lora=False):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    sampling_params = SamplingParams(
        temperature=inference_config["sampling_params"]["temperature"],
        top_p=inference_config["sampling_params"]["top_p"],
        max_tokens=inference_config["sampling_params"]["max_tokens"],
    )

    lora = model.load_lora(saving_config["lora_dir"]) if use_lora else None
    output = (
        model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=lora,
        )[0]
        .outputs[0]
        .text
    )

    print(f"\n{'=' * 50}\nQuestion: {question}\n{'=' * 50}")
    print("Response:\n", output)
    return output


# Test without LoRA
generate_response("Calculate pi.", use_lora=False)

# Test with LoRA
generate_response("Calculate pi.", use_lora=True)

# ================ SAVING ================
# Save LoRA adapters
model.save_lora(saving_config["lora_dir"])
print(f"\nSaved LoRA adapters to '{saving_config['lora_dir']}' directory")

# Save merged models based on configuration
if saving_config["merged_16bit"]:
    model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
    print("Saved merged 16-bit model")

if saving_config["merged_4bit"]:
    model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")
    print("Saved merged 4-bit model")

# Save GGUF formats
if saving_config["gguf"]:
    model.save_pretrained_gguf("model", tokenizer)
    print("Saved GGUF model (Q8_0)")

if saving_config["gguf_f16"]:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
    print("Saved GGUF model (F16)")

if saving_config["gguf_q4"]:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
    print("Saved GGUF model (Q4_K_M)")

print("\nAll saving operations completed!")
