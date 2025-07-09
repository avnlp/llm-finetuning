# Code taken from: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/HuggingFace%20Course-Gemma3_(1B)-GRPO.ipynb
## Dataset preprocessing for GSM8K is imported from the data_preprocessing.py file, and reward functions are imported from the rewards.py file.

from unsloth import FastLanguageModel
import torch
import re
from datasets import load_dataset, Dataset
from unsloth import is_bfloat16_supported
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer
from rewards import (
    extract_xml_answer,
    correctness_reward_func,
    int_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
    xmlcount_reward_func,
    count_xml,
)

# Global parameters
max_seq_length = 1024
lora_rank = 32
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,
)

# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


# ================ DATASET PREPROCESSING ================
# Helper functions
def extract_hash_answer(text: str) -> str | None:
    return text.split("####")[1].strip() if "####" in text else None


# Dataset preparation
def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return data


# Load and process dataset
dataset = get_gsm8k_questions()


# ================ TRAINING CONFIGURATION ================
# Calculate max prompt length
tokenized_lengths = dataset.map(
    lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
    batched=False,
)["tokens"]
max_prompt_length = max(len(tokens) for tokens in tokenized_lengths)

# Training arguments
training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    gradient_accumulation_steps=1,
    num_generations=6,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
)

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

# Start training
print("Starting GRPO training...")
trainer.train()


# ================ INFERENCE ================
# Inference function
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
        temperature=0.8,
        top_p=0.95,
        max_tokens=256,
    )

    lora = model.load_lora("grpo_saved_lora") if use_lora else None
    output = (
        model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=lora,
        )[0]
        .outputs[0]
        .text
    )

    print(f"\n{'='*50}\nQuestion: {question}\n{'='*50}")
    print("Response:\n", output)
    return output


# Test without LoRA
generate_response("Calculate pi.", use_lora=False)

# Test with LoRA
generate_response("Calculate pi.", use_lora=True)


# ================ SAVING ================
# Save LoRA adapters
model.save_lora("grpo_saved_lora")
print("\nSaved LoRA adapters to 'grpo_saved_lora' directory")

# Save merged models (conditional flags)
SAVE_MERGED_16BIT = False
if SAVE_MERGED_16BIT:
    model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
    print("Saved merged 16-bit model")

SAVE_MERGED_4BIT = False
if SAVE_MERGED_4BIT:
    model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")
    print("Saved merged 4-bit model")

# Save GGUF format (conditional flags)
SAVE_GGUF = False
if SAVE_GGUF:
    model.save_pretrained_gguf("model", tokenizer)
    print("Saved GGUF model (Q8_0)")

SAVE_GGUF_F16 = False
if SAVE_GGUF_F16:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
    print("Saved GGUF model (F16)")

SAVE_GGUF_Q4 = False
if SAVE_GGUF_Q4:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
    print("Saved GGUF model (Q4_K_M)")

print("\nAll saving operations completed!")
