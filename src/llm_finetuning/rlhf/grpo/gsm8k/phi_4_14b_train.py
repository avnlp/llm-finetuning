"""Phi-4-14B GRPO training script for GSM8K dataset."""
# type: ignore[import-not-found, misc]
# Code taken from:
# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4_(14B)-GRPO.ipynb

from typing import Any

from data_preprocessing import (  # type: ignore[import-not-found, misc]
    format_gsm8k_dataset,  # type: ignore[import-not-found, misc]
)
from datasets import load_dataset
from rewards import (  # type: ignore[import-not-found, misc]
    correctness_reward_func,
    int_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    xmlcount_reward_func,
)
from trl import GRPOConfig, GRPOTrainer
from unsloth import (  # type: ignore[import-untyped]
    FastLanguageModel,
    is_bfloat16_supported,
)
from vllm import SamplingParams  # type: ignore[import-not-found, misc]


# Configuration
max_seq_length = 512
lora_rank = 16

# Load base model (Phi-4 14B)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-4",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.7,
)

# Apply LoRA configuration
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Dataset Preprocessing
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def get_gsm8k_questions(split: str = "train") -> Any:
    """Load and preprocess GSM8K dataset."""
    data = load_dataset("openai/gsm8k", "main")[split]
    return format_gsm8k_dataset(data, SYSTEM_PROMPT)


# Load training dataset
dataset = get_gsm8k_questions()


# Training Configuration
training_args = GRPOConfig(
    use_vllm=True,
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
    max_prompt_length=256,
    max_completion_length=200,
    max_steps=100,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
)

# Initialize GRPO trainer
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


# =====================
# 6. TRAINING EXECUTION
# =====================
print("Starting GRPO training...")
trainer.train()
print("Training completed!")


# =====================
# 7. INFERENCE DEMONSTRATION
# =====================

# Untrained model example
print("\n=== Untrained Model Output ===")
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Which is bigger? 9.11 or 9.9?"}],
    tokenize=False,
    add_generation_prompt=True,
)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)
output = model.fast_generate([text], sampling_params=sampling_params)[0].outputs[0].text
print(output)

# Save trained LoRA weights
model.save_lora("phi4_grpo_lora")

# Trained model example
print("\n=== Trained Model Output ===")
text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Which is bigger? 9.11 or 9.9?"},
    ],
    tokenize=False,
    add_generation_prompt=True,
)
output = (
    model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=model.load_lora("phi4_grpo_lora"),
    )[0]
    .outputs[0]
    .text
)
print(output)


# =====================
# 8. MODEL SAVING & EXPORT
# =====================
# Save merged 16-bit model
model.save_pretrained_merged("phi4_grpo_model", tokenizer, save_method="merged_16bit")

# Save GGUF format (quantized)
model.save_pretrained_gguf(
    "phi4_gguf",
    tokenizer,
    quantization_method=["q4_k_m", "q8_0"],
)

print("\nModel saved successfully!")
