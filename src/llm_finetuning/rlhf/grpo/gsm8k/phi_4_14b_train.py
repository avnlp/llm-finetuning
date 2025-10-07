# Code taken from: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4_(14B)-GRPO.ipynb
# Dataset preprocessing for GSM8K is imported from the data_preprocessing.py file, and reward functions are imported from the rewards.py file.
import re

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
    r=lora_rank,  # LoRA rank
    target_modules=["gate_proj", "up_proj", "down_proj"],  # Target modules
    lora_alpha=lora_rank,  # LoRA alpha
    use_gradient_checkpointing="unsloth",  # Enable for long-context training
    random_state=3407,  # Random seed for reproducibility
)

# Dataset Preprocessing
# System prompt to enforce structured reasoning
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


# Helper functions for answer extraction


# Dataset loader for GSM8K
from data_preprocessing import extract_xml_answer, format_gsm8k_dataset


def get_gsm8k_questions(split="train"):
    """Loads and preprocesses GSM8K dataset using shared preprocessing."""
    data = load_dataset("openai/gsm8k", "main")[split]
    return format_gsm8k_dataset(data, SYSTEM_PROMPT)


# Load training dataset
dataset = get_gsm8k_questions()


# =====================
# 4. REWARD FUNCTIONS
# =====================
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Rewards correct final answers (2.0 points)."""
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """Rewards integer answers (0.5 points)."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Strict XML format validation (0.5 points)."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Lenient XML format validation (0.5 points)."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Detailed XML structure scoring (variable points)."""

    def count_xml(text) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            count -= len(text.split("\n</answer>\n")[-1]) * 0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# Training Configuration
# Training hyperparameters
training_args = GRPOConfig(
    use_vllm=True,  # Use vLLM for accelerated generation
    learning_rate=5e-6,  # Lower learning rate for stable fine-tuning
    adam_beta1=0.9,  # Adam optimizer parameters
    adam_beta2=0.99,
    weight_decay=0.1,  # Weight regularization
    warmup_ratio=0.1,  # Warmup schedule
    lr_scheduler_type="cosine",  # Learning rate schedule
    optim="paged_adamw_8bit",  # Optimizer with 8-bit precision
    logging_steps=1,  # Log metrics every step
    per_device_train_batch_size=1,  # Batch size per device
    bf16=is_bfloat16_supported(),  # Use bfloat16 if supported
    fp16=not is_bfloat16_supported(),  # Fallback to float16
    gradient_accumulation_steps=1,  # Increase for smoother training
    num_generations=6,  # Number of generations per prompt
    max_prompt_length=256,  # Maximum prompt length
    max_completion_length=200,  # Maximum generation length
    max_steps=100,  # Training steps (increase for full training)
    save_steps=250,  # Save checkpoint interval
    max_grad_norm=0.1,  # Gradient clipping
    report_to="none",  # Disable external logging
    output_dir="outputs",  # Output directory
)

# Initialize GRPO trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,  # XML structure quality
        soft_format_reward_func,  # Basic format compliance
        strict_format_reward_func,  # Strict format compliance
        int_reward_func,  # Integer answer format
        correctness_reward_func,  # Final answer correctness
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
    quantization_method=["q4_k_m", "q8_0"],  # Multiple quantization options
)

# Save to Hugging Face Hub (requires authentication)
# model.push_to_hub_merged("your-username/phi4-grpo", tokenizer, token="hf_...")

print("\nModel saved successfully!")
