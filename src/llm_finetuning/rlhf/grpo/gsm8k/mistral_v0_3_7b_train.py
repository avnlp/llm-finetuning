# This implementaion is based on the codefrom: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-GRPO.ipynb
# Dataset preprocessing for GSM8K is imported from the data_preprocessing.py file, and reward functions are imported from the rewards.py file.

from unsloth import FastLanguageModel
import torch
import re
from datasets import load_dataset, Dataset
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from rewards import (
    extract_xml_answer,
    correctness_reward_func_mistral as correctness_reward_func,
    int_reward_func_mistral as int_reward_func,
    strict_format_reward_func_mistral as strict_format_reward_func,
    soft_format_reward_func_mistral as soft_format_reward_func,
    xmlcount_reward_func_mistral as xmlcount_reward_func,
)

# Mistral model configuration
max_seq_length = 1024
lora_rank = 32

# Load Mistral model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",  # Mistral model
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,
)

# Configure Mistral with LoRA
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


# ==========================================
# 2. Dataset Preprocessing
# ==========================================
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


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


dataset = get_gsm8k_questions()


# ==========================================
# 4. Training Configuration
# ==========================================
max_prompt_length = 256

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


# ==========================================
# 5. Training Execution
# ==========================================
trainer.train()


# ==========================================
# 6. Inference with Mistral
# ==========================================
# Pre-training inference
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Calculate pi."}],
    tokenize=False,
    add_generation_prompt=True,
)

from vllm import SamplingParams

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
)
output = (
    model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=None,
    )[0]
    .outputs[0]
    .text
)
print(output)

# Post-training inference
model.save_lora("grpo_saved_lora")

text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Calculate pi."},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

output = (
    model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=model.load_lora("grpo_saved_lora"),
    )[0]
    .outputs[0]
    .text
)
print(output)


# ==========================================
# 7. Saving Mistral Model
# ==========================================
# Save to 16bit
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")

# Save to 4bit
model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")

# Save LoRA adapters
model.save_pretrained("model")
tokenizer.save_pretrained("model")

# Save to GGUF/llama.cpp format
model.save_pretrained_gguf("model", tokenizer)  # 8bit Q8_0
model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")  # 16bit
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")  # 4bit
