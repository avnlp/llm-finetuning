# Code taken from: https://colab.research.google.com/drive/1MRgGtLWuZX4ypSfGguFgC-IblTvO2ivM?usp=sharing
# The code has been rewritten to work with the UltraFeedback dataset.

from unsloth import FastLanguageModel, PatchDPOTrainer
import torch
import numpy as np
from datasets import load_dataset, Dataset
from trl import KTOConfig, KTOTrainer
from transformers import TextStreamer

# Apply KTO patch first
PatchDPOTrainer()

# Model configuration
max_seq_length = 4096
dtype = None
load_in_4bit = True

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add proper chat template
if tokenizer.chat_template is None:
    DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

# Configure LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


# New function to process UltraFeedback for KTO
def process_ultra_feedback(example):
    instruction = example["instruction"]
    completions = example["completions"]

    # Create KTO examples
    kto_examples = []
    for comp in completions:
        try:
            # Extract helpfulness rating
            rating = float(comp["annotations"]["helpfulness"][0]["Rating"])
            response = comp["response"]

            # Create messages structure
            messages = [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]

            # Apply chat template
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            # Split into prompt and completion
            assistant_token = "<|assistant|>\n"
            assistant_index = full_text.rfind(assistant_token)

            if assistant_index != -1:
                prompt = full_text[: assistant_index + len(assistant_token)]
                completion = full_text[assistant_index + len(assistant_token) :]

                # Create KTO example
                kto_examples.append(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "label": rating >= 4,  # True for high-quality (rating 4-5), False for low-quality (1-3)
                    }
                )
        except (KeyError, IndexError, ValueError) as e:
            print(f"Skipping completion due to error: {str(e)}")
            continue

    return kto_examples


# Load and process dataset
print("Loading UltraFeedback dataset...")
dataset = load_dataset("openbmb/UltraFeedback", split="train")
print(f"Loaded {len(dataset)} examples")

# Process dataset and create KTO examples
print("Processing dataset for KTO training...")
kto_data = []
for example in dataset.select(range(1000)):
    kto_data.extend(process_ultra_feedback(example))

print(f"Created {len(kto_data)} KTO examples")
train_dataset = Dataset.from_list(kto_data)

# Set up KTO trainer
kto_trainer = KTOTrainer(
    model=model,
    args=KTOConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=5e-7,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        output_dir="outputs",
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        seed=42,
        report_to="none",
    ),
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Show memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Train the model
print("Starting training...")
kto_trainer.train()

# Save the model
model.save_pretrained("kto_model")
tokenizer.save_pretrained("kto_model")
print("Model saved to 'kto_model' directory")


# Inference function
def generate_response(message):
    messages = [{"role": "user", "content": message}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(
        "cuda"
    )

    text_streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    outputs = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        temperature=0.7,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Test questions
test_questions = [
    "What are the best things to do in Cairo with family?",
    "Explain quantum computing in simple terms",
    "How do I set up a Python development environment?",
    "What's the difference between AI and machine learning?",
]

# Generate responses
for question in test_questions:
    print("\n" + "=" * 50)
    print(f"QUESTION: {question}")
    print("-" * 50 + "\nRESPONSE:")
    generate_response(question)
