import torch
import time
import re
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments
from trl import GRPOTrainer, GRPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from evidently.metrics import BinaryClassificationPromptTemplate, LLMEvaluator
import openai
import os

# Set API keys
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY_HERE"
os.environ["EVIDENTLY_API_KEY"] = (
    "dG9rbgFghvBbZL5Dcap6z03hM0BA9WSqCNyRfEg1BSkcLxIpbABQj+cplAvj/eqbKyAKdnqGEfy9jeUEEULac3fWzNHBoMB2FTCLcttnVirpMkorxtGgEG2f20LZL2Tqstr3mPScgT2Dl8S7B4hdz/YkrhpROFJ1eT0W"
)
openai.api_key = os.getenv("GROQ_API_KEY")
openai.base_url = "https://api.groq.com/openai/v1"

# Initialize DeepEval metric with Groq
relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model="groq/llama3-70b-8192", include_reason=False)

# Initialize Evidently Correctness Evaluator
correctness_prompt = BinaryClassificationPromptTemplate(
    criteria="""An ANSWER is correct when it matches the REFERENCE in all factual details;
it is incorrect if it contradicts, omits, or adds unsupported information.

REFERENCE:
=====
{target_response}
=====""",
    target_category="incorrect",
    non_target_category="correct",
    uncertainty="unknown",
    include_reasoning=True,
    pre_messages=[("system", "You are an expert evaluator. You will be given an ANSWER and a REFERENCE.")],
)

correctness_evaluator = LLMEvaluator(prompts=[correctness_prompt], openai_params={"model": "llama3-70b-8192"})


def reward_func(prompts, completions, answers, **kwargs) -> list[float]:
    """Compute rewards using DeepEval and Evidently metrics"""
    # Batch correctness evaluation
    test_df = pd.DataFrame({"prediction": completions, "target": answers})
    correctness_result = correctness_evaluator.evaluate(test_df, prompt_column="prediction", reference_column="target")

    rewards = []
    for idx, (prompt, completion, answer) in enumerate(zip(prompts, completions, answers)):
        try:
            # Extract context and question
            context_match = re.search(r"Context: (.*?)(?:\nQuestion:|\n|$)", prompt, re.DOTALL)
            question_match = re.search(r"Question: (.*?)(?:\n|$)", prompt, re.DOTALL)
            context = context_match.group(1).strip() if context_match else ""
            question = question_match.group(1).strip() if question_match else ""

            # Compute Answer Relevancy
            relevancy_test = LLMTestCase(input=question, actual_output=completion)
            relevancy_metric.measure(relevancy_test)
            relevancy_score = relevancy_metric.score

            # Extract correctness score
            correctness_label = correctness_result.iloc[idx]["correctness"].label
            correctness_score = 1.0 if correctness_label == "correct" else 0.0

            # Answer match (fallback)
            answer_match = 1.0 if answer.lower() in completion.lower() else 0.3

            # Combine scores (40% relevancy, 40% correctness, 20% answer match)
            reward = 0.4 * relevancy_score + 0.4 * correctness_score + 0.2 * answer_match
            rewards.append(reward)

        except Exception as e:
            print(f"Error computing reward: {str(e)}")
            rewards.append(0.5)

    return rewards


# Configuration
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# Preprocessing function for FreshQA
def preprocess_freshqa(examples):
    inputs = []
    answers = []
    groups = []

    for i in range(len(examples["question"])):
        context = examples["context"][i] or ""
        question = examples["question"][i]
        answer = examples["answer"][i]
        reasoning_level = examples["reasoning_level"][i]
        temporal_category = examples["temporal_category"][i]

        group_id = {"Memory": 0, "Simple Reasoning": 1, "Complex Reasoning": 2}.get(reasoning_level, 0)

        group_id += {"Pre-2023": 0, "2023": 3, "2024": 6}.get(temporal_category, 0)

        inputs.append(f"Context: {context}\nQuestion: {question}")
        answers.append(answer)
        groups.append(group_id)

    return {"query": inputs, "answer": answers, "group": groups}


# Load and preprocess dataset
dataset = load_dataset("freshqa/freshqa", split="train")
dataset = dataset.map(preprocess_freshqa, batched=True, remove_columns=dataset.column_names)

# Initialize models
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# GRPO Configuration
grpo_config = GRPOConfig(
    batch_size=2,
    mini_batch_size=1,
    learning_rate=1.41e-5,
    adap_kl_ctrl=True,
    init_kl_coef=0.2,
    group_reward_mode="relative",
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.1,
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./grpo_freshqa",
    per_device_train_batch_size=grpo_config.batch_size,
    remove_unused_columns=False,
    num_train_epochs=1,
    logging_steps=1,
    save_steps=10,
    fp16=True,
)

# Initialize GRPO Trainer
grpo_trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    config=grpo_config,
    dataset=dataset,
)

# Generation config
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 150,
    "temperature": 0.7,
}

# Training loop
for epoch, batch in enumerate(grpo_trainer.dataloader):
    if epoch >= 20:
        break

    queries = batch["query"]
    answers = batch["answer"]
    groups = batch["group"]

    # Tokenize queries
    inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(
        grpo_trainer.accelerator.device
    )

    # Generate responses
    start_time = time.time()
    response_tensors = grpo_trainer.generate(
        inputs.input_ids, attention_mask=inputs.attention_mask, **generation_kwargs
    )
    completions = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    gen_time = time.time() - start_time

    # Compute rewards
    start_reward = time.time()
    reward_list = reward_func(prompts=queries, completions=completions, answers=answers)
    reward_time = time.time() - start_reward

    rewards = [torch.tensor(r, device=grpo_trainer.accelerator.device) for r in reward_list]

    # Run GRPO step
    stats = grpo_trainer.step(response_tensors, rewards, groups)

    # Log progress
    avg_reward = sum(reward_list) / len(reward_list)
    print(f"Step {epoch} | Avg Reward: {avg_reward:.4f}")

# Save final model
model.save_pretrained("./grpo_freshqa_final")
tokenizer.save_pretrained("./grpo_freshqa_final")
