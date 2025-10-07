import os
import re

import openai
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from evidently.metrics import BinaryClassificationPromptTemplate, LLMEvaluator
from transformers import AutoTokenizer, TrainingArguments
from trl import AutoModelForCausalLMWithValueHead, GRPOConfig, GRPOTrainer


# Load configuration
with open("train_llama_3_freshqa.yaml") as f:
    config = yaml.safe_load(f)

# Set API keys
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY_HERE"
os.environ["EVIDENTLY_API_KEY"] = "YOUR_EVIDENTLY_API_KEY_HERE"
openai.api_key = os.getenv("GROQ_API_KEY")
openai.base_url = "https://api.groq.com/openai/v1"

# Initialize DeepEval metric with Groq
relevancy_metric = AnswerRelevancyMetric(
    threshold=config["reward"]["relevancy"]["threshold"],
    model=config["reward"]["relevancy"]["model"],
    include_reason=False,
)

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
    pre_messages=[
        (
            "system",
            "You are an expert evaluator. You will be given an ANSWER and a REFERENCE.",
        )
    ],
)

correctness_evaluator = LLMEvaluator(
    prompts=[correctness_prompt],
    openai_params={"model": config["reward"]["correctness"]["model"]},
)


def reward_func(prompts, completions, answers, **kwargs) -> list[float]:
    # Batch correctness evaluation
    test_df = pd.DataFrame({"prediction": completions, "target": answers})
    correctness_result = correctness_evaluator.evaluate(
        test_df, prompt_column="prediction", reference_column="target"
    )

    rewards = []
    weights = config["reward"]["weights"]

    for idx, (prompt, completion, answer) in enumerate(
        zip(prompts, completions, answers)
    ):
        try:
            # Extract context and question
            context_match = re.search(
                r"Context: (.*?)(?:\nQuestion:|\n|$)", prompt, re.DOTALL
            )
            question_match = re.search(r"Question: (.*?)(?:\n|$)", prompt, re.DOTALL)
            context_match.group(1).strip() if context_match else ""
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

            # Combine scores
            reward = (
                weights["relevancy"] * relevancy_score
                + weights["correctness"] * correctness_score
                + weights["answer_match"] * answer_match
            )
            rewards.append(reward)

        except Exception as e:
            print(f"Error computing reward: {e!s}")
            rewards.append(0.5)

    return rewards


# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
tokenizer.pad_token = tokenizer.eos_token


# Preprocessing function for FreshQA
def preprocess_freshqa(examples):
    inputs, answers, groups = [], [], []
    for i in range(len(examples["question"])):
        context = examples["context"][i] or ""
        question = examples["question"][i]
        answer = examples["answer"][i]
        reasoning_level = examples["reasoning_level"][i]
        temporal_category = examples["temporal_category"][i]

        group_id = {"Memory": 0, "Simple Reasoning": 1, "Complex Reasoning": 2}.get(
            reasoning_level, 0
        )
        group_id += {"Pre-2023": 0, "2023": 3, "2024": 6}.get(temporal_category, 0)

        inputs.append(f"Context: {context}\nQuestion: {question}")
        answers.append(answer)
        groups.append(group_id)

    return {"query": inputs, "answer": answers, "group": groups}


# Load and preprocess dataset
dataset = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"]).map(
    preprocess_freshqa, batched=True, remove_columns=dataset.column_names
)

# Initialize models
model = AutoModelForCausalLMWithValueHead.from_pretrained(config["model"]["name"])
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config["model"]["name"])

# GRPO Configuration
grpo_config = GRPOConfig(
    batch_size=config["grpo"]["batch_size"],
    mini_batch_size=config["grpo"]["mini_batch_size"],
    learning_rate=config["grpo"]["learning_rate"],
    adap_kl_ctrl=config["grpo"]["adap_kl_ctrl"],
    init_kl_coef=config["grpo"]["init_kl_coef"],
    group_reward_mode=config["grpo"]["group_reward_mode"],
    cliprange=config["grpo"]["cliprange"],
    cliprange_value=config["grpo"]["cliprange_value"],
    vf_coef=config["grpo"]["vf_coef"],
)

# Training arguments
training_args = TrainingArguments(
    output_dir=config["training"]["output_dir"],
    per_device_train_batch_size=grpo_config.batch_size,
    remove_unused_columns=False,
    num_train_epochs=config["training"]["num_train_epochs"],
    logging_steps=config["training"]["logging_steps"],
    save_steps=config["training"]["save_steps"],
    fp16=config["training"]["fp16"],
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
    "min_length": config["generation"]["min_length"],
    "top_k": config["generation"]["top_k"],
    "top_p": config["generation"]["top_p"],
    "do_sample": config["generation"]["do_sample"],
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": config["generation"]["max_new_tokens"],
    "temperature": config["generation"]["temperature"],
}

# Training loop
for epoch, batch in enumerate(grpo_trainer.dataloader):
    if epoch >= config["training"]["max_steps"]:
        break

    queries = batch["query"]
    answers = batch["answer"]
    groups = batch["group"]

    # Tokenize queries
    inputs = tokenizer(
        queries, return_tensors="pt", padding=True, truncation=True, max_length=8192
    ).to(grpo_trainer.accelerator.device)

    # Generate responses
    response_tensors = grpo_trainer.generate(
        inputs.input_ids, attention_mask=inputs.attention_mask, **generation_kwargs
    )
    completions = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute rewards
    reward_list = reward_func(prompts=queries, completions=completions, answers=answers)
    rewards = [
        torch.tensor(r, device=grpo_trainer.accelerator.device) for r in reward_list
    ]

    # Run GRPO step
    stats = grpo_trainer.step(response_tensors, rewards, groups)

    # Log progress
    avg_reward = sum(reward_list) / len(reward_list)
    print(f"Step {epoch} | Avg Reward: {avg_reward:.4f}")

# Save final model
model.save_pretrained(config["training"]["output_dir"] + "_final")
tokenizer.save_pretrained(config["training"]["output_dir"] + "_final")
