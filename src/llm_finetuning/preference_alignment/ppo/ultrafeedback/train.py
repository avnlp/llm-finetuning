"""PPO training script for preference alignment on UltraFeedback."""

from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from llm_finetuning.core import DatasetConfig
from llm_finetuning.preference_alignment.ppo.ultrafeedback.data_processing import (
    UltraFeedbackPPOLoader,
)
from llm_finetuning.preference_alignment.reward_models.pointwise_reward_model import (
    PointwiseRewardModel,
)


CONFIG_PATH = Path(__file__).parent / "config.yaml"


def main() -> None:
    """Load config, initialize model, and run PPO training on UltraFeedback."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config["model_id"],
        load_in_4bit=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_model = PointwiseRewardModel()
    loader_config = DatasetConfig(
        dataset_id=config.get("dataset_id", UltraFeedbackPPOLoader.CONFIG.dataset_id),
        subset=config.get("dataset_subset", UltraFeedbackPPOLoader.CONFIG.subset),
    )
    loader = UltraFeedbackPPOLoader(config=loader_config)
    dataset = loader.load(config.get("dataset_split", "train_prefs"))
    dataset = loader.tokenize(dataset, tokenizer)

    trainer = PPOTrainer(
        args=PPOConfig(
            output_dir=config["output_dir"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            mini_batch_size=config["mini_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            ppo_epochs=config.get("ppo_epochs", 4),
            logging_steps=1,
            report_to="none",
        ),
        model=model,
        processing_class=tokenizer,
        dataset=dataset,
    )

    generation_kwargs = {
        "max_new_tokens": config.get("max_new_tokens", 128),
        "do_sample": True,
        "temperature": 0.7,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for batch in trainer.dataloader:
        query_tensors = batch["input_ids"]
        response_tensors = trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = tokenizer.batch_decode(
            response_tensors, skip_special_tokens=True
        )

        prompts = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
        rewards = [
            torch.tensor(reward_model.score(p, r))
            for p, r in zip(prompts, batch["response"])
        ]

        trainer.step(list(query_tensors), list(response_tensors), rewards)

    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()
