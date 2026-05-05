"""GRPO training script for math reasoning on GSM8K."""

import argparse
from pathlib import Path

import yaml
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel  # type: ignore[import-untyped]

from llm_finetuning.core import DatasetConfig
from llm_finetuning.math_reasoning.data_processing import GSM8KLoader
from llm_finetuning.math_reasoning.reward_functions.correctness.answer_correctness import (
    AnswerCorrectnessReward,
)
from llm_finetuning.math_reasoning.reward_functions.format.multiline_compliance import (
    MultilineComplianceReward,
)
from llm_finetuning.math_reasoning.reward_functions.format.reasoning_tags import (
    ReasoningTagsReward,
)
from llm_finetuning.math_reasoning.reward_functions.format.response_structure import (
    ResponseStructureReward,
)
from llm_finetuning.math_reasoning.reward_functions.format.step_format import (
    StepFormatReward,
)


CONFIG_DIR = Path(__file__).parent


def main() -> None:
    """Parse config and run GRPO training on GSM8K."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Config filename (default: config.yaml)",
    )
    args = parser.parse_args()

    config_path = CONFIG_DIR / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_id"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    loader_config = DatasetConfig(
        dataset_id=config.get("dataset_id", GSM8KLoader.CONFIG.dataset_id),
        subset=config.get("dataset_subset", GSM8KLoader.CONFIG.subset),
    )
    dataset = GSM8KLoader(config=loader_config).load(
        config.get("dataset_split", "train")
    )

    training_args = GRPOConfig(
        output_dir=config["output_dir"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_generations=config.get("num_generations", 4),
        max_prompt_length=config.get("max_prompt_length", 256),
        max_completion_length=config.get("max_completion_length", 512),
        num_train_epochs=config.get("num_train_epochs", 1),
        max_grad_norm=config.get("max_grad_norm", 0.1),
        logging_steps=config.get("logging_steps", 1),
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            AnswerCorrectnessReward(),
            ReasoningTagsReward(),
            StepFormatReward(),
            MultilineComplianceReward(),
            ResponseStructureReward(),
        ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()
