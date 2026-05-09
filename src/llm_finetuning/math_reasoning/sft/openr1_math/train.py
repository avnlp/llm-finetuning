"""SFT format-priming on OpenR1-Math for Qwen-3 Base.

Stage 1 of a two-stage pipeline. Teaches the base model to follow
<reasoning>/<answer> output format using verified reasoning traces from
OpenR1-Math-220k. The output checkpoint is consumed by Stage 2 (GRPO on
GSM8K) via its config_qwen3.yaml model_id field.

Usage::

    python src/llm_finetuning/math_reasoning/sft/openr1_math/train.py
"""

from pathlib import Path

import yaml
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel  # type: ignore[import-untyped]

from llm_finetuning.core import DatasetConfig
from llm_finetuning.math_reasoning.sft.openr1_math.data_processing import (
    OpenR1MathSFTLoader,
)


CONFIG_PATH = Path(__file__).parent / "config.yaml"


def main() -> None:
    """Run SFT format-priming on OpenR1-Math."""
    with open(CONFIG_PATH) as f:
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
        dataset_id=config.get("dataset_id", OpenR1MathSFTLoader.CONFIG.dataset_id),
        subset=config.get("dataset_subset", OpenR1MathSFTLoader.CONFIG.subset),
    )
    dataset = OpenR1MathSFTLoader(
        tokenizer=tokenizer,
        config=loader_config,
        max_example_tokens=config.get("max_example_tokens", 1024),
        max_samples=config.get("max_samples"),
        require_reasoning_complete=config.get("require_reasoning_complete", True),
    ).load(config.get("dataset_split", "train"))

    training_args = SFTConfig(
        output_dir=config["output_dir"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config.get("num_train_epochs", 2),
        logging_steps=config.get("logging_steps", 5),
        save_strategy="epoch",
        report_to="none",
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()
