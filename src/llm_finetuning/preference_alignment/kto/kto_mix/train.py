"""KTO training script for preference alignment on kto-mix-14k."""

from pathlib import Path

import yaml
from trl import KTOConfig, KTOTrainer
from unsloth import FastLanguageModel  # type: ignore[import-untyped]

from llm_finetuning.core import DatasetConfig
from llm_finetuning.preference_alignment.kto.kto_mix.data_processing import (
    KTOMixLoader,
)


CONFIG_PATH = Path(__file__).parent / "config.yaml"


def main() -> None:
    """Load config, initialize model, and run KTO training on kto-mix-14k."""
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
        dataset_id=config.get("dataset_id", KTOMixLoader.CONFIG.dataset_id),
        subset=config.get("dataset_subset", KTOMixLoader.CONFIG.subset),
    )
    dataset = KTOMixLoader(config=loader_config).load(
        config.get("dataset_split", "train")
    )

    trainer = KTOTrainer(
        model=model,
        args=KTOConfig(
            output_dir=config["output_dir"],
            learning_rate=config["learning_rate"],
            num_train_epochs=config["num_train_epochs"],
            per_device_train_batch_size=config["per_device_train_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            logging_steps=1,
            report_to="none",
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()
