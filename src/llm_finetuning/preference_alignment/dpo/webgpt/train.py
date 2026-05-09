"""DPO training script for preference alignment on WebGPT comparisons."""

from pathlib import Path

import yaml
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer  # type: ignore[import-untyped]

from llm_finetuning.core import DatasetConfig
from llm_finetuning.preference_alignment.dpo.webgpt.data_processing import (
    WebGPTDPOLoader,
)


PatchDPOTrainer()

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def main() -> None:
    """Load config, initialize model, and run DPO training on WebGPT comparisons."""
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
        dataset_id=config.get("dataset_id", WebGPTDPOLoader.CONFIG.dataset_id),
        subset=config.get("dataset_subset", WebGPTDPOLoader.CONFIG.subset),
    )
    dataset = WebGPTDPOLoader(tokenizer, config=loader_config).load(
        config.get("dataset_split", "train")
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=DPOConfig(
            output_dir=config["output_dir"],
            learning_rate=config["learning_rate"],
            num_train_epochs=config["num_train_epochs"],
            per_device_train_batch_size=config["per_device_train_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            beta=config.get("dpo_beta", 0.1),
            max_length=config["max_seq_length"],
            max_prompt_length=config["max_seq_length"] // 2,
            logging_steps=1,
            report_to="none",
        ),
        beta=config.get("dpo_beta", 0.1),
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()
