"""Earnings Calls Prefix-Tuning supervised fine-tuning training script.

Fine-tunes a causal language model on the Earnings Calls dataset using
Prefix-Tuning via the PEFT library. A set of trainable prefix vectors are
prepended to the key and value tensors of every attention layer. All base
model parameters remain frozen.

Dataset:
    HuggingFace ID: lamini/earnings-calls-qa

Training pipeline:
    1. Loads hyperparameters from ``config.yaml`` (co-located with this script).
    2. Formats each example with ``apply_chat_template`` into a single ``text``
       field consumed by ``SFTTrainer``.
    3. Wraps the base model with Prefix-Tuning adapters.
    4. Trains with ``SFTTrainer`` and saves the adapter weights to ``output_dir``.

Usage::

    python src/llm_finetuning/supervised_finetuning/prefix_tuning/earnings_call/train.py
"""

from pathlib import Path

import torch
import yaml
from peft import PrefixTuningConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from llm_finetuning.core import DatasetConfig
from llm_finetuning.supervised_finetuning.loaders import EarningsCallLoader


CONFIG_PATH = Path(__file__).parent / "config.yaml"


def main() -> None:
    """Run the Earnings Calls Prefix-Tuning fine-tuning pipeline end-to-end.

    Reads all hyperparameters from the ``config.yaml`` file that lives
    alongside this script. The base model is loaded in bfloat16; prefix
    vectors are prepended to every attention layer. Prefix weights are
    saved to ``output_dir`` together with the tokenizer.
    """
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader_config = DatasetConfig(
        dataset_id=config.get("dataset_id", EarningsCallLoader.CONFIG.dataset_id),
        subset=config.get("dataset_subset", EarningsCallLoader.CONFIG.subset),
    )
    dataset = EarningsCallLoader(tokenizer, config=loader_config).load(config["split"])

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=config["num_virtual_tokens"],
    )
    model = get_peft_model(model, peft_config)

    training_args = SFTConfig(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        bf16=True,
        save_strategy=config.get("save_strategy", "epoch"),
        logging_steps=config.get("logging_steps", 10),
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
