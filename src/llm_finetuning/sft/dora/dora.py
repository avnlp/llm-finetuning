import argparse
import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, LoraRuntimeConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from llm_finetuning.sft.data_preparation import Dataloader


def main(args):

    # Initialize the Dataloader class
    dataloader = Dataloader(
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        system_message_inputs=args.system_message_inputs,
        user_message_inputs=args.user_message_inputs,
    )

    # Load formatted dataset from Dataloader
    dataset = dataloader.dataset

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16 if args.use_bfloat16 else torch.float32,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.use_bfloat16 else torch.float32,
        quantization_config=bnb_config,
        use_cache=args.use_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optimizer,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        learning_rate=args.learning_rate,
        bf16=args.use_bfloat16,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        push_to_hub=args.push_to_hub,
        report_to=args.report_to,
    )

    # LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias=args.lora_bias,
        target_modules=args.target_modules.split(","),
        task_type="CAUSAL_LM",
        use_dora=args.use_dora,
        runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=args.dora_ephemeral_gpu_offload),
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()

    # Save model
    trainer.save_model()

    # Load Trained Model
    peft_model = AutoPeftModelForCausalLM.from_pretrained(
        args.output_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # Merge LoRA and base model, then save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="2GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training and fine-tuning a model using LoRA")

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to load.")
    parser.add_argument("--dataset_split", type=str, default="train", help="Split of the dataset to use.")

    # User-defined prompt configurations
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="System prompt.")
    parser.add_argument(
        "--user_prompt", type=str, default="Answer the following question: {question}", help="User prompt."
    )
    parser.add_argument("--system_message_inputs", type=dict, default={}, help="System message inputs.")
    parser.add_argument("--user_message_inputs", type=dict, default={}, help="User message inputs.")

    # Model arguments
    parser.add_argument("--model_id", type=str, required=True, help="Model ID or path to pre-trained model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")

    # Quantization arguments
    parser.add_argument("--load_in_4bit", type=bool, default=True, help="Load model in 4-bit precision.")
    parser.add_argument(
        "--bnb_4bit_use_double_quant", type=bool, default=True, help="Use double quantization for 4-bit."
    )
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="Quantization type for 4-bit.")
    parser.add_argument("--use_bfloat16", type=bool, default=True, help="Use bfloat16 precision.")
    parser.add_argument("--use_cache", type=bool, default=False, help="Use cache during model loading.")

    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=2, help="Number of steps for gradient accumulation."
    )
    parser.add_argument(
        "--gradient_checkpointing", type=bool, default=True, help="Use gradient checkpointing to save memory."
    )
    parser.add_argument("--optimizer", type=str, default="adamw_torch_fused", help="Optimizer to use.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Save strategy for checkpoints.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for training.")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Maximum gradient norm.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="Type of learning rate scheduler.")
    parser.add_argument("--push_to_hub", type=bool, default=True, help="Push model to the Hugging Face Hub.")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting tool for training metrics.")

    # LoRA arguments
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha value.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_bias", type=str, default="none", help="Bias configuration for LoRA.")
    parser.add_argument(
        "--target_modules", type=str, default="query_key_value", help="Comma-separated list of target modules for LoRA."
    )

    # DoRA arguments
    parser.add_argument("--use_dora", type=bool, default=True, help="Use Dora for training.")
    parser.add_argument("--dora_ephemeral_gpu_offload", type=bool, default=True, help="Use CPU offloading for Dora.")

    args = parser.parse_args()

    main(args)
