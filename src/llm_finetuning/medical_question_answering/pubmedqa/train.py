"""PubMedQA GRPO fine-tuning training script."""

from pathlib import Path

import yaml
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel  # type: ignore[import-untyped]

from llm_finetuning.core import DatasetConfig
from llm_finetuning.medical_question_answering.pubmedqa.data_processing import (
    PubMedQALoader,
)
from llm_finetuning.medical_question_answering.reward_functions.correctness.deepeval_answer_relevancy import (
    DeepEvalAnswerRelevancyReward,
)
from llm_finetuning.medical_question_answering.reward_functions.correctness.deepeval_gevalrag import (
    DeepEvalGEvalRAGReward,
)
from llm_finetuning.medical_question_answering.reward_functions.correctness.deepeval_summarization import (
    DeepEvalSummarizationReward,
)
from llm_finetuning.medical_question_answering.reward_functions.correctness.evidently_correctness_llm import (
    EvidentlyCorrectnessLLMReward,
)
from llm_finetuning.medical_question_answering.reward_functions.format.multiline_compliance import (
    MultilineComplianceReward,
)
from llm_finetuning.medical_question_answering.reward_functions.format.reasoning_tags import (
    ReasoningTagsReward,
)
from llm_finetuning.medical_question_answering.reward_functions.format.response_format import (
    ResponseFormatReward,
)
from llm_finetuning.medical_question_answering.reward_functions.format.structure_validation import (
    StructureValidationReward,
)


CONFIG_PATH = Path(__file__).parent / "config.yaml"


def main() -> None:
    """Load config, initialise model, and run GRPO training on PubMedQA."""
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
        dataset_id=config.get("dataset_id", PubMedQALoader.CONFIG.dataset_id),
        subset=config.get("dataset_subset", PubMedQALoader.CONFIG.subset),
    )
    dataset = PubMedQALoader(config=loader_config).load(config["dataset_split"])

    training_args = GRPOConfig(
        output_dir=config["output_dir"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_generations=config.get("num_generations", 4),
        max_prompt_length=config.get("max_prompt_length", 512),
        max_completion_length=config.get("max_completion_length", 512),
        num_train_epochs=config.get("num_train_epochs", 1),
        logging_steps=config.get("logging_steps", 1),
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            DeepEvalGEvalRAGReward(),
            DeepEvalSummarizationReward(),
            DeepEvalAnswerRelevancyReward(),
            EvidentlyCorrectnessLLMReward(),
            ReasoningTagsReward(),
            MultilineComplianceReward(),
            StructureValidationReward(),
            ResponseFormatReward(),
        ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()
