# Medical Question Answering

GRPO + QLoRA training on biomedical QA datasets. The model learns to reason through clinical and biomedical questions using structured `<reasoning>` and `<answer>` tags, evaluated by 8 reward functions including LLM-as-judge correctness metrics.

## What this module trains

Biomedical and clinical question answering across three dataset types: multiple-choice (MedQA), free-text biomedical (BioASQ), and research article QA (PubMedQA). GRPO with LLM-as-judge reward functions trains the model to produce accurate, well-structured medical answers.

## Prerequisites

**`OPENAI_API_KEY` is required.** The four correctness reward functions call an LLM judge via DeepEval and Evidently. Set the key before running any pipeline in this module:

```bash
export OPENAI_API_KEY="your-key"
```

## Expected output format

The model is trained to produce responses in this structure:

```
<reasoning>
The question asks about the mechanism of action of metformin.
Metformin is a biguanide that primarily works by inhibiting hepatic gluconeogenesis.
It activates AMPK, which reduces glucose production in the liver.
It also improves peripheral insulin sensitivity and reduces intestinal glucose absorption.
</reasoning>
<answer>Metformin inhibits hepatic gluconeogenesis by activating AMPK, reducing liver glucose output.</answer>
```

The correctness reward functions evaluate the `<answer>` content against the ground truth using LLM judges. The format reward functions enforce structured, appropriately-lengthed responses.

## How GRPO training works

```
MedQALoader(config=loader_config).load(config["dataset_split"])
    → {"prompt": list[dict], "answer": str}
    → FastLanguageModel.from_pretrained(load_in_4bit=True)
    → FastLanguageModel.get_peft_model(model, r=lora_r, lora_alpha=...)  # apply QLoRA
    → GRPOTrainer(reward_funcs=[...])      # 8 reward functions injected here
    → trainer.train()
```

At each training step:
1. GRPO samples a batch of prompts and generates `num_generations=4` completions per prompt
2. Each of the 8 reward functions scores all completions (correctness functions make LLM API calls)
3. Scores are normalised within each group of 4 to compute relative advantages
4. Policy is updated to increase the probability of higher-advantage completions

## Datasets

The values shown are defaults from each loader's `CONFIG`. All three loaders support YAML dataset overrides via `dataset_id`, `dataset_subset`, and `dataset_split`.

| Dataset | Default HF ID | Default Config | Default Split | Size | Notes |
|---------|---------------|----------------|---------------|------|-------|
| MedQA | `bigbio/med_qa` | `med_qa_en_4options_bigbio_qa` | `train` | ~10.2k | Multiple-choice; choices formatted as `A. text\nB. text\n...` |
| BioASQ | `enelpol/rag-mini-bioasq` | `question-answer-passages` | `train` | 4,010 | Free-text biomedical QA; `relevant_passage_ids` not used in training |
| PubMedQA | `qiaojin/PubMedQA` | `pqa_artificial` | `train` | 211k | Research QA; `long_answer` used as target, `final_decision` not used |

> **BioASQ note**: The canonical dataset (`bigbio/bioasq_task_b`) requires manual registration at participants-area.bioasq.org. This module uses `enelpol/rag-mini-bioasq` as a publicly available substitute.

## Reward functions

Three of the four correctness functions (`DeepEvalSummarizationReward`, `DeepEvalAnswerRelevancyReward`, `EvidentlyCorrectnessLLMReward`) are generic and shared across modules. `DeepEvalGEvalRAGReward` is the only domain-adapted function — its GEval criteria string reads "correctly addresses the **medical** question".

**Correctness** (`reward_functions/correctness/`):

| Function | Score range | What it measures |
|----------|-------------|-----------------|
| `DeepEvalGEvalRAGReward` | 0.0–1.0 | LLM-as-judge via DeepEval GEval with medical question framing |
| `DeepEvalSummarizationReward` | 0.0–1.0 | DeepEval SummarizationMetric: faithfulness of the response |
| `DeepEvalAnswerRelevancyReward` | 0.0–1.0 | DeepEval AnswerRelevancyMetric: relevance of the answer to the question |
| `EvidentlyCorrectnessLLMReward` | 0.0 or 1.0 | Evidently binary judge: correct (1.0) or incorrect (0.0) |

**Format** (`reward_functions/format/`):

| Function | Score range | What it rewards |
|----------|-------------|----------------|
| `ReasoningTagsReward` | 0.0 or 1.0 | Presence of a complete `<reasoning>...</reasoning>` block (binary) |
| `MultilineComplianceReward` | 0.0–1.0 | At least 3 non-empty lines; partial credit below threshold |
| `StructureValidationReward` | 0.0–1.0 | Both `<reasoning>` and `<answer>` blocks present; 0.5 each |
| `ResponseFormatReward` | 0.0–1.0 | Response length between 50–2000 characters; scaled penalty outside range |

See [`docs/reward_functions.md`](../../../docs/reward_functions.md) for full scoring logic.

## How to run

```bash
python src/llm_finetuning/medical_question_answering/medqa/train.py
python src/llm_finetuning/medical_question_answering/bioasq/train.py
python src/llm_finetuning/medical_question_answering/pubmedqa/train.py
```

## Config reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_id` | str | `"unsloth/Llama-3.2-3B-Instruct"` | HuggingFace model ID (Unsloth variant) |
| `max_seq_length` | int | `2048` | Maximum sequence length |
| `output_dir` | str | — | Where to save the trained model |
| `dataset_split` | str | `"train"` | Dataset split to load |
| `learning_rate` | float | `5.0e-6` | Learning rate |
| `per_device_train_batch_size` | int | `1` | Batch size per GPU |
| `gradient_accumulation_steps` | int | `4` | Steps before optimizer update |
| `num_generations` | int | `4` | Completions generated per prompt (GRPO group size) |
| `max_prompt_length` | int | `512` | Max tokens in prompt |
| `max_completion_length` | int | `512` | Max tokens in completion |
| `num_train_epochs` | int | `1` | Training epochs |
| `logging_steps` | int | `1` | Log every N steps |
| `lora_r` | int | `8` | QLoRA rank |
| `lora_alpha` | int | `8` | QLoRA scaling factor |
| `target_modules` | list[str] | `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` | Modules to apply QLoRA to |
| `dataset_id` | str \| null | — | Optional HuggingFace dataset ID override. Falls back to loader `CONFIG.dataset_id` if omitted. |
| `dataset_subset` | str \| null | — | Optional dataset config/subset override. Falls back to loader `CONFIG.subset` if omitted. |

## Models

Default base model: `unsloth/Llama-3.2-3B-Instruct`.

To use a different model, copy `config.yaml` from an existing pipeline (e.g. `medqa/config.yaml`), update `model_id` and `output_dir`, and run `train.py`.

---

## Adding a new medical QA pipeline

Medical QA pipelines sit directly under the module root (e.g. `medqa/`, `bioasq/`,
`pubmedqa/`) — there is no `grpo/` subdirectory as in multi-hop QA.

**Example: add a pipeline for a new dataset called `newdataset`**

### 1. Create the directory

```bash
mkdir -p src/llm_finetuning/medical_question_answering/newdataset
touch src/llm_finetuning/medical_question_answering/newdataset/__init__.py
```

### 2. Write `data_processing.py`

Model the loader on `MedQALoader`. Each pipeline has its own `data_processing.py` —
there is no shared base loader file in this module.

```python
# src/llm_finetuning/medical_question_answering/newdataset/data_processing.py

from __future__ import annotations

from typing import Any

from llm_finetuning.core import BaseDatasetLoader, DatasetConfig


class NewDatasetLoader(BaseDatasetLoader):
    CONFIG = DatasetConfig(dataset_id="your-hf-id", subset=None)
    SYSTEM_PROMPT = (
        "You are a medical expert. Reason through the clinical question step by step "
        "inside <reasoning> tags. Provide your final answer inside <answer> tags."
    )

    def __init__(self, config: DatasetConfig | None = None) -> None:
        super().__init__(config or self.CONFIG)

    def format_example(self, example: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["answer"],
        }
```

### 3. Write `train.py`

Copy `medqa/train.py` and update:

- The loader import: `from llm_finetuning.medical_question_answering.newdataset.data_processing import NewDatasetLoader`
- The dataset line to use the override-aware pattern:

```python
loader_config = DatasetConfig(
    dataset_id=config.get("dataset_id", NewDatasetLoader.CONFIG.dataset_id),
    subset=config.get("dataset_subset", NewDatasetLoader.CONFIG.subset),
)
dataset = NewDatasetLoader(config=loader_config).load(config["dataset_split"])
```

- The `CONFIG_PATH`: already points to `Path(__file__).parent / "config.yaml"`

### 4. Write `config.yaml`

Copy `medqa/config.yaml` and update `model_id`, `output_dir`, and `dataset_split`. Optionally add:

```yaml
dataset_id: "your-hf-id"       # optional override
dataset_subset: null            # optional override
dataset_split: "train"
```

> **Note**: `OPENAI_API_KEY` must be set before running — all four correctness reward
> functions make LLM API calls.

### 5. Run

```bash
export OPENAI_API_KEY="your-key"
python src/llm_finetuning/medical_question_answering/newdataset/train.py
```
