# Multi-Hop Question Answering

GRPO + QLoRA training on multi-hop QA datasets. The model learns to reason across multiple steps inside `<reasoning>` tags before providing a final answer.

## What this module trains

Multi-hop questions require combining information from multiple sources or reasoning steps. This module trains models to produce structured reasoning traces before answering, using GRPO with 8 reward functions: 4 correctness-based (LLM-as-judge) and 4 format-based.

## Prerequisites

**`OPENAI_API_KEY` is required.** The four correctness reward functions call an LLM judge via DeepEval and Evidently. Set the key before running any pipeline in this module:

```bash
export OPENAI_API_KEY="your-key"
```

## Expected output format

The model is trained to produce responses in this structure:

```
<reasoning>
The question asks who founded the company that acquired X.
First, I need to identify which company acquired X — that was Company Y, which acquired X in 2015.
Company Y was founded by Jane Smith in 1998.
</reasoning>
<answer>Jane Smith</answer>
```

The correctness reward functions compare the content of `<answer>` against the ground truth using LLM judges. The format reward functions push the model toward structured, multi-line reasoning traces.

## How GRPO training works

```
HotpotQALoader(config=loader_config).load(config["dataset_split"])
    → {"prompt": list[dict], "answer": str}
    → FastLanguageModel.from_pretrained(load_in_4bit=True)
    → FastLanguageModel.get_peft_model(r=lora_r, lora_alpha=...)  # apply QLoRA
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
| HotpotQA | `hotpotqa/hotpot_qa` | `distractor` | `train` | ~90k | Uses `distractor` config (not `fullwiki`) |
| FreshQA | `vtllms/sealqa` | `longseal` | `test` | 264 | Only `test` split available; all 264 rows used for training |
| MuSiQue | `dgslibisey/MuSiQue` | — | `train` | ~19.9k | Filtered to `answerable == True` rows only |

## Reward functions

**Correctness** (`reward_functions/correctness/`):

| Function | Score range | What it measures |
|----------|-------------|-----------------|
| `DeepEvalGEvalRAGReward` | 0.0–1.0 | LLM-as-judge via DeepEval GEval: factual accuracy, supporting evidence, completeness |
| `DeepEvalSummarizationReward` | 0.0–1.0 | DeepEval SummarizationMetric: coverage and faithfulness of the response against source content |
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

> The reward function implementations in this module are **copied, not imported**, from `medical_question_answering`. Changes here do not affect that module and vice versa.

## How to run

```bash
python src/llm_finetuning/multi_hop_question_answering/grpo/hotpotqa/train.py
python src/llm_finetuning/multi_hop_question_answering/grpo/freshqa/train.py
python src/llm_finetuning/multi_hop_question_answering/grpo/musique/train.py
```

## Config reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_id` | str | `"unsloth/Llama-3.2-3B-Instruct"` | HuggingFace model ID (Unsloth variant) |
| `max_seq_length` | int | `2048` | Maximum sequence length for model loading |
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

To use a different model, copy `config.yaml` from an existing pipeline, update `model_id` with an Unsloth-compatible model (e.g. `unsloth/Llama-3.1-8B-Instruct`), update `output_dir`, and run `train.py`.

---

## Adding a new GRPO pipeline

**Example: add a GRPO pipeline for a new multi-hop QA dataset called `newqa`**

### 1. Create the directory

```bash
mkdir -p src/llm_finetuning/multi_hop_question_answering/grpo/newqa
touch src/llm_finetuning/multi_hop_question_answering/grpo/newqa/__init__.py
```

### 2. Add a dataset loader class

Open `src/llm_finetuning/multi_hop_question_answering/data_processing.py` and add a
subclass of `MultiHopLoader` (which already defines `SYSTEM_PROMPT` and
`format_example`):

```python
from llm_finetuning.core import DatasetConfig


class NewQALoader(MultiHopLoader):
    CONFIG = DatasetConfig(dataset_id="your-hf-id")

    def __init__(self, config: DatasetConfig | None = None) -> None:
        super().__init__(config or self.CONFIG)
```

`MultiHopLoader.format_example` returns:

```python
{
    "prompt": [
        {"role": "system", "content": self.SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
    ],
    "answer": example["answer"],
}
```

If your dataset uses different field names, override `format_example` in `NewQALoader`.

### 3. Write `train.py`

Copy `grpo/hotpotqa/train.py` and replace the dataset import with `NewQALoader` and the
dataset line with the override-aware pattern:

```python
loader_config = DatasetConfig(
    dataset_id=config.get("dataset_id", NewQALoader.CONFIG.dataset_id),
    subset=config.get("dataset_subset", NewQALoader.CONFIG.subset),
)
dataset = NewQALoader(config=loader_config).load(config["dataset_split"])
```

### 4. Write `config.yaml`

Copy `grpo/hotpotqa/config.yaml` and update `output_dir` and `dataset_split`. Optionally add:

```yaml
dataset_id: "your-hf-id"       # optional override
dataset_subset: null            # optional override
dataset_split: "train"
```

### 5. Run

```bash
python src/llm_finetuning/multi_hop_question_answering/grpo/newqa/train.py
```
