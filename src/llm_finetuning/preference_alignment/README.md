# Preference Alignment

Six pipelines for aligning model outputs with human preferences: DPO, KTO, ORPO, and PPO. All use QLoRA except PPO, which requires a value head via `AutoModelForCausalLMWithValueHead`.

## What this module trains

Preference alignment trains models to prefer human-preferred responses over rejected ones. Different methods encode this preference signal differently — DPO/ORPO use paired comparisons, KTO uses binary labels, and PPO uses a reward model at training time.

## Datasets

The values shown are defaults from each loader's `CONFIG`. All preference alignment pipelines support YAML dataset overrides via `dataset_id`, `dataset_subset`, and `dataset_split`.

| Dataset | Default HF ID | Default Config | Default Split | Size | Used by |
|---------|---------------|----------------|---------------|------|---------|
| UltraFeedback Binarized | `HuggingFaceH4/ultrafeedback_binarized` | — | `train_prefs` | ~60k | DPO, ORPO, PPO |
| KTO Mix | `trl-lib/kto-mix-14k` | — | `train` | 14k | KTO |
| WebGPT Comparisons | `openai/webgpt_comparisons` | — | `train` | ~19k | DPO, PPO |

## Techniques

| Method | Trainer | Reference model | Key notes |
|--------|---------|----------------|-----------|
| **DPO** | `DPOTrainer` | None (implicit via `ref_model=None`) | Requires `PatchDPOTrainer()` from Unsloth before trainer import |
| **KTO** | `KTOTrainer` | None | Dataset already in KTO format (`prompt/completion/label`); no preprocessing |
| **ORPO** | `ORPOTrainer` | None (combined loss) | Requires `PatchDPOTrainer()` from Unsloth; no separate reference model |
| **PPO** | `PPOTrainer` | `AutoModelForCausalLMWithValueHead` | `PointwiseRewardModel` (`OpenAssistant/reward-model-deberta-v3-large-v2`) scores completions at training time; manual rollout loop |

## How training works

### DPO, ORPO, KTO — trainer-driven

These three methods are fully trainer-driven: format the dataset, pass it to the trainer, and call `trainer.train()`. The preference signal is encoded in the data columns.

```python
loader_config = DatasetConfig(
    dataset_id=config.get("dataset_id", UltraFeedbackDPOLoader.CONFIG.dataset_id),
    subset=config.get("dataset_subset", UltraFeedbackDPOLoader.CONFIG.subset),
)
dataset = UltraFeedbackDPOLoader(tokenizer, config=loader_config).load(
    config.get("dataset_split", "train_prefs")
)  # returns prompt/chosen/rejected
trainer = DPOTrainer(model=model, train_dataset=dataset, ...)
trainer.train()
```

KTO and PPO use the same dataset-override pattern, with their own loader defaults.

Data format per method:

| Method | Required columns | How the signal is encoded |
|--------|-----------------|--------------------------|
| DPO | `prompt`, `chosen`, `rejected` | Paired comparison: maximise P(chosen) / P(rejected) |
| ORPO | `prompt`, `chosen`, `rejected` | Combined SFT + odds-ratio preference loss |
| KTO | `prompt`, `completion`, `label` (bool) | Binary: True = desirable, False = undesirable |

DPO and ORPO use `apply_chat_template` on the chosen/rejected lists during preprocessing. KTO Mix is already in the right format — no preprocessing needed.

> **DPO/ORPO note**: `PatchDPOTrainer()` must be called at module level before importing `DPOTrainer` or `ORPOTrainer`. This is an Unsloth compatibility patch and is already in all train scripts.

### PPO — manual rollout loop

PPO differs from the others in two important ways:

**1. Model loading**: PPO requires a value head (the critic network that estimates expected return):

```python
# PPO only — not FastLanguageModel
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id, load_in_4bit=True)
```

**2. Manual training loop**: There is no `trainer.train()` call. The loop runs manually:

```python
for batch in dataloader:
    # Generate responses
    response_tensors = trainer.generate(query_tensors, max_new_tokens=config["max_new_tokens"])

    # Score with OpenAssistant pointwise reward model
    rewards = []
    for prompt, response in zip(decoded_queries, decoded_responses):
        score = reward_model.score(prompt, response)
        rewards.append(torch.tensor(score))

    # PPO update
    stats = trainer.step(query_tensors, response_tensors, rewards)
```

## Reward model

**PointwiseRewardModel** (`reward_models/pointwise_reward_model.py`): wraps `OpenAssistant/reward-model-deberta-v3-large-v2`, a DeBERTa-v3 model trained on human preference data to produce a scalar quality score for a single (prompt, response) pair.

`PointwiseRewardModel.score(prompt, response)`:
1. Tokenises the prompt + response as a pair
2. Runs `AutoModelForSequenceClassification` forward pass
3. Returns `logits[0].item()` — a scalar reward (higher = better)

The reward model is downloaded automatically from HuggingFace on first run.

## How to run

```bash
# DPO
python src/llm_finetuning/preference_alignment/dpo/ultrafeedback/train.py
python src/llm_finetuning/preference_alignment/dpo/webgpt/train.py

# KTO
python src/llm_finetuning/preference_alignment/kto/kto_mix/train.py

# ORPO
python src/llm_finetuning/preference_alignment/orpo/ultrafeedback/train.py

# PPO
python src/llm_finetuning/preference_alignment/ppo/ultrafeedback/train.py
python src/llm_finetuning/preference_alignment/ppo/webgpt/train.py
```

## Config reference

**DPO / ORPO** (`dpo/`, `orpo/`):

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | str | HuggingFace model ID (Unsloth variant) |
| `max_seq_length` | int | Maximum sequence length |
| `output_dir` | str | Where to save the trained model |
| `learning_rate` | float | Learning rate |
| `num_train_epochs` | int | Training epochs |
| `per_device_train_batch_size` | int | Batch size per GPU |
| `gradient_accumulation_steps` | int | Steps before optimizer update |
| `dataset_id` | str \| null | Optional HuggingFace dataset ID override. Falls back to loader `CONFIG.dataset_id` if omitted. |
| `dataset_subset` | str \| null | Optional dataset config/subset override. Falls back to loader `CONFIG.subset` if omitted. |
| `dataset_split` | str | Dataset split. Default: `train_prefs` for UltraFeedback, `train` for WebGPT. |
| `dpo_beta` / `orpo_beta` | float | KL penalty / odds-ratio coefficient (default 0.1) |
| `lora_r` | int | QLoRA rank |
| `lora_alpha` | int | QLoRA scaling factor |
| `target_modules` | list[str] | Modules to apply QLoRA to |

**KTO** (`kto/`):

Same as DPO/ORPO but no `dpo_beta`/`orpo_beta`. Supports `dataset_id`, `dataset_subset`, and `dataset_split` (default: `train`). The `trl-lib/kto-mix-14k` dataset is passed directly to `KTOTrainer` — no tokenizer-based preprocessing.

**PPO** (`ppo/`):

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | str | HuggingFace model ID |
| `output_dir` | str | Where to save the trained model |
| `learning_rate` | float | Learning rate |
| `dataset_id` | str \| null | Optional HuggingFace dataset ID override. Falls back to loader `CONFIG.dataset_id` if omitted. |
| `dataset_subset` | str \| null | Optional dataset config/subset override. Falls back to loader `CONFIG.subset` if omitted. |
| `dataset_split` | str | Dataset split. Default: `train_prefs` for UltraFeedback, `train` for WebGPT. |
| `batch_size` | int | Total rollout batch size |
| `mini_batch_size` | int | Mini-batch size for PPO optimization step |
| `gradient_accumulation_steps` | int | Steps before optimizer update |
| `ppo_epochs` | int | PPO optimization epochs per rollout batch |
| `max_new_tokens` | int | Max tokens to generate per step |

## Models

| Pipeline | Default model |
|----------|--------------|
| DPO UltraFeedback | `unsloth/zephyr-sft-bnb-4bit` |
| DPO WebGPT | `unsloth/llama-3-8b-bnb-4bit` |
| KTO Mix | `unsloth/Qwen2.5-1.5B-Instruct` |
| ORPO UltraFeedback | `unsloth/llama-3-8b-bnb-4bit` |
| PPO UltraFeedback | `unsloth/llama-3-8b-bnb-4bit` |
| PPO WebGPT | `unsloth/llama-3-8b-bnb-4bit` |

To use a different model, copy `config.yaml`, update `model_id` and `output_dir`. For DPO/ORPO, ensure the model has a chat template compatible with `apply_chat_template`. For PPO, any causal LM supported by `AutoModelForCausalLMWithValueHead` will work.

---

## Adding new pipelines

### Adding a new DPO pipeline

**Example: add DPO on a new dataset called `newpref`**

#### 1. Create the directory

```bash
mkdir -p src/llm_finetuning/preference_alignment/dpo/newpref
touch src/llm_finetuning/preference_alignment/dpo/newpref/__init__.py
```

#### 2. Write `data_processing.py`

Implement a `PreferenceDatasetLoader` subclass returning `prompt`, `chosen`, and `rejected`
columns as strings (after `apply_chat_template`):

```python
from llm_finetuning.core import DatasetConfig
from llm_finetuning.preference_alignment.base_loader import PreferenceDatasetLoader


class NewPrefDPOLoader(PreferenceDatasetLoader):
    CONFIG = DatasetConfig(dataset_id="your-hf-id", subset=None)

    def __init__(self, tokenizer, config: DatasetConfig | None = None) -> None:
        super().__init__(config or self.CONFIG, tokenizer)

    def format_example(self, example):
        # return {"prompt": str, "chosen": str, "rejected": str}
        ...
```

#### 3. Write `train.py`

Copy `dpo/ultrafeedback/train.py`, replace the data loading import with your loader class
and the dataset line with the override-aware pattern:

```python
loader_config = DatasetConfig(
    dataset_id=config.get("dataset_id", NewPrefDPOLoader.CONFIG.dataset_id),
    subset=config.get("dataset_subset", NewPrefDPOLoader.CONFIG.subset),
)
dataset = NewPrefDPOLoader(tokenizer, config=loader_config).load(
    config.get("dataset_split", "train")
)
```

#### 4. Write `config.yaml`

Copy `dpo/ultrafeedback/config.yaml`, update `model_id` and `output_dir`. Optionally add:

```yaml
dataset_id: "your-hf-id"       # optional override
dataset_subset: null            # optional override
dataset_split: "train"
```

---

### Adding a new ORPO pipeline

**Example: add ORPO on a new dataset called `newpref`**

ORPO uses the same `prompt/chosen/rejected` format as DPO with a combined odds-ratio loss.

#### 1. Create the directory

```bash
mkdir -p src/llm_finetuning/preference_alignment/orpo/newpref
touch src/llm_finetuning/preference_alignment/orpo/newpref/__init__.py
```

#### 2. Write `data_processing.py`

Same format as DPO — implement a loader returning `prompt/chosen/rejected` columns.

#### 3. Write `train.py`

Copy `orpo/ultrafeedback/train.py` and replace the data loading import with your loader
and the dataset line with the override-aware pattern:

```python
loader_config = DatasetConfig(
    dataset_id=config.get("dataset_id", NewPrefORPOLoader.CONFIG.dataset_id),
    subset=config.get("dataset_subset", NewPrefORPOLoader.CONFIG.subset),
)
dataset = NewPrefORPOLoader(tokenizer, config=loader_config).load(
    config.get("dataset_split", "train")
)
```

> **Important**: `PatchDPOTrainer()` must be called at module level before importing
> `ORPOTrainer`. This is already in the copied file — do not remove it.

```python
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()  # must come before trl imports

from trl import ORPOTrainer, ORPOConfig
```

#### 4. Write `config.yaml`

Copy `orpo/ultrafeedback/config.yaml`, update `model_id`, `output_dir`, and `orpo_beta`
if needed.

---

### Adding a new KTO pipeline

**Example: add KTO on a new dataset called `newpref`**

KTO uses binary desirability labels instead of paired comparisons.

#### 1. Create the directory

```bash
mkdir -p src/llm_finetuning/preference_alignment/kto/newpref
touch src/llm_finetuning/preference_alignment/kto/newpref/__init__.py
```

#### 2. Write `data_processing.py`

Implement a loader returning a dataset with these columns as plain strings and a bool:

```python
{
    "prompt": "the user's question",
    "completion": "the model's response",
    "label": True,   # True = desirable, False = undesirable
}
```

> **Note**: `KTOTrainer` does not take a tokenizer argument for data loading. Format the
> columns as plain strings without applying a chat template.

#### 3. Write `train.py`

Copy `kto/kto_mix/train.py` and replace the data loading import with your loader and the
dataset line with the override-aware pattern:

```python
loader_config = DatasetConfig(
    dataset_id=config.get("dataset_id", NewPrefKTOLoader.CONFIG.dataset_id),
    subset=config.get("dataset_subset", NewPrefKTOLoader.CONFIG.subset),
)
dataset = NewPrefKTOLoader(config=loader_config).load(config.get("dataset_split", "train"))
```

#### 4. Write `config.yaml`

Copy `kto/kto_mix/config.yaml`, update `model_id` and `output_dir`. Optionally add:

```yaml
dataset_id: "your-hf-id"       # optional override
dataset_subset: null            # optional override
dataset_split: "train"
```

---

### Adding a new PPO pipeline

**Example: add PPO on a new dataset called `newpref`**

PPO requires a value head and a manual rollout loop scored by the OpenAssistant pointwise reward model.

#### 1. Create the directory

```bash
mkdir -p src/llm_finetuning/preference_alignment/ppo/newpref
touch src/llm_finetuning/preference_alignment/ppo/newpref/__init__.py
```

#### 2. Write `data_processing.py`

Implement a loader returning a dataset with a single `prompt` column (str):

```python
{"prompt": "the user's question"}
```

#### 3. Write `train.py`

Copy `ppo/ultrafeedback/train.py` and replace the data loading import with your loader
and the dataset line with the override-aware pattern:

```python
loader_config = DatasetConfig(
    dataset_id=config.get("dataset_id", NewPrefPPOLoader.CONFIG.dataset_id),
    subset=config.get("dataset_subset", NewPrefPPOLoader.CONFIG.subset),
)
dataset = NewPrefPPOLoader(config=loader_config).load(config.get("dataset_split", "train"))
```

#### 4. Write `config.yaml`

Copy `ppo/ultrafeedback/config.yaml`, update `model_id` and `output_dir`. Optionally add:

```yaml
dataset_id: "your-hf-id"       # optional override
dataset_subset: null            # optional override
dataset_split: "train"
```
