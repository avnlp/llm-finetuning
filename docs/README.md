# Architecture and Training Patterns

This document explains how the repository is structured, why it is designed the way it is,
and how the five training paradigms connect to data, models, and rewards.

---

## Design philosophy

The repo contains **33 self-contained pipelines** — each with its own `train.py` and
`config.yaml`. There is no central dispatcher and no shared trainer class.

This design makes individual trade-offs explicit:

- **Isolation**: changing one pipeline (e.g. PPO on UltraFeedback) cannot break another
  (e.g. GRPO on HotpotQA). Each `train.py` owns its full training loop.
- **Readability**: every pipeline is independently readable from top to bottom without
  following imports across multiple abstraction layers.
- **Reproducibility**: each pipeline carries its own `config.yaml`. Pinning a config
  snapshot reproduces a run without worrying about shared defaults changing.

The trade-off is deliberate repetition. Boilerplate is accepted in exchange for clarity.
The only shared code lives in `core/` (three small abstractions) and in each module's
loader and reward files.

---

## The 3-file pipeline pattern

Every leaf pipeline directory contains exactly three files:

```
supervised_finetuning/lora/arc/
├── __init__.py
├── config.yaml      # all hyperparameters and dataset settings
└── train.py         # loads config, builds model, loads data, runs trainer
```

| File | Responsibility |
|------|---------------|
| `config.yaml` | All hyperparameters: model ID, learning rate, LoRA rank, output dir, split, etc. Dataset defaults live in the loader class `CONFIG`; optional `dataset_id`, `dataset_subset`, and `split`/`dataset_split` keys override those defaults when present. |
| `train.py` | Reads config, loads model + tokenizer, calls the dataset loader, constructs the trainer, calls `trainer.train()`, saves adapter weights. |
| `__init__.py` | Empty marker so the directory is importable. |

Running a pipeline is always the same command pattern:

```bash
python src/llm_finetuning/<module>/<method>/<dataset>/train.py
```

---

## The role of `core/`

`src/llm_finetuning/core/` contains three abstractions that all pipelines build on:

### `PromptTemplate`

An immutable dataclass holding a system prompt string, a user template string, and the
name of the response field. Subclasses of `SFTDatasetLoader` use it to format examples:

```python
# From src/llm_finetuning/core/prompt_template.py
@dataclass(frozen=True, slots=True)
class PromptTemplate:
    system_prompt: str
    user_template: str
    response_field: str = "answer"

    def render_user(self, **fields: Any) -> str:
        return self.user_template.format(**fields)

    def to_messages(self, **fields: Any) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.render_user(**fields)},
        ]
```

`PromptTemplate` is used only by SFT pipelines. GRPO loaders build message dicts
directly in `format_example()` because they need a list output, not a formatted string.

### `BaseDatasetLoader`

Abstract base class for all dataset loaders. Wraps `datasets.load_dataset()` and
`Dataset.map()`. Subclasses implement one method:

```python
# From src/llm_finetuning/core/dataset_loader.py
class BaseDatasetLoader(ABC):
    def __init__(self, config: DatasetConfig) -> None: ...

    def load(self, split: str) -> Dataset:
        ds = load_dataset(
            self.config.dataset_id,
            self.config.subset,
            split=split,
            cache_dir=self.config.cache_dir,
            **self.config.extra_load_kwargs,
        )
        return self._format_dataset(ds)

    @abstractmethod
    def format_example(self, example: dict[str, Any]) -> dict[str, Any]: ...
```

`DatasetConfig` carries the HuggingFace dataset ID, subset, cache dir, and whether to
remove source columns after mapping. Loader classes define default dataset metadata via a
class-level `CONFIG = DatasetConfig(...)`. Each `train.py` may construct a `DatasetConfig`
override from YAML and pass it into the loader; if the override keys are omitted, the
loader falls back to its `CONFIG` defaults.

```python
loader_config = DatasetConfig(
    dataset_id=config.get("dataset_id", Loader.CONFIG.dataset_id),
    subset=config.get("dataset_subset", Loader.CONFIG.subset),
)
dataset = Loader(config=loader_config).load(config["dataset_split"])
```

### `BaseReward`

Abstract callable for GRPO reward functions. Subclasses implement `__call__` with the
exact TRL signature and are passed directly to `GRPOTrainer(reward_funcs=[...])`:

```python
# From src/llm_finetuning/core/reward.py
class BaseReward(ABC):
    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self.__name__ = self.config.name

    @abstractmethod
    def __call__(
        self, prompts: list, completions: list, **kwargs: Any
    ) -> list[float]: ...
```

`BaseReward` sets `self.__name__` from `RewardConfig(name=...)` so TRL logs the reward
by a stable human-readable name rather than a generic class name.

---

## How trainers connect to data

Each trainer type expects a specific set of dataset columns. The loader's `format_example`
method is responsible for producing exactly those columns.

| Trainer | Required columns | Loader base | Where loaders live |
|---------|-----------------|-------------|-------------------|
| `SFTTrainer` | `{"text": str}` | `SFTDatasetLoader` | `supervised_finetuning/loaders.py` |
| `GRPOTrainer` | `{"prompt": list[dict], "answer": str}` | `BaseDatasetLoader` | `<module>/data_processing.py` |
| `DPOTrainer` / `ORPOTrainer` | `{"prompt": str, "chosen": str, "rejected": str}` | `BaseDatasetLoader` | `preference_alignment/<method>/` |
| `PPOTrainer` | `{"prompt": str}` | `BaseDatasetLoader` | `preference_alignment/ppo/` |
| `KTOTrainer` | `{"prompt": str, "completion": str, "label": bool}` | `BaseDatasetLoader` | `preference_alignment/kto/` |

The `GRPOTrainer` also passes all extra dataset columns as `**kwargs` to each reward
function. This is how `answer` reaches the correctness reward functions at training time.

---

## Adapter techniques vs RL techniques

### SFT pipelines — PEFT adapters + `SFTTrainer`

All 25 supervised fine-tuning pipelines follow the same flow:

```
Loader(tokenizer, config=loader_config).load(config["split"])
    → load_dataset + Dataset.map(format_example)
    → tokenizer.apply_chat_template → {"text": str}
    → get_peft_model(model, peft_config)
    → SFTTrainer(dataset_text_field="text")
    → trainer.train()
```

The five adapter techniques differ only in how `get_peft_model` prepares the model:

| Technique | PEFT config class | Key parameter |
|-----------|------------------|---------------|
| LoRA | `LoraConfig` | `r` (rank), `lora_alpha`, `lora_dropout` |
| QLoRA | `LoraConfig` + `BitsAndBytesConfig(load_in_4bit=True)` | NF4 quantization |
| DoRA | `LoraConfig(use_dora=True)` | Magnitude+direction decomposition |
| P-Tuning | `PromptEncoderConfig` | `num_virtual_tokens`, `encoder_hidden_size` |
| Prefix-Tuning | `PrefixTuningConfig` | `num_virtual_tokens` prepended per layer |

### GRPO pipelines — Unsloth + `GRPOTrainer`

The three GRPO modules (math reasoning, multi-hop QA, medical QA) use Unsloth's
`FastLanguageModel` for 4-bit loading and QLoRA, then pass callable reward objects
directly to `GRPOTrainer`:

```
Loader(config=loader_config).load(config["dataset_split"])
    → {"prompt": list[dict], "answer": str}
    → FastLanguageModel.from_pretrained(load_in_4bit=True)
    → FastLanguageModel.get_peft_model(model, LoraConfig(...))
    → GRPOTrainer(reward_funcs=[RewardA(), RewardB(), ...])
    → trainer.train()
```

### Preference alignment — TRL trainers

DPO, ORPO, and KTO are fully trainer-driven. PPO uses a manual rollout loop with
a `PointwiseRewardModel` (`OpenAssistant/reward-model-deberta-v3-large-v2`) scoring completions at each step.

DPO and ORPO require `PatchDPOTrainer()` from Unsloth at module level before importing
the TRL trainer — this is an Unsloth compatibility patch already in all affected
`train.py` files.

---

## Reward functions in GRPO

No reward model is trained in GRPO pipelines. Reward functions are plain Python callables
evaluated at each training step.

All reward functions are `BaseReward` subclasses with this call signature (the exact
signature TRL requires):

```python
def __call__(
    self, prompts: list, completions: list, **kwargs: Any
) -> list[float]:
    ...
```

- `prompts`: list of prompt message lists (each prompt is a `list[dict]` with
  `role`/`content` keys).
- `completions`: list where each item is a list containing one assistant message dict.
  Access the text as `completion[0]["content"]`.
- `**kwargs`: one key per extra dataset column (e.g. `kwargs["answer"]` for the ground
  truth).

Multiple reward functions are composed by passing them all to `reward_funcs=[...]`.
GRPO normalises scores within each generation group (default size 4) to compute relative
advantages before the policy update. This means absolute score magnitudes matter less
than relative scores within the group.

Reward functions are registered in each module's `reward_functions/__init__.py` and
imported in `train.py`.

---

## Output layout

All pipelines save to a path constructed from the module, method, and dataset:

```
./outputs/<module>/<method>/<dataset>/
```

Examples:

| Pipeline | Default `output_dir` |
|----------|---------------------|
| LoRA on ARC | `./outputs/supervised_finetuning/lora/arc` |
| GRPO on GSM8K | `./outputs/math_reasoning/grpo/gsm8k` |
| DPO on UltraFeedback | `./outputs/preference_alignment/dpo/ultrafeedback` |
| PPO on WebGPT | `./outputs/preference_alignment/ppo/webgpt` |
| GRPO on MedQA | `./outputs/medical_question_answering/medqa` |

Each pipeline saves two artifacts to `output_dir`:

1. **Adapter weights** — `model.save_pretrained(output_dir)`. For SFT and GRPO pipelines
   these are LoRA/QLoRA adapter files only (not the full base model weights).
2. **Tokenizer** — `tokenizer.save_pretrained(output_dir)`.

To change the output location, edit `output_dir` in the pipeline's `config.yaml`.
