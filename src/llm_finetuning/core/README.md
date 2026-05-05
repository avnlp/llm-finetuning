# Core (Class-Based Architecture)

This package contains the repo’s class-based building blocks used across all pipelines:

- `PromptTemplate`: frozen prompt container with `render_user()` and `to_messages()`.
- `DatasetConfig`: frozen configuration for `BaseDatasetLoader` (dataset ID, subset, caching, etc.).
- `BaseDatasetLoader`: abstract dataset loader that wraps `datasets.load_dataset()` and `Dataset.map()`.
- `RewardConfig`: frozen configuration for `BaseReward` (sets the TRL log name).
- `BaseReward`: abstract callable reward compatible with TRL `GRPOTrainer(reward_funcs=[...])`.

Evaluation-specific rewards (DeepEval, Evidently) live in `llm_finetuning.core.llm_judges` — importing them is opt-in to avoid pulling heavyweight evaluation dependencies into unrelated code paths.

---

## Dataset Loader Contract

All dataset loaders are subclasses of `BaseDatasetLoader` and implement:

```python
def format_example(self, example: dict[str, Any]) -> dict[str, Any]:
    ...
```

`BaseDatasetLoader.load(split)` loads the HF dataset split and maps `format_example()` over it (optionally removing source columns).

For PPO, call `BaseDatasetLoader.tokenize(ds, tokenizer)` after `load()` to convert the `"prompt"` column to `"input_ids"`.

Concrete loaders define a class-level `CONFIG = DatasetConfig(...)` as their default dataset settings. They accept an optional `config: DatasetConfig | None = None` parameter and pass `config or self.CONFIG` to `BaseDatasetLoader`. This lets `train.py` override the dataset from YAML without subclassing:

```python
class SomeLoader(BaseDatasetLoader):
    CONFIG = DatasetConfig(dataset_id="...", subset="...")

    def __init__(self, config: DatasetConfig | None = None) -> None:
        super().__init__(config or self.CONFIG)
```

SFT and preference loaders may also accept extra constructor arguments such as `tokenizer`, `fraction`, or `max_example_tokens`.

### Expected output schemas

| Trainer | Expected dataset columns |
|--------:|-------------------------|
| SFT (`SFTTrainer`) | `{"text": str}` |
| GRPO (`GRPOTrainer`) | `{"prompt": list[dict], "answer": str}` |
| DPO / ORPO | `{"prompt": str, "chosen": str, "rejected": str}` |
| PPO | `{"prompt": str}` → tokenized to `{"input_ids": list[int]}` |
| KTO | `{"prompt": str, "completion": str, "label": bool}` |

---

## Reward Contract (GRPO)

All GRPO rewards are subclasses of `BaseReward` and implement the exact TRL callable signature:

```python
def __call__(self, prompts: list, completions: list, **kwargs: Any) -> list[float]:
    ...
```

Inputs:
- `prompts`: list of prompt message lists (each prompt is a list of `{role, content}` dicts).
- `completions`: list where each item is a list containing a single assistant message dict; access text via `completion[0]["content"]`.
- `**kwargs`: passthrough of additional dataset columns (e.g. `answer`).

Output:
- `list[float]` with one score per completion.

Naming:
- TRL logs `reward_func.__name__`. `BaseReward` sets this from `RewardConfig(name=...)`.

Pickling / strict type checks:
- Call `reward.as_fn()` to get a plain function wrapper with the same signature and `__name__`. Use this when a framework requires a picklable callable or rejects non-function callables.

---

## LLM Judge Rewards (`llm_judges`)

`llm_finetuning.core.llm_judges` provides ready-made `BaseReward` subclasses backed by DeepEval and Evidently. Import from here when you need a concrete reward; the core namespace intentionally does not re-export these to keep heavyweight evaluation dependencies opt-in.

Class hierarchy:

```
BaseReward                                              (core/reward.py)
└── BaseLLMJudgeReward                                  (core/llm_judges/base.py)
    ├── DeepEvalSimpleReward                            (abstract — no reference answer)
    │   ├── DeepEvalAnswerRelevancyReward               (concrete)
    │   └── DeepEvalSummarizationReward                 (concrete)
    ├── AbstractDeepEvalGEvalRAGReward                  (abstract — requires reference answer)
    │   └── (domain subclasses declare rag_criteria)
    └── EvidentlyCorrectnessLLMReward                   (concrete — binary 1.0 / 0.0)
```

`JudgeInput` is the shared value object that extracts `question`, `response`, and optional `reference` from TRL’s nested message-list format.
