# Supervised Fine-tuning

Adapter-based supervised fine-tuning across five techniques and five datasets. All pipelines use `SFTTrainer` with `apply_chat_template` formatting.

## What this module trains

Classification, QA, and generation tasks where the model learns to produce a target response given a formatted prompt. Five adapter methods are compared on identical datasets to benchmark adapter performance differences.

## Datasets

The values shown are defaults from each loader's `CONFIG`. They can be overridden from `config.yaml` using `dataset_id` and `dataset_subset`. This module uses `split` (not `dataset_split`) to select the HF split.

| Dataset | Default HF ID | Default Config | Default Split | Size | Notes |
|---------|---------------|----------------|---------------|------|-------|
| ARC | `allenai/ai2_arc` | `ARC-Challenge` | `train` | ~1.1k | Multiple-choice science questions; choices formatted as `A. text\nB. text\n...` |
| Earnings Calls | `lamini/earnings-calls-qa` | — | `train` | ~3.7k | `transcript` field copied to `context` before template substitution |
| FactScore | `awinml/factscore_unlabelled_alpaca_13b_retrieval` | — | `train` | 500 | Biography generation; `input` → user message, `output` → assistant response |
| PopQA | `akariasai/PopQA` | — | `test` | ~14k | `possible_answers` is a list; `str()` conversion applied as response |
| TriviaQA | `mandarjoshi/trivia_qa` | `rc` | `train` | ~88k | `answer["value"]` extracted; context from first `search_results["search_context"]` entry |

## How dataset loading works

All 25 pipelines use class-based dataset loaders in `src/llm_finetuning/supervised_finetuning/loaders.py`. Each loader converts a raw HuggingFace dataset into a single `"text"` column consumed by `SFTTrainer`.

```
loader_config = DatasetConfig(
    dataset_id=config.get("dataset_id", ARCLoader.CONFIG.dataset_id),
    subset=config.get("dataset_subset", ARCLoader.CONFIG.subset),
)
ARCLoader(tokenizer, config=loader_config).load(config["split"])
    → load_dataset(dataset_id, subset, split)
    → dataset.map(loader.format_example)              # may call loader.preprocess()
    → template.render_user(**example)                  # fill placeholders
    → [system_msg, user_msg, assistant_msg]           # assemble conversation
    → tokenizer.apply_chat_template(tokenize=False)   # serialize to string
    → Dataset({"text": [...]})
    → SFTTrainer(dataset_text_field="text")
```

If `dataset_id` and `dataset_subset` are omitted from YAML, the loader falls back to its class-level `CONFIG` defaults.

Dataset-specific preprocessing happens in each loader's `preprocess()` override. For ARC, it flattens the nested choices dict into `"A. text\nB. text\n..."`. For TriviaQA, it extracts `answer["value"]` and builds a `context` field. For Earnings Calls, it copies `transcript` to `context`. For FactScore and PopQA, no preprocessing is required.

An example formatted `"text"` entry for ARC (using Llama-3 chat template):

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a science reasoning expert. Analyze the question, evaluate each choice, and provide the correct answer.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Question: Which factor will most increase the rate of photosynthesis?
Choices:
A. higher CO2 concentration
B. lower temperature
C. more intense light
D. less water

Think step by step and provide the correct answer letter.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

C<|eot_id|>
```

All five adapter methods train on exactly this format. They differ in how the model is prepared before training: LoRA, DoRA, P-Tuning, and Prefix-Tuning call `get_peft_model(model, peft_config)`, while QLoRA loads the base model in 4-bit quantization and passes `peft_config` directly to `SFTTrainer`.

## Techniques

| Technique | PEFT Config | Key difference |
|-----------|------------|----------------|
| **LoRA** | `LoraConfig(r, lora_alpha, lora_dropout)` | Low-rank matrix decomposition; base weights frozen; trains in bfloat16 |
| **QLoRA** | Same as LoRA + `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` | 4-bit NF4 quantized base model; reduces VRAM by ~50% vs LoRA |
| **DoRA** | `LoraConfig(..., use_dora=True)` | Decomposes weight updates into magnitude + direction components |
| **P-Tuning** | `PromptEncoderConfig(num_virtual_tokens, encoder_hidden_size)` | Trains a small MLP that generates soft prompt embeddings; base weights frozen |
| **Prefix-Tuning** | `PrefixTuningConfig(num_virtual_tokens)` | Prepends trainable prefix vectors to key/value tensors at every attention layer; base weights frozen |

All techniques use `SFTTrainer` with `packing=False` and save both the adapter weights and tokenizer to `output_dir`.

## How to run

```bash
# LoRA
python src/llm_finetuning/supervised_finetuning/lora/arc/train.py
python src/llm_finetuning/supervised_finetuning/lora/earnings_call/train.py
python src/llm_finetuning/supervised_finetuning/lora/factscore/train.py
python src/llm_finetuning/supervised_finetuning/lora/popqa/train.py
python src/llm_finetuning/supervised_finetuning/lora/triviaqa/train.py

# QLoRA (same pattern for all 5 datasets)
python src/llm_finetuning/supervised_finetuning/qlora/arc/train.py

# DoRA
python src/llm_finetuning/supervised_finetuning/dora/arc/train.py

# P-Tuning
python src/llm_finetuning/supervised_finetuning/p_tuning/arc/train.py

# Prefix-Tuning
python src/llm_finetuning/supervised_finetuning/prefix_tuning/arc/train.py
```

## Config reference

**LoRA / QLoRA / DoRA configs** (`lora/`, `qlora/`, `dora/`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_id` | str | `"meta-llama/Llama-3.2-3B"` | HuggingFace model ID |
| `dataset_id` | str \| null | loader default | Optional HuggingFace dataset ID override. Read by `train.py` to build a `DatasetConfig` passed into the loader. |
| `dataset_subset` | str \| null | loader default | Optional dataset config/subset override. |
| `split` | str | dataset-specific | Dataset split passed to `loader.load(...)`. |
| `output_dir` | str | — | Where to save the trained model |
| `num_train_epochs` | int | `3` | Training epochs |
| `per_device_train_batch_size` | int | `1` | Batch size per GPU |
| `gradient_accumulation_steps` | int | `8` | Steps before optimizer update |
| `learning_rate` | float | `2.0e-4` | Learning rate |
| `save_strategy` | str | `"epoch"` | Checkpoint save strategy |
| `logging_steps` | int | `10` | Log every N steps |
| `lora_r` | int | `8` | LoRA rank |
| `lora_alpha` | int | `32` | LoRA scaling factor |
| `lora_dropout` | float | `0.05` | LoRA dropout |
| `use_dora` | bool | `false` | Enable DoRA (dora/ only) |
| `target_modules` | list[str] | `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` | Modules to apply LoRA to |

**P-Tuning config** (`p_tuning/`):

Replaces LoRA fields with:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_virtual_tokens` | int | `20` | Number of soft prompt tokens |
| `encoder_hidden_size` | int | `128` | MLP encoder hidden size |

**Prefix-Tuning config** (`prefix_tuning/`):

Replaces LoRA fields with:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_virtual_tokens` | int | `20` | Number of prefix tokens per layer |

## Models

Default base model: `meta-llama/Llama-3.2-3B`.

To run a different model, update `model_id` in `config.yaml`. The default `target_modules` list covers LLaMA-family attention and MLP layers; adjust for non-LLaMA architectures (e.g. Mistral uses the same names; Phi-3 uses `qkv_proj`, `o_proj`).

---

## Adding a new SFT pipeline

**Example: add QLoRA fine-tuning on a new dataset called `mydata`**

### 1. Create the directory

```bash
mkdir -p src/llm_finetuning/supervised_finetuning/qlora/mydata
touch src/llm_finetuning/supervised_finetuning/qlora/mydata/__init__.py
```

### 2. Add a prompt file

Create `src/llm_finetuning/supervised_finetuning/data_preparation/mydata_prompt.py`:

```python
from llm_finetuning.core import PromptTemplate

TEMPLATE = PromptTemplate(
    system_prompt="Your system prompt here.",
    user_template="Question: {question}",
    response_field="answer",
)
```

Add an import to `src/llm_finetuning/supervised_finetuning/data_preparation/__init__.py`.

### 3. Add a loader class

Open `src/llm_finetuning/supervised_finetuning/loaders.py` and add a loader subclass:

Add a new class to `src/llm_finetuning/supervised_finetuning/loaders.py`:

```python
class MyDataLoader(SFTDatasetLoader):
    CONFIG = DatasetConfig(dataset_id="your-hf-dataset-id", subset=None)

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: DatasetConfig | None = None,
    ) -> None:
        super().__init__(config or self.CONFIG, MYDATA_TEMPLATE, tokenizer)

    # Optional: override preprocess() for field reshaping
```

Import the prompt template at the top of `loaders.py`:

```python
from llm_finetuning.supervised_finetuning.data_preparation.mydata_prompt import (
    TEMPLATE as MYDATA_TEMPLATE,
)
```

See `loaders.py` for the five existing loader classes (`ARCLoader`, `TriviaQALoader`,
`FactScoreLoader`, `PopQALoader`, `EarningsCallLoader`) as reference implementations.

### 4. Write `train.py`

Copy the closest existing `train.py` (e.g. `qlora/arc/train.py`) and update:

- The dataset loader import: `from llm_finetuning.supervised_finetuning.loaders import MyDataLoader`
- The dataset line to use the override-aware pattern:

```python
loader_config = DatasetConfig(
    dataset_id=config.get("dataset_id", MyDataLoader.CONFIG.dataset_id),
    subset=config.get("dataset_subset", MyDataLoader.CONFIG.subset),
)
dataset = MyDataLoader(tokenizer, config=loader_config).load(config["split"])
```

- The `CONFIG_PATH` already points to `Path(__file__).parent / "config.yaml"`

### 5. Write `config.yaml`

```yaml
model_id: "meta-llama/Llama-3.2-3B"
split: "train"
output_dir: "./outputs/supervised_finetuning/qlora/mydata"
# Optional dataset overrides (loader class CONFIG.dataset_id / CONFIG.subset used if omitted):
# dataset_id: "your-hf-dataset-id"
# dataset_subset: null

num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
save_strategy: "epoch"
logging_steps: 10

lora_r: 8
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

> The loader class `CONFIG` provides the default dataset ID and subset. Including `dataset_id`
> and `dataset_subset` in YAML overrides those defaults at runtime. If omitted, the loader
> falls back to its `CONFIG`.

### 6. Run

```bash
python src/llm_finetuning/supervised_finetuning/qlora/mydata/train.py
```

---

## Unit testing guidance for SFT loaders

- Unit-test `format_example()` and `preprocess()` with a single raw example dict — no
  HuggingFace download required.
- Integration-test `load(split)` only if you can rely on HF downloads in CI; otherwise
  mock `load_dataset`.
- For `PromptTemplate`: assert `render_user()` correctly fills placeholders, and
  `to_messages()` returns the expected two-message list.
