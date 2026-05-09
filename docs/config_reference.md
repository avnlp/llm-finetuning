# Config Field Reference

All fields across all `config.yaml` files. Each field lists its type, default (if any), and which modules use it.

---

## Common fields (all modules)

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | str | HuggingFace model identifier. Used to load model and tokenizer. |
| `output_dir` | str | Path where trained model and tokenizer are saved. |
| `learning_rate` | float | Optimizer learning rate. |
| `num_train_epochs` | int | Number of full passes over the training dataset. |
| `per_device_train_batch_size` | int | Batch size per GPU. |
| `gradient_accumulation_steps` | int | Number of forward passes before each optimizer step. Effective batch size = `per_device_train_batch_size × gradient_accumulation_steps`. |
| `logging_steps` | int | Log training metrics every N steps. |
| `dataset_id` | str \| null | Optional HuggingFace dataset ID override. If omitted, the loader class `CONFIG.dataset_id` default is used. |
| `dataset_subset` | str \| null | Optional HuggingFace dataset config/subset override. If omitted, the loader class `CONFIG.subset` default is used. |

---

## Supervised fine-tuning fields (`supervised_finetuning/`)

| Field | Type | Modules | Description |
|-------|------|---------|-------------|
| `dataset_id` | str \| null | all SFT | Optional HuggingFace dataset ID override. Read by `train.py` to build a `DatasetConfig` override passed into the loader. If omitted, the loader class `CONFIG.dataset_id` default is used. |
| `dataset_subset` | str \| null | all SFT | Optional dataset config/subset override. Read by `train.py` to build a `DatasetConfig` override passed into the loader. If omitted, the loader class `CONFIG.subset` default is used. |
| `split` | str | all SFT | Dataset split to load (e.g. `"train"`). Note: this module uses `split`, not `dataset_split`. |
| `save_strategy` | str | all SFT | When to save checkpoints: `"epoch"` or `"steps"`. Default: `"epoch"`. |

### LoRA / QLoRA / DoRA only

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lora_r` | int | `8` | LoRA rank. Higher = more parameters, more expressive. |
| `lora_alpha` | int | `32` | LoRA scaling factor. Effective scale = `lora_alpha / lora_r`. |
| `lora_dropout` | float | `0.05` | Dropout applied to LoRA matrices. |
| `use_dora` | bool | `false` | Enable DoRA (weight decomposition). DoRA configs only. |
| `target_modules` | list[str] | `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` | Transformer modules to apply LoRA to. |

### P-Tuning only

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_virtual_tokens` | int | `20` | Number of trainable soft prompt tokens prepended to input. |
| `encoder_hidden_size` | int | `128` | Hidden size of the MLP encoder that generates soft prompt embeddings. |

### Prefix-Tuning only

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_virtual_tokens` | int | `20` | Number of prefix tokens prepended at each transformer layer. |

---

## GRPO fields (`multi_hop_question_answering/`, `math_reasoning/`, `medical_question_answering/`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_seq_length` | int | `2048` | Maximum sequence length passed to `FastLanguageModel.from_pretrained`. |
| `dataset_id` | str \| null | — | Optional HuggingFace dataset ID override. If omitted, the loader class `CONFIG.dataset_id` default is used. |
| `dataset_subset` | str \| null | — | Optional dataset config/subset override. If omitted, the loader class `CONFIG.subset` default is used. |
| `dataset_split` | str | `"train"` | Dataset split passed to `loader.load(...)`. |
| `num_generations` | int | `4` | Number of completions generated per prompt. GRPO compares these to compute relative rewards. |
| `max_prompt_length` | int | `512` | Maximum number of tokens in the prompt. Longer prompts are truncated. |
| `max_completion_length` | int | `512` | Maximum number of tokens in each generated completion. |
| `max_grad_norm` | float | `0.1` | Gradient clipping norm. Math reasoning only. |
| `lora_r` | int | `8` | QLoRA rank. |
| `lora_alpha` | int | `8` | QLoRA scaling factor. |
| `target_modules` | list[str] | `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` | Modules to apply QLoRA to. |

---

## Preference alignment fields (`preference_alignment/`)

### DPO (`dpo/`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_seq_length` | int | `4096` | Maximum sequence length. |
| `dataset_id` | str \| null | — | Optional HuggingFace dataset ID override. |
| `dataset_subset` | str \| null | — | Optional dataset config/subset override. |
| `dataset_split` | str | `"train_prefs"` | Dataset split. Default varies by dataset: `train_prefs` for UltraFeedback, `train` for WebGPT. |
| `dpo_beta` | float | `0.1` | KL divergence penalty coefficient. Higher = stay closer to reference model. |
| `lora_r` | int | `64` | QLoRA rank (higher than GRPO to support larger preference datasets). |
| `lora_alpha` | int | `64` | QLoRA scaling factor. |
| `target_modules` | list[str] | — | Same as GRPO. |

### KTO (`kto/`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_seq_length` | int | `4096` | Maximum sequence length. |
| `dataset_id` | str \| null | — | Optional HuggingFace dataset ID override. |
| `dataset_subset` | str \| null | — | Optional dataset config/subset override. |
| `dataset_split` | str | `"train"` | Dataset split passed to `loader.load(...)`. |
| `lora_r` | int | `16` | QLoRA rank. |
| `lora_alpha` | int | `16` | QLoRA scaling factor. |
| `target_modules` | list[str] | — | Same as GRPO. |

### ORPO (`orpo/`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_seq_length` | int | `4096` | Maximum sequence length. |
| `dataset_id` | str \| null | — | Optional HuggingFace dataset ID override. |
| `dataset_subset` | str \| null | — | Optional dataset config/subset override. |
| `dataset_split` | str | `"train_prefs"` | Dataset split passed to `loader.load(...)`. |
| `orpo_beta` | float | `0.1` | ORPO odds-ratio penalty coefficient. |
| `lora_r` | int | `16` | QLoRA rank. |
| `lora_alpha` | int | `16` | QLoRA scaling factor. |
| `target_modules` | list[str] | — | Same as GRPO. |

### PPO (`ppo/`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_id` | str \| null | — | Optional HuggingFace dataset ID override. |
| `dataset_subset` | str \| null | — | Optional dataset config/subset override. |
| `dataset_split` | str | `"train"` | Dataset split. Default varies: `train_prefs` for UltraFeedback, `train` for WebGPT. |
| `batch_size` | int | `64` | Total rollout batch size. |
| `mini_batch_size` | int | `1` | Mini-batch size for PPO optimization step. |
| `ppo_epochs` | int | `4` | Number of optimization epochs per rollout batch. |
| `max_new_tokens` | int | `128` | Max tokens to generate per step in the rollout loop. |

---

## Field presence by module

> **Split key note**: `supervised_finetuning` uses `split`; all other modules use `dataset_split`. The `math_reasoning/sft/openr1_math` pipeline also uses `dataset_split`.

| Field | SFT LoRA/QLoRA/DoRA | SFT P-Tuning | SFT Prefix | GRPO | DPO/ORPO | KTO | PPO |
|-------|---------------------|-------------|------------|------|----------|-----|-----|
| `model_id` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `output_dir` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `learning_rate` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `num_train_epochs` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| `per_device_train_batch_size` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| `gradient_accumulation_steps` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `logging_steps` | ✓ | ✓ | ✓ | ✓ | — | — | — |
| `dataset_id` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `dataset_subset` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `split` | ✓ | ✓ | ✓ | — | — | — | — |
| `dataset_split` | — | — | — | ✓ | ✓ | ✓ | ✓ |
| `save_strategy` | ✓ | ✓ | ✓ | — | — | — | — |
| `lora_r` | ✓ | — | — | ✓ | ✓ | ✓ | — |
| `lora_alpha` | ✓ | — | — | ✓ | ✓ | ✓ | — |
| `lora_dropout` | ✓ | — | — | — | — | — | — |
| `use_dora` | DoRA only | — | — | — | — | — | — |
| `target_modules` | ✓ | — | — | ✓ | ✓ | ✓ | — |
| `num_virtual_tokens` | — | ✓ | ✓ | — | — | — | — |
| `encoder_hidden_size` | — | ✓ | — | — | — | — | — |
| `max_seq_length` | — | — | — | ✓ | ✓ | ✓ | — |
| `num_generations` | — | — | — | ✓ | — | — | — |
| `max_prompt_length` | — | — | — | ✓ | ✓ | — | — |
| `max_completion_length` | — | — | — | ✓ | — | — | — |
| `max_grad_norm` | — | — | — | math only | — | — | — |
| `dpo_beta` | — | — | — | — | DPO only | — | — |
| `orpo_beta` | — | — | — | — | ORPO only | — | — |
| `batch_size` | — | — | — | — | — | — | ✓ |
| `mini_batch_size` | — | — | — | — | — | — | ✓ |
| `ppo_epochs` | — | — | — | — | — | — | ✓ |
| `max_new_tokens` | — | — | — | — | — | — | ✓ |
