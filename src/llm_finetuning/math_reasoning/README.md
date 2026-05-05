# Math Reasoning

SFT + GRPO + QLoRA training on GSM8K and OpenR1-Math-220k. The model learns to produce step-by-step solutions inside `<reasoning>` tags with a final numeric answer inside `<answer>` tags.

## What this module trains

Grade-school math word problems requiring multi-step arithmetic reasoning. GRPO rewards correct numeric answers and well-structured reasoning traces, encouraging the model to show its work.

## Expected output format

The model is trained to produce responses in this structure:

```
<reasoning>
Step 1: Janet's ducks lay 16 eggs per day.
Step 2: She eats 3 for breakfast and uses 4933828 for muffins — wait, let me re-read.
Step 3: She eats 3 for breakfast and bakes 4 into muffins, so she uses 3 + 4 = 7 eggs daily.
Step 4: Eggs remaining to sell: 16 - 7 = 9.
Step 5: At $2 per egg: 9 × $2 = $18.
</reasoning>
<answer>18</answer>
```

The `AnswerCorrectnessReward` extracts the numeric value from `<answer>` tags and compares it against the GSM8K ground truth. The format reward functions push the model toward multi-step, well-structured traces.

## How GRPO training works

```
FastLanguageModel.from_pretrained(load_in_4bit=True)
    → FastLanguageModel.get_peft_model(r=..., lora_alpha=..., ...)   # apply QLoRA
    → GSM8KLoader(config=loader_config).load(config.get("dataset_split", "train"))
    → GRPOTrainer(reward_funcs=[...])      # 5 reward functions injected here
    → trainer.train()
```

At each training step:
1. GRPO samples a batch of prompts and generates `num_generations=4` completions per prompt
2. Each of the 5 reward functions scores all completions in the batch
3. Scores are normalised within each group of 4 to compute relative advantages
4. Policy is updated to increase the probability of higher-advantage completions

No reward model is trained. All reward functions are callable Python objects evaluated at each step.

## Two-stage pipeline (Qwen-3 Base)

For base models without instruction-following capability, a two-stage pipeline is available:

1. **Stage 1 — SFT format-priming**: Train on a small subset of OpenR1-Math-220k to teach the model the `<reasoning>/<answer>` output format.
2. **Stage 2 — GRPO on GSM8K**: Fine-tune mathematical reasoning with reward functions using the format-primed checkpoint.

### How to run

```bash
# Stage 1: SFT on OpenR1-Math (~60–128 short examples, ~3 min on T4)
python src/llm_finetuning/math_reasoning/sft/openr1_math/train.py

# Stage 2: GRPO on GSM8K using the SFT checkpoint
python src/llm_finetuning/math_reasoning/grpo/gsm8k/train.py --config config_qwen3.yaml
```

## Datasets

The values shown are defaults from each loader's `CONFIG`. Both Stage 1 (`OpenR1MathSFTLoader`) and Stage 2 (`GSM8KLoader`) support YAML dataset overrides via `dataset_id`, `dataset_subset`, and `dataset_split`.

| Dataset | Default HF ID | Default Config | Default Split | Size | Notes |
|---------|---------------|----------------|---------------|------|-------|
| GSM8K | `openai/gsm8k` | `main` | `train` | 7.47k | Numeric answer extracted via regex from `#### {number}` at end of chain-of-thought |
| OpenR1-Math-220k | `open-r1/OpenR1-Math-220k` | `default` | `train` | 93.7k (filtered to ~60–128) | Format-priming SFT; `<think>` tags converted to `<reasoning>/<answer>` |

## Reward functions

**Correctness** (`reward_functions/correctness/`):

| Function | Score range | What it measures |
|----------|-------------|-----------------|
| `AnswerCorrectnessReward` | −1.0 to 3.0 | Extracts answer from `<answer>` tags; scores: 3.0 exact match, 0.5 within 10%, 0.25 within 20%, −1.0 wrong, −0.5 answer not numeric, 0.0 no `<answer>` tag |

**Format** (`reward_functions/format/`):

| Function | Score range | What it rewards |
|----------|-------------|----------------|
| `ReasoningTagsReward` | −2.0 to 2.0 | Exactly one each of `<reasoning>`, `</reasoning>`, `<answer>`, `</answer>`; +0.5 per correct tag, −0.5 per incorrect |
| `StepFormatReward` | 0.0 to 1.0 | At least 3 numbered/bulleted steps; partial credit below threshold |
| `MultilineComplianceReward` | 0.0 to 1.0 | At least 5 non-empty lines; partial credit below threshold |
| `ResponseStructureReward` | 0.0 to 1.0 | Having both `<reasoning>` and `<answer>` blocks; 0.5 each |

See [`docs/reward_functions.md`](../../../docs/reward_functions.md) for full scoring logic.

## How to run

```bash
python src/llm_finetuning/math_reasoning/grpo/gsm8k/train.py --config config.yaml
```

To run a different model, create a new config file and pass it with `--config`:

```bash
cp src/llm_finetuning/math_reasoning/grpo/gsm8k/config.yaml \
   src/llm_finetuning/math_reasoning/grpo/gsm8k/config_phi4.yaml
# Edit config_phi4.yaml: update model_id and output_dir
python src/llm_finetuning/math_reasoning/grpo/gsm8k/train.py --config config_phi4.yaml
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
| `max_prompt_length` | int | `256` | Max tokens in prompt |
| `max_completion_length` | int | `512` | Max tokens in completion |
| `num_train_epochs` | int | `1` | Training epochs |
| `max_grad_norm` | float | `0.1` | Gradient clipping norm |
| `logging_steps` | int | `1` | Log every N steps |
| `lora_r` | int | `8` | QLoRA rank |
| `lora_alpha` | int | `8` | QLoRA scaling factor |
| `target_modules` | list[str] | `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` | Modules to apply QLoRA to |
| `dataset_id` | str \| null | — | Optional HuggingFace dataset ID override. Falls back to loader `CONFIG.dataset_id` if omitted. |
| `dataset_subset` | str \| null | — | Optional dataset config/subset override. Falls back to loader `CONFIG.subset` if omitted. |

> The table above covers the GRPO (Stage 2) pipeline. The Stage 1 `openr1_math` SFT pipeline also accepts `dataset_id`, `dataset_subset`, and `dataset_split`.

## Models

Default base model: `unsloth/Llama-3.2-3B-Instruct`.

Compatible Unsloth models: `unsloth/Phi-4`, `unsloth/mistral-7b-bnb-4bit`, `unsloth/Llama-3.1-8B-Instruct`, `unsloth/gemma-3-1b-it`.
