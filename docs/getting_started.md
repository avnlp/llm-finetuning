# Getting Started

## Prerequisites

Before installing, make sure you have:

- **Python 3.10+**
- **CUDA-capable GPU** (see [GPU memory guidance](#gpu-memory-guidance) below)
- **`uv` package manager**
- **HuggingFace account** — required to download gated models (e.g. Llama-3.2-3B)
- **OpenAI API key** — required only for `multi_hop_question_answering` and `medical_question_answering` pipelines, which use DeepEval and Evidently LLM-as-judge reward functions

Set up credentials before running:

```bash
# Required for gated HuggingFace models (Llama, Mistral, Gemma, etc.)
huggingface-cli login

# Required only for multi_hop_question_answering and medical_question_answering
export OPENAI_API_KEY="your-key"
```

---

## Installation

```bash
git clone https://github.com/avnlp/llm-finetuning
cd llm-finetuning
uv sync
```

This installs all dependencies including `transformers`, `trl`, `peft`, `unsloth`, `deepeval`, and `evidently`.

---

## All pipelines

39 pipelines across five modules. Each is self-contained — pick one and run it.

### Supervised fine-tuning (25 pipelines)

5 adapter methods × 5 datasets:

| Dataset | LoRA | QLoRA | DoRA | P-Tuning | Prefix-Tuning |
|---------|------|-------|------|----------|---------------|
| ARC | `lora/arc/train.py` | `qlora/arc/train.py` | `dora/arc/train.py` | `p_tuning/arc/train.py` | `prefix_tuning/arc/train.py` |
| Earnings Calls | `lora/earnings_call/train.py` | `qlora/earnings_call/train.py` | `dora/earnings_call/train.py` | `p_tuning/earnings_call/train.py` | `prefix_tuning/earnings_call/train.py` |
| FactScore | `lora/factscore/train.py` | `qlora/factscore/train.py` | `dora/factscore/train.py` | `p_tuning/factscore/train.py` | `prefix_tuning/factscore/train.py` |
| PopQA | `lora/popqa/train.py` | `qlora/popqa/train.py` | `dora/popqa/train.py` | `p_tuning/popqa/train.py` | `prefix_tuning/popqa/train.py` |
| TriviaQA | `lora/triviaqa/train.py` | `qlora/triviaqa/train.py` | `dora/triviaqa/train.py` | `p_tuning/triviaqa/train.py` | `prefix_tuning/triviaqa/train.py` |

All paths are relative to `src/llm_finetuning/supervised_finetuning/`.

### Math reasoning (2 pipelines)

| Pipeline | Command |
|----------|---------|
| Stage 1 — SFT on OpenR1-Math (format priming) | `python src/llm_finetuning/math_reasoning/sft/openr1_math/train.py` |
| Stage 2 — GRPO on GSM8K | `python src/llm_finetuning/math_reasoning/grpo/gsm8k/train.py` |

> **Two-stage pipeline**: Stage 1 fine-tunes `Qwen3-4B-Base` on `open-r1/OpenR1-Math-220k` to prime the model for `<reasoning>/<answer>` output format. Stage 2 then runs GRPO on GSM8K using the Stage 1 checkpoint as the base model (via `config_qwen3.yaml`). Run Stage 1 before Stage 2 when using the Qwen3 variant. Stage 2 can also be run standalone with its default `config.yaml` (Llama-3.2).

### Multi-hop question answering (3 pipelines)

> Requires `OPENAI_API_KEY`.

| Pipeline | Command |
|----------|---------|
| GRPO on HotpotQA | `python src/llm_finetuning/multi_hop_question_answering/grpo/hotpotqa/train.py` |
| GRPO on FreshQA | `python src/llm_finetuning/multi_hop_question_answering/grpo/freshqa/train.py` |
| GRPO on MuSiQue | `python src/llm_finetuning/multi_hop_question_answering/grpo/musique/train.py` |

### Medical question answering (3 pipelines)

> Requires `OPENAI_API_KEY`.

| Pipeline | Command |
|----------|---------|
| GRPO on MedQA | `python src/llm_finetuning/medical_question_answering/medqa/train.py` |
| GRPO on BioASQ | `python src/llm_finetuning/medical_question_answering/bioasq/train.py` |
| GRPO on PubMedQA | `python src/llm_finetuning/medical_question_answering/pubmedqa/train.py` |

### Preference alignment (6 pipelines)

| Pipeline | Command |
|----------|---------|
| DPO on UltraFeedback | `python src/llm_finetuning/preference_alignment/dpo/ultrafeedback/train.py` |
| DPO on WebGPT | `python src/llm_finetuning/preference_alignment/dpo/webgpt/train.py` |
| KTO on KTO Mix | `python src/llm_finetuning/preference_alignment/kto/kto_mix/train.py` |
| ORPO on UltraFeedback | `python src/llm_finetuning/preference_alignment/orpo/ultrafeedback/train.py` |
| PPO on UltraFeedback | `python src/llm_finetuning/preference_alignment/ppo/ultrafeedback/train.py` |
| PPO on WebGPT | `python src/llm_finetuning/preference_alignment/ppo/webgpt/train.py` |

> PPO pipelines use `OpenAssistant/reward-model-deberta-v3-large-v2` (a DeBERTa-v3 pointwise reward model) downloaded automatically via `PointwiseRewardModel` on first run.

---

## Running your first pipeline

### Supervised fine-tuning — LoRA on ARC

```bash
python src/llm_finetuning/supervised_finetuning/lora/arc/train.py
```

Reads `config.yaml` in the same directory, downloads the loader's default dataset (`allenai/ai2_arc` for ARC, unless `dataset_id` is overridden in `config.yaml`), and saves the trained model to `./outputs/supervised_finetuning/lora/arc`.

### GRPO math reasoning — GSM8K

```bash
python src/llm_finetuning/math_reasoning/grpo/gsm8k/train.py
```

### GRPO multi-hop QA — HotpotQA

```bash
export OPENAI_API_KEY="your-key"
python src/llm_finetuning/multi_hop_question_answering/grpo/hotpotqa/train.py
```

### GRPO medical QA — MedQA

```bash
export OPENAI_API_KEY="your-key"
python src/llm_finetuning/medical_question_answering/medqa/train.py
```

### DPO preference alignment — UltraFeedback

```bash
python src/llm_finetuning/preference_alignment/dpo/ultrafeedback/train.py
```

### PPO preference alignment — UltraFeedback

```bash
python src/llm_finetuning/preference_alignment/ppo/ultrafeedback/train.py
```

---

## Overriding the default dataset

Each pipeline's loader has a built-in default dataset. You can override it from `config.yaml` without touching code.

**For `supervised_finetuning/` pipelines** (`split` key):

```yaml
dataset_id: "allenai/ai2_arc"
dataset_subset: "ARC-Challenge"
split: "train"
```

**For GRPO, math reasoning, medical QA, and preference alignment pipelines** (`dataset_split` key):

```yaml
dataset_id: "openai/gsm8k"
dataset_subset: "main"
dataset_split: "train"
```

All three keys are optional — omitting them falls back to the loader's class-level defaults. `supervised_finetuning` uses `split`; all other modules use `dataset_split`.

---

## Changing the base model

Open the pipeline's `config.yaml` and update `model_id`:

```yaml
# Before
model_id: "meta-llama/Llama-3.2-3B"

# After
model_id: "meta-llama/Llama-3.1-8B"
```

For GRPO and preference alignment pipelines, use Unsloth-quantized variants for reduced VRAM:

```yaml
model_id: "unsloth/Llama-3.1-8B-Instruct"
```

To run a pipeline with a different config file:

```bash
python src/llm_finetuning/math_reasoning/grpo/gsm8k/train.py --config config_mistral7b.yaml
```

---

## GPU memory guidance

| Technique | Typical VRAM |
|-----------|-------------|
| SFT LoRA (3B model) | 8–12 GB |
| SFT QLoRA (3B model) | 6–8 GB |
| GRPO QLoRA (3B model) | 12–16 GB |
| DPO QLoRA (7B model) | 16–24 GB |
| ORPO QLoRA (7B model) | 16–24 GB |
| KTO QLoRA (1.5B model) | 8–12 GB |
| PPO (8B model) | 40+ GB |

Reduce `per_device_train_batch_size` to `1` and increase `gradient_accumulation_steps` if you run out of memory.

---

## Output location

All pipelines save to `./outputs/<module>/<method>/<dataset>/` by default. Change `output_dir` in `config.yaml` to save elsewhere.
