# Dataset Reference

All 16 datasets used across the five training modules. This is the authoritative source for the default loader values: HuggingFace IDs, configs, splits, and preprocessing details.

The HF ID, config, and split values shown in each table are the defaults baked into each loader's class-level `CONFIG`. They can be overridden at runtime from `config.yaml` using:
- `dataset_id` — overrides the HF dataset ID
- `dataset_subset` — overrides the dataset config/subset
- `split` (supervised_finetuning only) or `dataset_split` (all other modules) — selects the HF split

Omitting any of these keys preserves the loader's default.

---

## Supervised Fine-tuning Datasets

Used by all 5 adapter methods (LoRA, QLoRA, DoRA, P-Tuning, Prefix-Tuning). Output: single `"text"` column via `apply_chat_template`, consumed by `SFTTrainer(dataset_text_field="text")`. In code, these datasets are loaded via `SFTDatasetLoader` subclasses in `src/llm_finetuning/supervised_finetuning/loaders.py`.

The values shown are the default dataset settings used when `dataset_id`, `dataset_subset`, and `split` are omitted from `config.yaml`.

### 1. ARC (AI2 Reasoning Challenge)

| Key | Value |
|-----|-------|
| **Default HF ID** | `allenai/ai2_arc` |
| **Default Config** | `ARC-Challenge` |
| **Default Split** | `train` |
| **Size** | ~1.1k |

**Raw fields**: `question` (str), `choices` (dict: `label` list[str], `text` list[str]), `answerKey` (str)

**Preprocessing**: `format_choices(labels, texts)` flattens the nested choices dict into `"A. text\nB. text\n..."` before template substitution.

---

### 2. Earnings Calls

| Key | Value |
|-----|-------|
| **Default HF ID** | `lamini/earnings-calls-qa` |
| **Default Config** | None |
| **Default Split** | `train` |
| **Size** | ~3.7k |

**Raw fields**: `question` (str), `answer` (str), `transcript` (str), `ticker` (str), `company` (str), `date` (str), `q` (str)

**Preprocessing**: `transcript` field renamed to `context` to match the `{context}` placeholder in the user template.

---

### 3. FactScore (Biography Generation)

| Key | Value |
|-----|-------|
| **Default HF ID** | `awinml/factscore_unlabelled_alpaca_13b_retrieval` |
| **Default Config** | None |
| **Default Split** | `train` |
| **Size** | 500 |

**Raw fields**: `input` (str — bio prompt), `output` (str — biography), `topic` (str), `ctxs` (list[dict] — 25 Wikipedia passages)

**Preprocessing**: No loader `preprocess()` override needed. `input` → user message, `output` → assistant response. `ctxs` not injected.

---

### 4. PopQA

| Key | Value |
|-----|-------|
| **Default HF ID** | `akariasai/PopQA` |
| **Default Config** | None |
| **Default Split** | `test` |
| **Size** | ~14k |

**Raw fields**: `question` (str), `possible_answers` (list[str]), `subj` (str), `prop` (str), `obj` (str)

**Preprocessing**: No loader `preprocess()` override needed. `possible_answers` is a list; `str()` conversion applied when used as response target.

---

### 5. TriviaQA

| Key | Value |
|-----|-------|
| **Default HF ID** | `mandarjoshi/trivia_qa` |
| **Default Config** | `rc` |
| **Default Split** | `train` |
| **Size** | ~88k |

**Raw fields**: `question` (str), `answer` (dict: `value` str, `aliases` list[str]), `search_results` (dict: `search_context` list, ...)

**Preprocessing**: `answer["value"]` extracted as flat string; `context` built from `search_results["search_context"][0]` (first result).

---

## Multi-Hop QA Datasets

GRPO + QLoRA training. Output: `"prompt"` column (list of message dicts) + `"answer"` column (str). `GRPOTrainer` passes all dataset columns as `**kwargs` to reward functions.

The values shown are the default dataset settings used when `dataset_id`, `dataset_subset`, and `dataset_split` are omitted from `config.yaml`.

### 6. HotpotQA

| Key | Value |
|-----|-------|
| **Default HF ID** | `hotpotqa/hotpot_qa` |
| **Default Config** | `distractor` |
| **Default Split** | `train` |
| **Size** | ~90k |

**Raw fields**: `question` (str), `answer` (str), `context` (dict), `supporting_facts` (dict), `type` (str), `level` (str)

**Preprocessing**: `question` → `prompt` message list; `answer` → `answer` column; context not injected.

> Note: Uses `distractor` config, not `fullwiki`.

---

### 7. FreshQA (via SealQA)

| Key | Value |
|-----|-------|
| **Default HF ID** | `vtllms/sealqa` |
| **Default Config** | `longseal` |
| **Default Split** | `test` |
| **Size** | 264 |

**Raw fields**: `question` (str), `answer` (str), `golds` (list[dict]), `12_docs`/`20_docs`/`30_docs`, `freshness`, `question_types`, `topic`

**Preprocessing**: `question` → `prompt` message list; `answer` → `answer` column; document fields not used.

> Note: Only a `test` split exists; all 264 rows are used for training.

---

### 8. MuSiQue

| Key | Value |
|-----|-------|
| **Default HF ID** | `dgslibisey/MuSiQue` |
| **Default Config** | None |
| **Default Split** | `train` |
| **Size** | ~19.9k (filtered) |

**Raw fields**: `id` (str), `question` (str), `answer` (str), `answer_aliases` (list[str]), `answerable` (bool), `paragraphs` (list[dict]), `question_decomposition` (list[dict])

**Preprocessing**: Filtered to `answerable == True`; `question` → `prompt` message list; `answer` → `answer` column.

---

## Math Reasoning Datasets

The values shown are the default dataset settings used when `dataset_id`, `dataset_subset`, and `dataset_split` are omitted from `config.yaml`. Both Stage 1 (`OpenR1MathSFTLoader`) and Stage 2 (`GSM8KLoader`) support YAML dataset overrides.

### 9. OpenR1-Math-220k (Stage 1 SFT)

| Key | Value |
|-----|-------|
| **Default HF ID** | `open-r1/OpenR1-Math-220k` |
| **Default Config** | `default` |
| **Default Split** | `train` |
| **Size** | ~220k |

**Raw fields**: `problem` (str), `solution` (str — chain-of-thought already formatted with `<reasoning>` and `<answer>` tags)

**Preprocessing**: None needed. `problem` → user message, `solution` → assistant response via chat template. Output: single `"text"` column consumed by `SFTTrainer`.

**Used by**: `math_reasoning/sft/openr1_math/` (Stage 1 of the two-stage math pipeline). The trained checkpoint is consumed by Stage 2 (`math_reasoning/grpo/gsm8k/` with `config_qwen3.yaml`).

---

### 10. GSM8K

| Key | Value |
|-----|-------|
| **Default HF ID** | `openai/gsm8k` |
| **Default Config** | `main` |
| **Default Split** | `train` |
| **Size** | 7.47k |

**Raw fields**: `question` (str), `answer` (str — chain-of-thought ending with `#### {number}`)

**Preprocessing**: Numeric answer extracted via `re.search(r"####\s*(.+)", text)`; `question` → `prompt` message list.

---

## Preference Alignment Datasets

The values shown are the default dataset settings used when `dataset_id`, `dataset_subset`, and `dataset_split` are omitted from `config.yaml`. All preference alignment pipelines support YAML dataset overrides.

### 11. UltraFeedback Binarized

| Key | Value |
|-----|-------|
| **Default HF ID** | `HuggingFaceH4/ultrafeedback_binarized` |
| **Default Config** | None |
| **Default Split** | `train_prefs` |
| **Size** | ~60k |

**Raw fields**: `chosen` (list[dict] — role/content pairs), `rejected` (list[dict])

**Preprocessing (DPO/ORPO)**: Extract user message from `chosen[0]`; `apply_chat_template` on chosen/rejected → `prompt/chosen/rejected` columns.

**Preprocessing (PPO)**: Load raw; `PointwiseRewardModel` (`OpenAssistant/reward-model-deberta-v3-large-v2`) scores completions at inference time.

---

### 12. KTO Mix

| Key | Value |
|-----|-------|
| **Default HF ID** | `trl-lib/kto-mix-14k` |
| **Default Config** | None |
| **Default Split** | `train` |
| **Size** | 14k |

**Raw fields**: `prompt` (str), `completion` (str), `label` (bool)

**Preprocessing**: None required — dataset is already in KTO format. Passed directly to `KTOTrainer`.

---

### 13. WebGPT Comparisons

| Key | Value |
|-----|-------|
| **Default HF ID** | `openai/webgpt_comparisons` |
| **Default Config** | None |
| **Default Split** | `train` |
| **Size** | ~19k |

**Raw fields**: `question` (dict: `full_text` str), `answer_0` (dict: `text` str), `answer_1` (dict: `text` str), `preference` (float)

**Preprocessing**: `preference > 0` → answer_0 = chosen; `preference < 0` → answer_1 = chosen; rows where `preference == 0` dropped.

---

## Medical QA Datasets

GRPO + QLoRA training. Output: `"prompt"` + `"answer"` columns.

The values shown are the default dataset settings used when `dataset_id`, `dataset_subset`, and `dataset_split` are omitted from `config.yaml`.

### 14. MedQA

| Key | Value |
|-----|-------|
| **Default HF ID** | `bigbio/med_qa` |
| **Default Config** | `med_qa_en_4options_bigbio_qa` |
| **Default Split** | `train` |
| **Size** | ~10.2k |

**Raw fields**: `question` (str), `choices` (dict: `key` list[str], `value` list[str]), `answer` (str)

**Preprocessing**: Choices formatted as `"A. text\nB. text\n..."` and appended to question; `answer` → `answer` column.

---

### 15. BioASQ

| Key | Value |
|-----|-------|
| **Default HF ID** | `enelpol/rag-mini-bioasq` |
| **Default Config** | `question-answer-passages` |
| **Default Split** | `train` |
| **Size** | 4,010 |

**Raw fields**: `question` (str), `answer` (str), `id` (int), `relevant_passage_ids` (list[int])

**Preprocessing**: `question` → `prompt` message list; `answer` → `answer` column; `relevant_passage_ids` not used.

> Note: `bigbio/bioasq_task_b` requires manual download from participants-area.bioasq.org. `enelpol/rag-mini-bioasq` is the publicly available substitute.

---

### 16. PubMedQA

| Key | Value |
|-----|-------|
| **Default HF ID** | `qiaojin/PubMedQA` |
| **Default Config** | `pqa_artificial` |
| **Default Split** | `train` |
| **Size** | 211k |

**Raw fields**: `pubid` (int), `question` (str), `context` (dict: `contexts` list[str], `labels` list[str], `meshes` list[str]), `long_answer` (str), `final_decision` (str)

**Preprocessing**: `question` → `prompt` message list; `long_answer` → `answer` column; `final_decision` and context passages not used.

---

## Summary

| # | Dataset | Default HF ID | Default Config | Default Split | Size | Module |
|---|---------|---------------|----------------|---------------|------|--------|
| 1 | ARC | `allenai/ai2_arc` | `ARC-Challenge` | `train` | ~1.1k | SFT |
| 2 | Earnings Calls | `lamini/earnings-calls-qa` | — | `train` | ~3.7k | SFT |
| 3 | FactScore | `awinml/factscore_unlabelled_alpaca_13b_retrieval` | — | `train` | 500 | SFT |
| 4 | PopQA | `akariasai/PopQA` | — | `test` | ~14k | SFT |
| 5 | TriviaQA | `mandarjoshi/trivia_qa` | `rc` | `train` | ~88k | SFT |
| 6 | HotpotQA | `hotpotqa/hotpot_qa` | `distractor` | `train` | ~90k | Multi-Hop QA |
| 7 | FreshQA | `vtllms/sealqa` | `longseal` | `test` | 264 | Multi-Hop QA |
| 8 | MuSiQue | `dgslibisey/MuSiQue` | — | `train` | ~19.9k | Multi-Hop QA |
| 9 | OpenR1-Math-220k | `open-r1/OpenR1-Math-220k` | `default` | `train` | ~220k | Math (Stage 1 SFT) |
| 10 | GSM8K | `openai/gsm8k` | `main` | `train` | 7.47k | Math (Stage 2 GRPO) |
| 11 | UltraFeedback | `HuggingFaceH4/ultrafeedback_binarized` | — | `train_prefs` | ~60k | Pref. Align. |
| 12 | KTO Mix | `trl-lib/kto-mix-14k` | — | `train` | 14k | Pref. Align. |
| 13 | WebGPT | `openai/webgpt_comparisons` | — | `train` | ~19k | Pref. Align. |
| 14 | MedQA | `bigbio/med_qa` | `med_qa_en_4options_bigbio_qa` | `train` | ~10.2k | Medical QA |
| 15 | BioASQ | `enelpol/rag-mini-bioasq` | `question-answer-passages` | `train` | 4,010 | Medical QA |
| 16 | PubMedQA | `qiaojin/PubMedQA` | `pqa_artificial` | `train` | 211k | Medical QA |
