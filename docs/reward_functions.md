# Reward Function Reference

Reward functions are Python callables injected into `GRPOTrainer` at construction. In this repo they are implemented as `BaseReward` subclasses (callable instances):

```python
trainer = GRPOTrainer(
    reward_funcs=[
        AnswerCorrectnessReward(),
        ReasoningTagsReward(),
        StepFormatReward(),
    ],
    ...
)
```

During training, GRPO generates `num_generations` completions per prompt. Each reward function is called once with the full batch:

```python
scores = reward_fn(prompts, completions, **kwargs)
```

- `prompts`: list of prompt message dicts
- `completions`: list of completion message dicts (each a list with one assistant message)
- `**kwargs`: all other dataset columns passed through (e.g. `answer`, `question`)

The callable returns `list[float]` — one score per completion. GRPO normalises scores within each group to compute relative advantages, then updates the policy.

---

## math_reasoning

Five reward functions are composed for GSM8K training. The dominant signal is `AnswerCorrectnessReward`; the format functions guide structure.

### Correctness

#### `AnswerCorrectnessReward`
**File**: `reward_functions/correctness/answer_correctness.py`
**Score range**: −1.0 to 3.0

Extracts the predicted answer from `<answer>...</answer>` tags in the completion, falling back to a `#### {number}` pattern. Compares against `kwargs["answer"]` (the ground truth numeric string from GSM8K).

| Condition | Score |
|-----------|-------|
| Exact string match | 3.0 |
| Numeric value within 10% of ground truth | 0.5 |
| Numeric value within 20% of ground truth | 0.25 |
| Wrong answer (parseable) | −1.0 |
| No parseable answer found | 0.0 |
| Exception during parsing | 0.0 |

---

### Format

#### `ReasoningTagsReward`
**File**: `reward_functions/format/reasoning_tags.py`
**Score range**: −2.0 to 2.0

Counts occurrences of each of the four tags: `<reasoning>`, `</reasoning>`, `<answer>`, `</answer>`. Expects exactly one of each.

- +0.5 per tag whose count equals 1
- −0.5 per tag whose count does not equal 1

Maximum score (all four correct): 2.0. Minimum (all four wrong): −2.0.

---

#### `StepFormatReward`
**File**: `reward_functions/format/step_format.py`
**Score range**: 0.0 to 1.0

Counts lines matching any step pattern: `Step N`, `N.`, or bullet characters (`-`, `*`, `•`).

| Steps found | Score |
|-------------|-------|
| ≥ 3 | 1.0 |
| < 3 | count / 3 |

---

#### `MultilineComplianceReward`
**File**: `reward_functions/format/multiline_compliance.py`
**Score range**: 0.0 to 1.0

Counts non-empty lines in the completion.

| Non-empty lines | Score |
|-----------------|-------|
| ≥ 5 | 1.0 |
| < 5 | count / 5 |

---

#### `ResponseStructureReward`
**File**: `reward_functions/format/response_structure.py`
**Score range**: 0.0 to 1.0

Regex-matches complete `<reasoning>...</reasoning>` and `<answer>...</answer>` blocks (content required between tags).

| Condition | Score |
|-----------|-------|
| Both blocks present | 1.0 |
| Only `<reasoning>` block | 0.5 |
| Only `<answer>` block | 0.5 |
| Neither | 0.0 |

---

## multi_hop_question_answering

Eight reward functions are composed for HotpotQA, FreshQA, and MuSiQue training: four correctness and four format.

> **Requires**: `OPENAI_API_KEY` must be set before running any pipeline in this module. All four correctness reward functions make LLM API calls via DeepEval or Evidently.

### Correctness

#### `DeepEvalGEvalRAGReward`
**File**: `reward_functions/correctness/deepeval_gevalrag.py`
**Score range**: 0.0 to 1.0
**Dependencies**: `deepeval`, OpenAI API

LLM-as-judge using DeepEval's `GEval` metric. Evaluates each completion against three RAG criteria:
- Factual accuracy: does the answer correctly address the multi-hop question?
- Supporting evidence: does the reasoning cite relevant evidence?
- Completeness: does the answer fully address all parts of the question?

Uses `kwargs["answer"]` as `expected_output`. Returns the normalised GEval score.

---

#### `DeepEvalAnswerRelevancyReward`
**File**: `reward_functions/correctness/deepeval_answer_relevancy.py`
**Score range**: 0.0 to 1.0
**Dependencies**: `deepeval`, OpenAI API

Uses DeepEval's `AnswerRelevancyMetric`. Measures how relevant the completion is to the original question — does not require a ground truth answer. Returns the metric score directly.

---

#### `DeepEvalSummarizationReward`
**File**: `reward_functions/correctness/deepeval_summarization.py`
**Score range**: 0.0 to 1.0
**Dependencies**: `deepeval`, OpenAI API

Uses DeepEval's `SummarizationMetric`. Measures faithfulness and coverage of the response as a summary of the reasoning chain. Returns the metric score directly.

---

#### `EvidentlyCorrectnessLLMReward`
**File**: `reward_functions/correctness/evidently_correctness_llm.py`
**Score range**: 0.0 or 1.0 (binary)
**Dependencies**: `evidently`, OpenAI API

Binary LLM judge using Evidently's `BinaryClassificationPromptTemplate`. Asks the judge: is this response correct and complete given the expected answer (`kwargs["answer"]`)? Returns 1.0 for correct, 0.0 for incorrect.

---

### Format

#### `ReasoningTagsReward`
**File**: `reward_functions/format/reasoning_tags.py`
**Score range**: 0.0 or 1.0 (binary)

> Note: different from the math_reasoning version. This is binary, not a per-tag count.

Returns 1.0 if the completion contains a complete `<reasoning>...</reasoning>` block (matched via regex, content required). Returns 0.0 otherwise.

---

#### `MultilineComplianceReward`
**File**: `reward_functions/format/multiline_compliance.py`
**Score range**: 0.0 to 1.0

> Note: threshold is 3 lines here, vs. 5 lines in math_reasoning.

| Non-empty lines | Score |
|-----------------|-------|
| ≥ 3 | 1.0 |
| < 3 | count / 3 |

---

#### `StructureValidationReward`
**File**: `reward_functions/format/structure_validation.py`
**Score range**: 0.0 to 1.0

Regex-matches complete `<reasoning>...</reasoning>` and `<answer>...</answer>` blocks.

| Condition | Score |
|-----------|-------|
| Both blocks present | 1.0 |
| Only one block | 0.5 |
| Neither | 0.0 |

---

#### `ResponseFormatReward`
**File**: `reward_functions/format/response_format.py`
**Score range**: 0.0 to 1.0

Scores based on response character length. Target range: 50–2000 characters.

| Length | Score |
|--------|-------|
| 50–2000 chars | 1.0 |
| < 50 chars | `length / 50 × 0.5` (max 0.5) |
| > 2000 chars | `2000 / length × 0.5` (max 0.5) |

---

## medical_question_answering

Identical reward function implementations to `multi_hop_question_answering`. The functions are **copied, not imported** — changes to one module's reward functions do not affect the other.

The only difference is in `DeepEvalGEvalRAGReward`: the GEval criteria string reads "correctly addresses the **medical** question" (vs. "multi-hop question" in multi_hop_question_answering).

> **Requires**: `OPENAI_API_KEY` must be set before running any pipeline in this module.

### Correctness

| Class | File | Score range | What it measures |
|----------|------|-------------|-----------------|
| `DeepEvalGEvalRAGReward` | `correctness/deepeval_gevalrag.py` | 0.0–1.0 | GEval RAG criteria with medical question framing |
| `DeepEvalAnswerRelevancyReward` | `correctness/deepeval_answer_relevancy.py` | 0.0–1.0 | AnswerRelevancyMetric |
| `DeepEvalSummarizationReward` | `correctness/deepeval_summarization.py` | 0.0–1.0 | SummarizationMetric |
| `EvidentlyCorrectnessLLMReward` | `correctness/evidently_correctness_llm.py` | 0.0 or 1.0 | Binary correct/incorrect |

### Format

| Class | File | Score range | What it measures |
|----------|------|-------------|-----------------|
| `ReasoningTagsReward` | `format/reasoning_tags.py` | 0.0 or 1.0 | Presence of `<reasoning>...</reasoning>` block |
| `MultilineComplianceReward` | `format/multiline_compliance.py` | 0.0–1.0 | ≥3 non-empty lines |
| `StructureValidationReward` | `format/structure_validation.py` | 0.0–1.0 | Both `<reasoning>` and `<answer>` blocks |
| `ResponseFormatReward` | `format/response_format.py` | 0.0–1.0 | Length 50–2000 characters |

See [multi_hop_question_answering](#multi_hop_question_answering) for full scoring details — the logic is identical.

---

## Reward callable signature

All reward callables follow the same `__call__` signature regardless of module:

```python
class MyReward(BaseReward):
    def __call__(self, prompts: list, completions: list, **kwargs) -> list[float]:
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            # compute score
            scores.append(score)
        return scores
```

`completions` is a list of lists: each inner list contains one dict with `{"role": "assistant", "content": "..."}`. Access the text via `completion[0]["content"]`.

`**kwargs` contains one key per additional dataset column. For example, if the dataset has an `answer` column, it is available as `kwargs["answer"]` — a list of ground truth values, one per completion.

---

## Adding a format reward function

**Example: add a new format reward to an existing GRPO module**

### 1. Create the file

```python
# src/llm_finetuning/math_reasoning/reward_functions/format/my_reward.py

from llm_finetuning.core import BaseReward, RewardConfig


class MyFormatReward(BaseReward):
    """Describe what this reward measures."""

    def __init__(self) -> None:
        super().__init__(RewardConfig(name="my_format_reward"))

    def __call__(self, prompts: list, completions: list, **kwargs) -> list[float]:
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            # compute score from response
            score = 1.0 if len(response) > 0 else 0.0
            scores.append(score)
        return scores
```

### 2. Register in `__init__.py`

```python
# math_reasoning/reward_functions/format/__init__.py
from llm_finetuning.math_reasoning.reward_functions.format.my_reward import MyFormatReward
```

### 3. Add to `reward_funcs` in `train.py`

```python
from llm_finetuning.math_reasoning.reward_functions.format.my_reward import (
    MyFormatReward,
)

# In GRPOTrainer call:
reward_funcs=[
    AnswerCorrectnessReward(),
    ReasoningTagsReward(),
    StepFormatReward(),
    MultilineComplianceReward(),
    ResponseStructureReward(),
    MyFormatReward(),       # add here
],
```

After adding a reward function, register it in the module's
`reward_functions/<type>/__init__.py` and add it to `reward_funcs=[...]` in `train.py`.

---

## Adding a DeepEval correctness reward

**Example: add a custom GEval correctness reward to a GRPO module**

Use this pattern when you want an LLM judge to score completions. Requires
`OPENAI_API_KEY`.

### 1. Create the file

```python
# src/llm_finetuning/multi_hop_question_answering/reward_functions/correctness/my_deepeval_reward.py

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from llm_finetuning.core import BaseReward, RewardConfig


MY_CRITERIA = (
    "The response correctly addresses the question with accurate, complete information."
)


class MyDeepEvalReward(BaseReward):
    """Score completions using a custom GEval criterion."""

    def __init__(self) -> None:
        super().__init__(RewardConfig(name="my_deepeval_reward"))

    def __call__(self, prompts: list, completions: list, **kwargs) -> list[float]:
        scores = []
        metric = GEval(
            name="MyReward",
            criteria=MY_CRITERIA,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
        )
        for i, completion in enumerate(completions):
            response = completion[0]["content"]
            question = prompts[i][-1]["content"]
            answer = kwargs["answer"][i]

            test_case = LLMTestCase(
                input=question,
                actual_output=response,
                expected_output=answer,
            )
            metric.measure(test_case)
            scores.append(metric.score if metric.score is not None else 0.0)
        return scores
```

### 2. Register in `__init__.py`

```python
# multi_hop_question_answering/reward_functions/correctness/__init__.py
from llm_finetuning.multi_hop_question_answering.reward_functions.correctness.my_deepeval_reward import (
    MyDeepEvalReward,
)
```

### 3. Add to `reward_funcs` in `train.py`

```python
from llm_finetuning.multi_hop_question_answering.reward_functions.correctness.my_deepeval_reward import (
    MyDeepEvalReward,
)

trainer = GRPOTrainer(
    reward_funcs=[
        DeepEvalGEvalRAGReward(),
        MyDeepEvalReward(),       # add here
        ...
    ],
    ...
)
```

### 4. Set the API key before running

```bash
export OPENAI_API_KEY="your-key"
python src/llm_finetuning/multi_hop_question_answering/grpo/hotpotqa/train.py
```

---

## Unit testing guidance for reward functions

- Create small `prompts`/`completions` fixtures that mirror the TRL structure:
  `completions = [[{"role": "assistant", "content": "..."}]]`
- Assert the returned list has length equal to `len(completions)`.
- Test edge cases: empty content, missing tags, boundary values for numeric scores.
- For LLM-judge rewards (`DeepEval`, `Evidently`), mock the external API call in unit
  tests to avoid requiring `OPENAI_API_KEY` in CI.
