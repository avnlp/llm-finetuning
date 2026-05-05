"""Pointwise reward model wrapper for PPO training."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PointwiseRewardModel:
    """OpenAssistant pointwise reward model for PPO training.

    Uses ``OpenAssistant/reward-model-deberta-v3-large-v2``, a DeBERTa-v2
    model trained on human preference data (WebGPT, Summarize from Feedback,
    Anthropic HH-RLHF) to produce a scalar quality score for a single
    (prompt, response) pair.
    """

    def __init__(
        self,
        model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2",
    ) -> None:
        """Load the reward model tokenizer and weights."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()

    def score(self, prompt: str, response: str) -> float:
        """Score a single response for a prompt (higher = better)."""
        inputs = self.tokenizer(
            prompt,
            response,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits[0].item()

    def score_batch(self, prompts: list[str], responses: list[str]) -> list[float]:
        """Score a batch of prompt-response pairs."""
        return [self.score(p, r) for p, r in zip(prompts, responses)]
