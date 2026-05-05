"""Abstract base class for TRL-compatible reward functions."""

from __future__ import annotations

import asyncio
import functools
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, Callable, TypeVar


_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Configuration for naming a reward function in TRL logs."""

    name: str = "reward"


class BaseReward(ABC):
    """Synchronous reward callable for TRL's ``GRPOTrainer``.

    Subclasses implement ``__call__`` to return one scalar reward per generated
    completion. Instances can be passed directly to ``reward_funcs=[...]``.

    TRL logs rewards by callable name, so this class mirrors the configured
    name onto ``self.__name__`` and exposes the class docstring via
    ``self.__doc__``.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        """Initialize the reward and expose a stable `__name__` for logging."""
        self.config = config or RewardConfig()
        self.__name__ = self.config.name
        self.__doc__ = self.__class__.__doc__

    @abstractmethod
    def __call__(self, prompts: list, completions: list, **kwargs: Any) -> list[float]:
        """Return one reward score for each generated completion.

        Args:
            prompts: Batch of prompt histories, where each item is a list of
                chat messages leading up to the assistant turn.
            completions: Batch of generated assistant message lists aligned
                with ``prompts``. Access response text via
                ``completion[0]["content"]``.
            **kwargs: Extra per-row dataset columns forwarded by
                ``GRPOTrainer``, such as ``answer`` for reference-based
                rewards.

        Returns:
            Reward values ordered to match ``completions``. GRPO normalises
            scores within each generation group to compute relative
            advantages.
        """
        raise NotImplementedError

    def as_fn(self) -> Callable[..., list[float]]:
        """Return a plain function wrapper for pickling or strict type checks.

        Returns:
            A plain function with the same signature as ``__call__`` and the
            same ``__name__`` as this reward. Use this when a framework
            requires a picklable callable or rejects non-function callables.
        """

        def reward_func(prompts: list, completions: list, **kwargs: Any) -> list[float]:
            return self(prompts, completions, **kwargs)

        functools.update_wrapper(reward_func, self.__call__)
        reward_func.__name__ = self.config.name
        return reward_func

    def _run_batch(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run an internal batch coroutine behind the synchronous reward API.

        TRL expects reward functions to be plain synchronous callables. Async
        rewards should expose their concurrency via a private coroutine and use
        this helper at the sync boundary in ``__call__``.
        """
        return asyncio.run(coro)
