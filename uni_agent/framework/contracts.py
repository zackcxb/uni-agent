from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from uni_agent.gateway.session import SessionHandle


@dataclass
class EpisodeResult:
    """Typed annotations returned by a managed Agent Runner."""

    reward: float | None = None
    metrics: dict[str, int | float | bool] = field(default_factory=dict)
    episode_finished: bool | None = None
    reward_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reward is not None:
            try:
                reward = float(self.reward)
            except (TypeError, ValueError) as exc:
                raise ValueError("EpisodeResult.reward must be a finite number or numeric string") from exc
            if not math.isfinite(reward):
                raise ValueError("EpisodeResult.reward must be a finite number or numeric string")
            self.reward = reward
        if self.episode_finished is not None and type(self.episode_finished) is not bool:
            raise ValueError("EpisodeResult.episode_finished must be a bool or None")
        if not isinstance(self.metrics, dict):
            raise ValueError("EpisodeResult.metrics must be a dict")
        for key, value in self.metrics.items():
            if not isinstance(key, str):
                raise ValueError("EpisodeResult.metrics keys must be strings")
            if key == "reward":
                raise ValueError("EpisodeResult.metrics key 'reward' is reserved for the outcome reward")
            if not isinstance(value, int | float | bool):
                raise ValueError(f"EpisodeResult.metrics[{key!r}] must be scalar (int/float/bool)")
        if not isinstance(self.reward_context, dict):
            raise ValueError("EpisodeResult.reward_context must be a dict")


class AgentRunner(Protocol):
    """Callable contract for OpenAI-compatible agent runners."""

    async def __call__(
        self,
        *,
        session: SessionHandle,
        raw_prompt: object,
        sample_index: int,
        **sample_runner_kwargs: object,
    ) -> EpisodeResult | None: ...
