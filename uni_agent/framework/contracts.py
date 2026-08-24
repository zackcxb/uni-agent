from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from uni_agent.gateway.session import SessionHandle
    from uni_agent.tasks import TaskResult


class AgentRunner(Protocol):
    """Callable that executes one agent episode against a Framework-owned Gateway session."""

    async def __call__(
        self,
        *,
        session: SessionHandle,
        raw_prompt: object,
        sample_index: int,
        **sample_runner_kwargs: object,
    ) -> TaskResult: ...
