import sys
from types import SimpleNamespace

import pytest

from uni_agent.agents import AgentConfig, AgentResult
from uni_agent.sandbox import ExecResult, SandboxConfig
from uni_agent.tasks.swe_bench_multilingual.task import (
    SWEBenchMultilingualTask,
    SWEBenchMultilingualTaskConfig,
)
from uni_agent.tasks.terminal_bench import task as terminal_bench_module
from uni_agent.tasks.terminal_bench.task import TerminalBenchTask, TerminalBenchTaskConfig


class _Sandbox:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def exec_shell(self, *args, **kwargs):
        return ExecResult(exit_code=0, stdout="", stderr="")


class _Agent:
    async def run(self, **kwargs):
        return AgentResult(info={"steps": 1}, finished=False)


@pytest.mark.asyncio
async def test_swe_bench_multilingual_propagates_finished(monkeypatch):
    async def compute_reward(*args, **kwargs):
        return {"resolved": True}

    monkeypatch.setitem(
        sys.modules,
        "uni_agent.tasks.swe_bench_multilingual.reward",
        SimpleNamespace(compute_reward=compute_reward),
    )
    task = SWEBenchMultilingualTask(
        SWEBenchMultilingualTaskConfig(
            sandbox=SandboxConfig(provider="local"),
            agent=AgentConfig(),
            metadata={"instance_id": "case-1"},
        )
    )
    monkeypatch.setattr(task, "build_sandbox", _Sandbox)
    monkeypatch.setattr(task, "build_agent", _Agent)

    result = await task.run()

    assert result.finished is False


@pytest.mark.asyncio
async def test_terminal_bench_propagates_finished(monkeypatch):
    async def compute_reward(*args, **kwargs):
        return {"reward": 1.0, "resolved": True}

    monkeypatch.setattr(terminal_bench_module, "build_terminal_bench_sandbox_config", lambda config, metadata: config)
    monkeypatch.setattr(terminal_bench_module, "build_sandbox", lambda config: _Sandbox())
    monkeypatch.setattr("uni_agent.tasks.terminal_bench.reward.compute_reward", compute_reward)
    task = TerminalBenchTask(
        TerminalBenchTaskConfig(
            sandbox=SandboxConfig(provider="local"),
            agent=AgentConfig(),
            metadata={
                "instance_id": "case-1",
                "dataset_version": "2.0",
                "agent_timeout": 30,
                "verifier_timeout": 30,
            },
        )
    )
    monkeypatch.setattr(task, "build_agent", _Agent)

    result = await task.run()

    assert result.finished is False
