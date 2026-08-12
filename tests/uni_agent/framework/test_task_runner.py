import pytest

from uni_agent.framework import EpisodeResult
from uni_agent.framework import task_runner as task_runner_module
from uni_agent.framework.task_runner import (
    _extract_upstream,
    _inject_gateway_tunnel,
    _rewrite_gateway_url,
    compute_score,
    run_task,
)
from uni_agent.gateway.session import SessionHandle
from uni_agent.tasks import TaskConfig, TaskResult


@pytest.mark.cpu
@pytest.mark.level0
def test_rewrite_gateway_url_replaces_host_with_tunnel_port():
    assert _rewrite_gateway_url("http://gateway.example:40169/sessions/abc/v1", 38197) == (
        "http://127.0.0.1:38197/sessions/abc/v1"
    )


@pytest.mark.cpu
@pytest.mark.level0
def test_rewrite_gateway_url_custom_proxy_port():
    assert _rewrite_gateway_url("http://gateway:8000/v1", 4242) == "http://127.0.0.1:4242/v1"


@pytest.mark.cpu
@pytest.mark.level0
def test_extract_upstream_returns_host_port():
    assert _extract_upstream("http://gateway.example:40169/sessions/abc/v1") == "gateway.example:40169"


@pytest.mark.cpu
@pytest.mark.level0
def test_extract_upstream_none_without_port():
    assert _extract_upstream("http://gateway/v1") is None


@pytest.mark.cpu
@pytest.mark.level0
def test_inject_gateway_tunnel_rewrites_upstream_and_base_url():
    task = {
        "sandbox": {"provider": "openyuanrong", "sandbox_kwargs": {"proxy_port": 38197, "image": "x"}},
        "agent": {"step_limit": 10},
    }
    merged = _inject_gateway_tunnel(task, "http://gateway.example:40169/sessions/abc/v1")

    assert merged["sandbox"]["sandbox_kwargs"]["upstream"] == "gateway.example:40169"
    assert merged["sandbox"]["sandbox_kwargs"]["proxy_port"] == 38197
    # The agent receives the tunnel-rewritten base_url; unrelated keys are preserved.
    assert merged["agent"]["model"]["base_url"] == "http://127.0.0.1:38197/sessions/abc/v1"
    assert merged["agent"]["step_limit"] == 10


@pytest.mark.cpu
@pytest.mark.level0
def test_inject_gateway_tunnel_raises_without_port():
    task = {"sandbox": {"provider": "openyuanrong", "sandbox_kwargs": {"proxy_port": 38197}}}
    with pytest.raises(ValueError, match="cannot derive gateway tunnel upstream"):
        _inject_gateway_tunnel(task, "http://gateway.example/v1")


@pytest.mark.cpu
@pytest.mark.level0
def test_inject_gateway_tunnel_rejects_non_yuanrong_sandbox():
    task = {"sandbox": {"provider": "local", "sandbox_kwargs": {"proxy_port": 38197}}}
    with pytest.raises(ValueError, match="supported only on 'openyuanrong'"):
        _inject_gateway_tunnel(task, "http://gateway.example:40169/v1")


@pytest.mark.cpu
@pytest.mark.level0
def test_task_result_positional_field_order():
    result = TaskResult(0.5, 1.0, False, {"reason": "limit"})

    assert result.reward == 0.5
    assert result.accuracy == 1.0
    assert result.episode_finished is False
    assert result.extra_info == {"reason": "limit"}


@pytest.mark.cpu
@pytest.mark.level0
def test_episode_result_separates_reward_status_metrics_and_context():
    result = EpisodeResult(
        reward=0.5,
        metrics={"acc": 1.0, "steps": 4},
        episode_finished=False,
        reward_context={"report": {"resolved": 1}},
    )

    assert result.reward == 0.5
    assert result.metrics == {"acc": 1.0, "steps": 4}
    assert result.episode_finished is False
    assert result.reward_context == {"report": {"resolved": 1}}


@pytest.mark.cpu
@pytest.mark.level0
def test_task_result_uses_episode_finished_name():
    result = TaskResult(reward=0.5, accuracy=1.0, episode_finished=False)

    assert result.episode_finished is False
    assert not hasattr(result, "finished")


@pytest.mark.cpu
@pytest.mark.level0
def test_episode_result_rejects_non_scalar_metrics():
    with pytest.raises(ValueError, match=r"metrics\['report'\] must be scalar"):
        EpisodeResult(metrics={"report": {"resolved": 1}})


@pytest.mark.parametrize(("reward", "expected"), [(True, 1.0), ("0.5", 0.5)])
def test_episode_result_normalizes_numeric_reward(reward, expected):
    assert EpisodeResult(reward=reward).reward == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("reward", ["not-a-number", float("inf")])
def test_episode_result_rejects_invalid_reward(reward):
    with pytest.raises(ValueError, match="finite number or numeric string"):
        EpisodeResult(reward=reward)  # type: ignore[arg-type]


def test_episode_result_rejects_reserved_reward_metric():
    with pytest.raises(ValueError, match="key 'reward' is reserved"):
        EpisodeResult(metrics={"reward": 0.5})


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_run_task_binds_raw_prompt_to_sample_task_config(monkeypatch, tmp_path):
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(
        """
- name: test_task
""".strip()
    )
    captured = {}

    class _FakeTask:
        def __init__(self, config):
            self.config = TaskConfig(
                name=config["name"],
                sandbox={"provider": "local"},
                prompt=config["prompt"],
                metadata=config["metadata"],
            )

        async def run(self):
            captured["config"] = self.config
            return TaskResult(reward=1.0, accuracy=1.0, episode_finished=True)

    monkeypatch.setattr(task_runner_module, "get_task", _FakeTask)
    source_prompt = [{"role": "user", "content": "Canonical source problem"}]

    await task_runner_module.run_task(
        session=SessionHandle(
            session_id="test-session",
            base_url="http://gateway/sessions/test/v1",
        ),
        raw_prompt=source_prompt,
        tools_kwargs={
            "task": {
                "name": "test_task",
                "metadata": {"problem_statement": "METADATA PROBLEM"},
            }
        },
        task_config_path=str(config_path),
    )

    assert captured["config"].prompt == source_prompt


@pytest.mark.asyncio
async def test_run_task_returns_episode_result(monkeypatch):
    task_result = TaskResult(
        reward=0.5,
        accuracy=1.0,
        episode_finished=False,
        extra_info={"report": {"resolved": 1}},
    )

    class _Resolver:
        def resolve(self, sample_config, runtime_model):
            assert sample_config == {"name": "stub"}
            assert runtime_model["base_url"] == "http://gateway/session/v1"
            return {"name": "stub"}

    class _Task:
        async def run(self):
            return task_result

    monkeypatch.setattr(task_runner_module, "TaskConfigResolver", _Resolver)
    monkeypatch.setattr(task_runner_module, "get_task", lambda task: _Task())

    result = await run_task(
        session=SessionHandle(session_id="session", base_url="http://gateway/session/v1"),
        tools_kwargs={"task": {"name": "stub"}},
    )

    assert result == EpisodeResult(
        reward=0.5,
        metrics={"acc": 1.0},
        episode_finished=False,
        reward_context={"report": {"resolved": 1}},
    )


def test_compute_score_passes_through_runner_reward_info():
    assert compute_score(
        data_source="stub",
        solution_str="unused",
        ground_truth="unused",
        extra_info={
            "runner_reward_info": {
                "reward": "0.5",
                "metrics": {"acc": True, "format": 0.8},
                "reward_context": {"trace": "unused by pass-through"},
            }
        },
    ) == {"score": 0.5, "acc": True, "format": 0.8}
