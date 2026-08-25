# Framework and Runner Naming Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clarify the Gateway-backed framework and private execution-helper names without changing runner, reward, Gateway, observability, or TransferQueue contracts.

**Architecture:** `GatewayAgentFramework` owns Gateway session lifecycle, runner dispatch/concurrency, trajectory reward handling, and TransferQueue serialization. Gateway adapters remain responsible for OpenAI/Anthropic wire protocols. The public `AgentRunner` protocol and serialized runner configuration remain unchanged; only internal registry-key locals become `runner_key`.

**Tech Stack:** Python, Ray, OmegaConf, pytest, pre-commit, dynamic fully-qualified imports.

---

## Approved naming decisions

| Existing symbol | New symbol | Compatibility impact |
|---|---|---|
| `OpenAICompatibleAgentFramework` | `GatewayAgentFramework` | Public import and default dynamic FQN change; explicit external `framework_class_fqn` users must migrate. No alias is added. |
| `_run_batch_to_tq` | `_run_batch_rollouts` | Private-only rename. |
| `_run_prompt_sessions_to_tq` | `_run_prompt_sessions` | Private-only rename. |
| `_run_session` | `_run_gateway_session` | Private-only rename. |
| internal `runner_name` locals | `runner_key` | Internal-only clarification; external `agent_name`, `agent_runners`, `runner_fqn`, `runner_kwargs`, and observability `runner_name` fields remain unchanged. |

The following remain unchanged: `AgentRunner`, `_RunnerConfig`, `runner_registry`, `runner_fqn`, `runner_kwargs`, `_materialize_runner`, `_run_agent_runner_ray_task`, `_run_session_with_concurrency_limit`, scoring helpers, TQ writer, `AgentFrameworkRolloutAdapter`, `build_gateway_manager`, `build_agent_framework`, `run_task`, and `task_runner.py`.

## Implementation tasks

### Task 1: Update the framework class and private helper vocabulary

**Files:**
- Modify: `uni_agent/framework/framework.py`
- Modify: `uni_agent/framework/__init__.py`
- Modify: `uni_agent/framework/entry.py`

Rename the class, return annotation, error messages, default dynamic FQN references, and approved private helpers. Rename only framework-local registry-key variables; preserve `session_trace.finish(runner_name=...)` as an observability field.

### Task 2: Update direct test references

**Files:**
- Modify: `tests/uni_agent/framework/test_generate_sequences_on_cpu.py`

Update imports, class patches, and descriptive text. Keep test behavior and fixtures unchanged.

### Task 3: Verify stale names and compatibility boundaries

Search all repository files excluding `verl/` and `.git/`. Confirm old class/helper names do not remain in production, tests, examples, docs, or dynamic FQNs. Preserve `AgentRunner`, `run_task`, `agent_name`, `agent_runners`, `runner_fqn`, and `runner_kwargs`.

### Task 4: Verify and prepare the PR

Run:

```bash
python -m pytest -q tests/uni_agent/framework/test_generate_sequences_on_cpu.py tests/uni_agent/framework/test_task_runner.py
pre-commit run --all-files --show-diff-on-failure --color=always
PR_TITLE='[framework] refactor: clarify Gateway framework and runner names' \
  python tests/special_sanity/check_pr_title.py
python ~/.codex/skills/prepare-uni-agent-pr/scripts/check_pr_readiness.py \
  --repo /home/cxb/rl_framework/uni-agent/.worktrees/reward-flow-pr \
  --base upstream/main \
  --title '[framework] refactor: clarify Gateway framework and runner names' \
  --body /tmp/uni-agent-framework-naming-pr-body.md
```

The dedicated evolution-ledger skill is unavailable in this environment. Record `ledger_update: none` in the PR body because this patch changes names only and does not change training-visible behavior; do not write remote ledger state.

