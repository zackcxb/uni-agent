# Framework and Runner Naming Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename the Gateway-backed framework and the private orchestration helpers so their names describe ownership and behavior without changing runner, reward, Gateway, or TransferQueue contracts.

**Architecture:** `GatewayAgentFramework` owns Gateway session lifecycle, runner dispatch/concurrency, `TaskResult` annotation, and TransferQueue serialization. Gateway adapters remain responsible for OpenAI/Anthropic wire protocols. The public `AgentRunner` protocol and serialized runner configuration remain unchanged; only internal registry-key locals become `runner_key`.

**Tech Stack:** Python, Ray, OmegaConf, pytest, pre-commit, dynamic fully-qualified imports.

---

## Approved naming decisions

| Existing symbol | New symbol | Compatibility impact |
|---|---|---|
| `OpenAICompatibleAgentFramework` | `GatewayAgentFramework` | Public import and default dynamic FQN change; explicit external `framework_class_fqn` users must migrate. No alias is added. |
| `_run_batch_to_tq` | `_run_batch_rollouts` | Private-only rename. |
| `_run_prompt_sessions_to_tq` | `_run_prompt_sessions` | Private-only rename. |
| `_run_session` | `_run_gateway_session` | Private-only rename. |
| internal `runner_name` locals | `runner_key` | Internal-only clarification; external `agent_name`, `agent_runners`, `runner_fqn`, and `runner_kwargs` remain unchanged. |

The following remain unchanged: `AgentRunner`, `_RunnerConfig`, `runner_registry`, `runner_fqn`, `runner_kwargs`, `_materialize_runner`, `_run_agent_runner_ray_task`, `_run_session_with_concurrency_limit`, scoring helpers, TQ writer, `AgentFrameworkRolloutAdapter`, `build_gateway_manager`, `build_agent_framework`, `run_task`, and `task_runner.py`.

## Implementation tasks

### Task 1: Update the framework class and private helper vocabulary

**Files:**
- Modify: `uni_agent/framework/framework.py`
- Modify: `uni_agent/framework/__init__.py`
- Modify: `uni_agent/framework/entry.py`

1. Rename the class, return annotation, error messages, and default dynamic FQN references.
2. Rename only the approved private helpers and update all internal calls.
3. Rename framework-local registry-key variables from `runner_name` to `runner_key`, while preserving external `agent_name` field handling and serialized config names.

### Task 2: Update direct test references

**Files:**
- Modify: `tests/uni_agent/framework/test_generate_sequences_on_cpu.py`

Update imports, class patches, direct private-helper calls, and keyword arguments. Keep test behavior and fixtures unchanged.

### Task 3: Update stale-name references and documentation

**Files:**
- Search all repository files excluding `verl/` and `.git/`.

Confirm the old class/helper names do not remain in production, tests, examples, docs, or dynamic FQNs. Preserve the documented external names `AgentRunner`, `run_task`, `agent_name`, `agent_runners`, `runner_fqn`, and `runner_kwargs`.

### Task 4: Verify and prepare the PR

Run:

```bash
pytest -q tests/uni_agent/framework/test_generate_sequences_on_cpu.py tests/uni_agent/framework/test_task_runner.py
pre-commit run --all-files --show-diff-on-failure --color=always
python tests/special_sanity/check_pr_title.py  # with PR_TITLE set
python ~/.codex/skills/prepare-uni-agent-pr/scripts/check_pr_readiness.py \
  --repo /home/cxb/rl_framework/uni-agent/.worktrees/reward-flow-pr \
  --base upstream/main \
  --title '[framework] refactor: clarify Gateway framework and runner names' \
  --body /tmp/uni-agent-framework-naming-pr-body.md
```

Run the framework evolution ledger skill in read-only scan mode and record `ledger_update: none` unless the scan finds a training-visible behavior change. Inspect the final diff for `verl/` changes, unrelated files, co-author trailers, stale names, and public/config churn.

