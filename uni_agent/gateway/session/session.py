"""Per-session gateway state, generation envelope, and lifecycle handling."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import HTTPException

from uni_agent.gateway.session.codec import MessageCodec
from uni_agent.gateway.session.types import InternalGenerationRequest, SessionHandle, Trajectory


logger = logging.getLogger(__name__)


class SessionPhase(str, Enum):
    """Lifecycle state for a gateway session.

    Attributes:
        ACTIVE: The session can accept generation and reward-info requests.
        FINALIZED: Final trajectories were returned and the session is closed.
        ABORTED: The session was cancelled and should not produce trajectories.
    """

    ACTIVE = "ACTIVE"
    FINALIZED = "FINALIZED"
    ABORTED = "ABORTED"


@dataclass
class LastTurnStart:
    """Stable lengths at the previous request delta start.

    This is the rollback target for a latest-turn rewrite. The truncation
    boundary, prefix-comparison tolerance boundary, and masked re-encode start
    are intentionally the same turn-level boundary.
    """

    response_ids_len: int
    response_mask_len: int
    response_logprobs_len: int
    message_history_len: int
    image_data_len: int
    video_data_len: int


@dataclass
class TrajectoryBuffer:
    """Mutable token buffer for the active trajectory under construction.

    Attributes:
        prompt_ids: Prompt token IDs for the current trajectory.
        response_ids: Accumulated response-side token IDs.
        response_mask: Labels aligned with ``response_ids``; ``1`` for model
            output and ``0`` for continuation context tokens.
        response_logprobs: Log probabilities aligned with ``response_ids`` when
            present; continuation context tokens use ``0.0``.
        last_turn_start: Stable lengths captured before the latest request-side
            delta was encoded; used as the rollback boundary for one-turn
            rewrites.
    """

    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    last_turn_start: LastTurnStart | None = None


@dataclass
class EncodedData:
    """Session-private data prepared before backend generation.

    The session uses this as the handoff between input preparation, backend
    generation, and the commit step. It is not an actor/runtime API.

    Attributes:
        buffer: Working trajectory buffer that becomes active only after commit.
        context_ids: Token IDs sent to the inference backend.
        sampling_params: Sampling params after request merge and budget clamp.
        messages: Normalized request messages snapshotted for commit.
        tools: Tool schemas used for both encoding and response decoding.
        image_data: Image inputs carried into backend generation and trajectory
            materialization.
        video_data: Video inputs carried into backend generation and trajectory
            materialization.
        materialized_trajectory: Previous active trajectory to append when the
            request changes context.
        length_exhausted_trajectory: Materialized trajectory for a length-budget
            early return, or ``None`` on the normal path.
    """

    buffer: TrajectoryBuffer
    context_ids: list[int]
    sampling_params: dict[str, Any]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    image_data: list[Any] | None
    video_data: list[Any] | None
    materialized_trajectory: Trajectory | None
    length_exhausted_trajectory: Trajectory | None


@dataclass
class GenerationOutcome:
    """Business result returned by ``GatewaySession.run_generation``.

    The session emits this instead of an HTTP response dict. ``_GatewayActor``
    passes it to the provider adapter for wire response serialization.

    Attributes:
        assistant_msg: Decoded assistant message, or an empty assistant message
            for length-exhausted early returns.
        finish_reason: Finish reason returned to the actor for serialization.
        prompt_tokens: Number of context tokens sent to the backend.
        completion_tokens: Number of generated response tokens.
    """

    assistant_msg: dict[str, Any]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class PrefixAlignmentResult:
    """Session-private prefix comparison classification.

    ``kind`` values:
    - ``EXACT_PREFIX``: base-canonicalized history matches the request prefix.
    - ``EQUIVALENT_PREFIX``: recipe normalizer makes the two prefixes match.
    - ``ROLLBACK_AND_REENCODE_CONTEXT``: latest-turn rewrite; truncate to the
      last turn start, re-encode the current request suffix as masked context.
    - ``SPLIT``: current request must start a materialized context split.

    Scheme R keeps recipe normalizers opaque to gateway code. Equivalent results
    therefore carry no reason tags; debug events record only the alignment kind.
    """

    kind: Literal["EXACT_PREFIX", "EQUIVALENT_PREFIX", "ROLLBACK_AND_REENCODE_CONTEXT", "SPLIT"]
    split_reason: str | None
    first_diff: dict[str, Any] | None


class GatewaySession:
    """Behavior-bearing state container for one gateway session.

    ``_GatewayActor`` owns instances of this class, calls ``run_generation`` for
    chat requests, and delegates lifecycle operations here. The session owns the
    conversation state and trajectory materialization, while the actor owns
    HTTP routing and provider response serialization.
    """

    def __init__(
        self,
        handle: SessionHandle,
        codec: MessageCodec,
        *,
        prompt_length: int | None = None,
        response_length: int | None = None,
        prefix_normalizer_fqn: str | None = None,
    ):
        """Create an active session bound to a handle and model codec."""
        self.handle = handle
        self._codec = codec
        self._prompt_length = prompt_length
        self._response_length = response_length
        self._prefix_normalizer = self._load_prefix_normalizer(prefix_normalizer_fqn)
        self.active_tool_schemas: list[dict[str, Any]] | None = None
        # Faithful record of the client-sent messages: prefix-comparison baseline,
        # length source for incremental slicing / num_turns. Never fed to encoding
        # (token truth lives in active_trajectory); comparison normalizes it on the
        # fly without persisting the lossy normalized form.
        self.message_history: list[dict[str, Any]] = []
        self.image_data: list[Any] | None = None
        self.video_data: list[Any] | None = None
        self.active_trajectory: TrajectoryBuffer | None = None
        self.trajectories: list[Trajectory] = []
        self.reward_info: dict[str, Any] = {}
        self.phase = SessionPhase.ACTIVE
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.request_lock = asyncio.Lock()
        # Serializes in-flight generation against the single-active state model.
        # This is an implementation detail, not GatewaySession's public contract.
        self.generation_lock = asyncio.Lock()
        self._debug_event_index = 0

    def _debug_dump_enabled(self) -> bool:
        return bool(os.getenv("UNI_AGENT_GATEWAY_DEBUG_DIR"))

    def _debug_include_payloads(self) -> bool:
        return os.getenv("UNI_AGENT_GATEWAY_DEBUG_LEVEL", "prefix").lower() == "full"

    def _debug_max_string_chars(self) -> int:
        raw = os.getenv("UNI_AGENT_GATEWAY_DEBUG_MAX_STRING_CHARS")
        if raw is None:
            return 4096
        try:
            return max(0, int(raw))
        except ValueError:
            return 4096

    def _debug_json_safe(self, value: Any, max_string_chars: int) -> Any:
        if isinstance(value, dict):
            return {str(k): self._debug_json_safe(v, max_string_chars) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [self._debug_json_safe(v, max_string_chars) for v in value]
        if isinstance(value, str):
            if len(value) > max_string_chars:
                omitted = len(value) - max_string_chars
                return f"{value[:max_string_chars]}...<truncated {omitted} chars>"
            return value
        if isinstance(value, int | float | bool) or value is None:
            return value
        return repr(value)

    def _debug_append_event(self, event: dict[str, Any]) -> None:
        debug_dir = os.getenv("UNI_AGENT_GATEWAY_DEBUG_DIR")
        if not debug_dir:
            return
        try:
            if not self._debug_include_payloads():
                event = {k: v for k, v in event.items() if k not in {"messages", "tools", "assistant_msg"}}
            path = Path(debug_dir) / "gateway" / f"{self.handle.session_id}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "schema_version": 1,
                "event_index": self._debug_event_index,
                "session_id": self.handle.session_id,
                "created_at": time.time(),
                **event,
            }
            self._debug_event_index += 1
            with path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(self._debug_json_safe(record, self._debug_max_string_chars()), ensure_ascii=False)
                    + "\n"
                )
        except Exception:
            logger.debug("Failed to write gateway debug event", exc_info=True)

    def _load_prefix_normalizer(self, fqn: str | None) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
        """Load a recipe prefix normalizer by FQN.

        Returns the raw callable, or ``None`` when no FQN is configured or the
        import fails (falls back to strict comparison). Call-site safety
        (deep-copy input, exceptions become mismatches) is applied where the
        normalizer is used, in ``_compute_prefix_alignment``.
        """
        if not fqn:
            return None

        from verl.utils.import_utils import load_class_from_fqn

        try:
            normalizer = load_class_from_fqn(fqn, description="prefix normalizer")
        except Exception:
            logger.error(
                "Failed to load prefix normalizer FQN '%s', falling back to strict",
                fqn,
                exc_info=True,
            )
            return None
        if not callable(normalizer):
            logger.error(
                "Prefix normalizer FQN '%s' did not resolve to a callable, falling back to strict",
                fqn,
            )
            return None
        return normalizer

    def _compute_prefix_alignment(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> PrefixAlignmentResult:
        """Compute typed prefix alignment with base canon plus optional recipe normalization."""
        if self.active_trajectory is None:
            return PrefixAlignmentResult(kind="SPLIT", split_reason="new_trajectory", first_diff=None)
        if self.active_tool_schemas != tools:
            return PrefixAlignmentResult(kind="SPLIT", split_reason="tool_schema_mismatch", first_diff=None)
        history = self.message_history

        base_history = [self._codec.canonicalize_message_for_prefix_comparison(m) for m in history]
        base_request = [self._codec.canonicalize_message_for_prefix_comparison(m) for m in messages]
        base_request_prefix = base_request[: len(history)]
        if len(messages) >= len(history) and base_history == base_request_prefix:
            return PrefixAlignmentResult(kind="EXACT_PREFIX", split_reason=None, first_diff=None)

        policy_history = base_history
        policy_request = base_request
        if self._prefix_normalizer is not None:
            # Deep-copy each message so the recipe normalizer is free to mutate
            # its copy without affecting the base-canonical payloads; treat any
            # normalizer exception as a mismatch (fall through to SPLIT) rather
            # than crashing the session.
            try:
                policy_history = [self._prefix_normalizer(copy.deepcopy(m)) for m in base_history]
                policy_request = [self._prefix_normalizer(copy.deepcopy(m)) for m in base_request]
                if len(messages) >= len(history) and policy_history == policy_request[: len(history)]:
                    return PrefixAlignmentResult(kind="EQUIVALENT_PREFIX", split_reason=None, first_diff=None)
            except Exception:
                logger.warning(
                    "Prefix normalizer raised exception, treating as mismatch",
                    exc_info=True,
                )
                return PrefixAlignmentResult(
                    kind="SPLIT",
                    split_reason="message_prefix_mismatch",
                    first_diff=None,
                )

        compare_len = min(len(history), len(messages))
        # Length divergence is folded into div instead of short-circuiting.
        # A shorter request is the normal signal for latest-turn deletion.
        content_diff = next(
            (
                i
                for i, (history_msg, request_msg) in enumerate(
                    zip(policy_history[:compare_len], policy_request[:compare_len], strict=False)
                )
                if history_msg != request_msg
            ),
            None,
        )
        if content_diff is not None:
            div = content_diff
        elif len(messages) < len(history):
            div = len(messages)
        else:
            div = None

        # first_diff carries full message bodies and is only consumed by debug
        # output, so compute it only when debug dumping is enabled.
        first_diff = None
        if div is not None and self._debug_dump_enabled():
            first_diff_index = next(
                (
                    i
                    for i, (history_msg, request_msg) in enumerate(
                        zip(base_history[:compare_len], base_request[:compare_len], strict=False)
                    )
                    if history_msg != request_msg
                ),
                None,
            )
            if first_diff_index is not None:
                first_diff = {
                    "index": first_diff_index,
                    "history": base_history[first_diff_index],
                    "request": base_request[first_diff_index],
                }
            elif len(messages) < len(history):
                first_diff = {
                    "index": len(messages),
                    "history": base_history[len(messages)],
                    "request": None,
                }

        if div is None:
            return PrefixAlignmentResult(
                kind="SPLIT",
                split_reason="message_prefix_mismatch",
                first_diff=first_diff,
            )

        last_turn_start = self.active_trajectory.last_turn_start
        if last_turn_start is None:
            return PrefixAlignmentResult(
                kind="SPLIT",
                split_reason="missing_last_turn_start",
                first_diff=first_diff,
            )

        stable_prefix_len = last_turn_start.message_history_len
        # The comparison tolerance boundary is exactly the truncation boundary:
        # any divergence at or after last_turn_start can be deleted and
        # re-encoded as mask=0 context; anything before it is a deeper rewrite.
        if div >= stable_prefix_len:
            return PrefixAlignmentResult(
                kind="ROLLBACK_AND_REENCODE_CONTEXT",
                split_reason=None,
                first_diff=first_diff,
            )

        return PrefixAlignmentResult(
            kind="SPLIT",
            split_reason="deeper_than_last_turn",
            first_diff=first_diff,
        )

    async def run_generation(self, request: InternalGenerationRequest, backend) -> GenerationOutcome:
        """Run one provider-normalized generation request and return its business outcome.

        The backend is passed in for this call only; the session does not own the
        backend lifecycle. The actor/provider adapter has already lowered the
        wire payload to the internal canonical request; session never sees raw
        wire payloads. Protocol capability checks happen in the actor before
        this method, while backend errors are converted into HTTP exceptions
        here.
        """
        async with self.generation_lock:
            async with self.request_lock:
                if self.phase != SessionPhase.ACTIVE:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Session {self.handle.session_id} is {self.phase.value.lower()}",
                    )
                encoded = await self._prepare_generation_inputs(request)
                if encoded.length_exhausted_trajectory is not None:
                    empty_msg = {"role": "assistant", "content": ""}
                    self.trajectories.append(encoded.length_exhausted_trajectory)
                    self.active_trajectory = None
                    self.message_history = list(encoded.messages) + [empty_msg]
                    self.image_data = list(encoded.image_data) if encoded.image_data is not None else None
                    self.video_data = list(encoded.video_data) if encoded.video_data is not None else None
                    self.active_tool_schemas = encoded.tools
                    self._touch()
                    self._debug_append_event(
                        {
                            "event": "generation_result",
                            "path": "length_exhausted",
                            "assistant_msg": empty_msg,
                            "finish_reason": "length",
                            "prompt_tokens": len(encoded.context_ids),
                            "completion_tokens": 0,
                            "message_history_len_after": len(self.message_history),
                            "num_materialized_trajectories": len(self.trajectories),
                        }
                    )
                    return GenerationOutcome(
                        assistant_msg=empty_msg,
                        finish_reason="length",
                        prompt_tokens=len(encoded.context_ids),
                        completion_tokens=0,
                    )

            try:
                output = await backend.generate(
                    request_id=self.handle.session_id,
                    prompt_ids=encoded.context_ids,
                    sampling_params=encoded.sampling_params,
                    image_data=encoded.image_data,
                    video_data=encoded.video_data,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"{e.__class__.__name__}: {e}") from e

            had_response_without_logprobs = bool(encoded.buffer.response_ids) and not encoded.buffer.response_logprobs
            response_ids = list(output.token_ids)
            encoded.buffer.response_ids.extend(response_ids)
            encoded.buffer.response_mask.extend([1] * len(response_ids))
            if output.log_probs is not None:
                log_probs = list(output.log_probs)
                assert len(log_probs) == len(response_ids), "backend logprobs must align with token_ids"
                if not had_response_without_logprobs:
                    encoded.buffer.response_logprobs.extend(log_probs)
            elif encoded.buffer.response_logprobs:
                raise RuntimeError("backend missing logprob mid-session")
            self._assert_response_logprob_alignment(encoded.buffer)

            assistant_msg, finish_reason = await self._codec.decode_response(
                response_ids,
                tools=encoded.tools,
                stop_reason=output.stop_reason,
            )

            async with self.request_lock:
                if self.phase != SessionPhase.ACTIVE:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Session {self.handle.session_id} is {self.phase.value.lower()}",
                    )
                if encoded.materialized_trajectory is not None:
                    self.trajectories.append(encoded.materialized_trajectory)
                self.active_trajectory = encoded.buffer
                self.message_history = list(encoded.messages) + [assistant_msg]
                self.image_data = list(encoded.image_data) if encoded.image_data is not None else None
                self.video_data = list(encoded.video_data) if encoded.video_data is not None else None
                self.active_tool_schemas = encoded.tools
                self._touch()
                self._debug_append_event(
                    {
                        "event": "generation_result",
                        "path": "normal",
                        "assistant_msg": assistant_msg,
                        "finish_reason": finish_reason,
                        "backend_stop_reason": output.stop_reason,
                        "prompt_tokens": len(encoded.context_ids),
                        "completion_tokens": len(response_ids),
                        "message_history_len_after": len(self.message_history),
                        "num_materialized_trajectories": len(self.trajectories),
                    }
                )
                return GenerationOutcome(
                    assistant_msg=assistant_msg,
                    finish_reason=finish_reason,
                    prompt_tokens=len(encoded.context_ids),
                    completion_tokens=len(response_ids),
                )

    async def _prepare_generation_inputs(self, request: InternalGenerationRequest) -> EncodedData:
        messages = request["messages"]
        tools = request["tools"]
        materialized_trajectory = None
        image_data = None
        video_data = None
        prefix_alignment = self._compute_prefix_alignment(messages, tools)
        stable_prefix_len = (
            self.active_trajectory.last_turn_start.message_history_len
            if self.active_trajectory is not None and self.active_trajectory.last_turn_start is not None
            else None
        )
        rollback_dropped_tokens = 0
        rollback_dropped_trainable_tokens = 0
        rollback_from_response_len = None
        rollback_to_response_len = None
        masked_reencoded_tokens = 0

        if self.active_trajectory is None:
            image_data, video_data = await self._codec.extract_multi_modal_data(messages)
            prompt_ids = self._codec.encode_full(
                messages,
                tools=tools,
                image_data=image_data,
                video_data=video_data,
            )
            buffer = TrajectoryBuffer(prompt_ids=prompt_ids)
            self._init_new_turn_start(
                buffer=buffer,
                messages=messages,
                image_data=image_data,
                video_data=video_data,
            )
        elif prefix_alignment.kind in {"EXACT_PREFIX", "EQUIVALENT_PREFIX"}:
            buffer = self._copy_trajectory_buffer(self.active_trajectory)
            self._assert_response_logprob_alignment(buffer)
            image_data = list(self.image_data) if self.image_data is not None else None
            video_data = list(self.video_data) if self.video_data is not None else None
            buffer.last_turn_start = self._snapshot_last_turn_start(
                buffer=buffer,
                message_history_len=len(self.message_history),
                image_data=image_data,
                video_data=video_data,
            )
            incremental_messages = messages[len(self.message_history) :]
            if incremental_messages:
                new_image_data, new_video_data = await self._codec.extract_multi_modal_data(incremental_messages)
                if new_image_data:
                    if image_data is None:
                        image_data = []
                    image_data.extend(new_image_data)
                if new_video_data:
                    if video_data is None:
                        video_data = []
                    video_data.extend(new_video_data)
                incremental_ids = self._codec.encode_incremental(
                    incremental_messages,
                    image_data=new_image_data,
                    video_data=new_video_data,
                )
                masked_reencoded_tokens = len(incremental_ids)
                if (
                    self._response_length is not None
                    and len(buffer.response_mask) + len(incremental_ids) >= self._response_length
                ):
                    context_ids = buffer.prompt_ids + buffer.response_ids
                    self._debug_append_event(
                        {
                            "event": "prepare_generation",
                            "path": "length_exhausted",
                            "alignment_kind": prefix_alignment.kind,
                            "split_reason": prefix_alignment.split_reason,
                            "first_diff": prefix_alignment.first_diff,
                            "messages": messages,
                            "tools": tools,
                            "history_len_before": len(self.message_history),
                            "request_len": len(messages),
                            "stable_prefix_len": stable_prefix_len,
                            "prompt_len": len(buffer.prompt_ids),
                            "response_len": len(buffer.response_ids),
                            "response_mask_len": len(buffer.response_mask),
                            "incremental_len": len(incremental_ids),
                            "context_len": len(context_ids),
                            "response_length_limit": self._response_length,
                            "rollback_dropped_tokens": rollback_dropped_tokens,
                            "rollback_dropped_trainable_tokens": rollback_dropped_trainable_tokens,
                            "rollback_from_response_len": rollback_from_response_len,
                            "rollback_to_response_len": rollback_to_response_len,
                            "masked_reencoded_tokens": masked_reencoded_tokens,
                            "sampling_params": {},
                        }
                    )
                    return EncodedData(
                        buffer=buffer,
                        context_ids=context_ids,
                        sampling_params={},
                        messages=list(messages),
                        tools=tools,
                        image_data=image_data,
                        video_data=video_data,
                        materialized_trajectory=None,
                        length_exhausted_trajectory=self._build_materialized_trajectory(
                            active=buffer,
                            extra_fields={"materialization_reason": "max_response_length"},
                        ),
                    )
                buffer.response_ids.extend(incremental_ids)
                buffer.response_mask.extend([0] * len(incremental_ids))
                if buffer.response_logprobs:
                    buffer.response_logprobs.extend([0.0] * len(incremental_ids))
                self._assert_response_logprob_alignment(buffer)
        elif prefix_alignment.kind == "ROLLBACK_AND_REENCODE_CONTEXT":
            last_turn_start = self.active_trajectory.last_turn_start
            assert last_turn_start is not None
            buffer = self._copy_trajectory_buffer(self.active_trajectory)
            self._assert_response_logprob_alignment(buffer)

            rollback_from_response_len = len(buffer.response_ids)
            rollback_to_response_len = last_turn_start.response_ids_len
            rollback_dropped_tokens = rollback_from_response_len - rollback_to_response_len
            rollback_dropped_trainable_tokens = sum(buffer.response_mask[last_turn_start.response_mask_len :])

            assert last_turn_start.response_ids_len <= len(buffer.response_ids)
            assert last_turn_start.response_mask_len <= len(buffer.response_mask)
            assert last_turn_start.response_logprobs_len <= len(buffer.response_logprobs)
            del buffer.response_ids[last_turn_start.response_ids_len :]
            del buffer.response_mask[last_turn_start.response_mask_len :]
            del buffer.response_logprobs[last_turn_start.response_logprobs_len :]
            self._assert_response_logprob_alignment(buffer)

            stored_image_data = list(self.image_data or [])
            stored_video_data = list(self.video_data or [])
            multimodal_safe = (
                last_turn_start.image_data_len <= len(stored_image_data)
                and last_turn_start.video_data_len <= len(stored_video_data)
            )
            boundary_image_data = stored_image_data[: last_turn_start.image_data_len] or None
            boundary_video_data = stored_video_data[: last_turn_start.video_data_len] or None
            image_data = list(boundary_image_data) if boundary_image_data is not None else None
            video_data = list(boundary_video_data) if boundary_video_data is not None else None
            suffix_messages = messages[last_turn_start.message_history_len :]
            suffix_ids: list[int] = []

            if multimodal_safe and suffix_messages:
                new_image_data, new_video_data = await self._codec.extract_multi_modal_data(suffix_messages)
                suffix_ids = self._codec.encode_incremental(
                    suffix_messages,
                    image_data=new_image_data,
                    video_data=new_video_data,
                )
                multimodal_safe = self._codec.validate_multi_modal_placeholders(
                    suffix_messages,
                    suffix_ids,
                    new_image_data,
                    new_video_data,
                )
                if multimodal_safe:
                    if new_image_data:
                        if image_data is None:
                            image_data = []
                        image_data.extend(new_image_data)
                    if new_video_data:
                        if video_data is None:
                            video_data = []
                        video_data.extend(new_video_data)
            elif multimodal_safe:
                multimodal_safe = self._codec.validate_multi_modal_placeholders(
                    suffix_messages,
                    suffix_ids,
                    None,
                    None,
                )

            if not multimodal_safe:
                prefix_alignment = PrefixAlignmentResult(
                    kind="SPLIT",
                    split_reason="multimodal_rollback_unsafe",
                    first_diff=prefix_alignment.first_diff,
                )
                rollback_dropped_tokens = 0
                rollback_dropped_trainable_tokens = 0
                rollback_from_response_len = None
                rollback_to_response_len = None
                masked_reencoded_tokens = 0
                materialized_trajectory = self._build_materialized_trajectory(active=self.active_trajectory)
                image_data, video_data = await self._codec.extract_multi_modal_data(messages)
                prompt_ids = self._codec.encode_full(
                    messages,
                    tools=tools,
                    image_data=image_data,
                    video_data=video_data,
                )
                buffer = TrajectoryBuffer(prompt_ids=prompt_ids)
                self._init_new_turn_start(
                    buffer=buffer,
                    messages=messages,
                    image_data=image_data,
                    video_data=video_data,
                )
            else:
                # Truncate deletes the latest turn. The current suffix is then
                # re-encoded as mask=0 context; only the backend output from
                # this request will be trainable mask=1.
                buffer.last_turn_start = self._snapshot_last_turn_start(
                    buffer=buffer,
                    message_history_len=last_turn_start.message_history_len,
                    image_data=boundary_image_data,
                    video_data=boundary_video_data,
                )
                masked_reencoded_tokens = len(suffix_ids)
                if (
                    self._response_length is not None
                    and len(buffer.response_mask) + len(suffix_ids) >= self._response_length
                ):
                    context_ids = buffer.prompt_ids + buffer.response_ids
                    self._debug_append_event(
                        {
                            "event": "prepare_generation",
                            "path": "length_exhausted",
                            "alignment_kind": prefix_alignment.kind,
                            "split_reason": prefix_alignment.split_reason,
                            "first_diff": prefix_alignment.first_diff,
                            "messages": messages,
                            "tools": tools,
                            "history_len_before": len(self.message_history),
                            "request_len": len(messages),
                            "stable_prefix_len": stable_prefix_len,
                            "prompt_len": len(buffer.prompt_ids),
                            "response_len": len(buffer.response_ids),
                            "response_mask_len": len(buffer.response_mask),
                            "incremental_len": len(suffix_ids),
                            "context_len": len(context_ids),
                            "response_length_limit": self._response_length,
                            "rollback_dropped_tokens": rollback_dropped_tokens,
                            "rollback_dropped_trainable_tokens": rollback_dropped_trainable_tokens,
                            "rollback_from_response_len": rollback_from_response_len,
                            "rollback_to_response_len": rollback_to_response_len,
                            "masked_reencoded_tokens": masked_reencoded_tokens,
                            "sampling_params": {},
                        }
                    )
                    return EncodedData(
                        buffer=buffer,
                        context_ids=context_ids,
                        sampling_params={},
                        messages=list(messages),
                        tools=tools,
                        image_data=image_data,
                        video_data=video_data,
                        materialized_trajectory=None,
                        length_exhausted_trajectory=self._build_materialized_trajectory(
                            active=buffer,
                            extra_fields={"materialization_reason": "max_response_length"},
                        ),
                    )
                buffer.response_ids.extend(suffix_ids)
                buffer.response_mask.extend([0] * len(suffix_ids))
                if buffer.response_logprobs:
                    buffer.response_logprobs.extend([0.0] * len(suffix_ids))
                self._assert_response_logprob_alignment(buffer)
        else:
            materialized_trajectory = self._build_materialized_trajectory(active=self.active_trajectory)
            image_data, video_data = await self._codec.extract_multi_modal_data(messages)
            prompt_ids = self._codec.encode_full(
                messages,
                tools=tools,
                image_data=image_data,
                video_data=video_data,
            )
            buffer = TrajectoryBuffer(prompt_ids=prompt_ids)
            self._init_new_turn_start(
                buffer=buffer,
                messages=messages,
                image_data=image_data,
                video_data=video_data,
            )

        context_ids = buffer.prompt_ids + buffer.response_ids
        sampling_params = dict(request["sampling_params"])
        remaining_response_budget = (
            self._response_length - len(buffer.response_mask) if self._response_length is not None else None
        )
        if remaining_response_budget is not None and "max_tokens" in sampling_params:
            sampling_params["max_tokens"] = min(sampling_params["max_tokens"], remaining_response_budget)
        path = "new_trajectory"
        if prefix_alignment.kind in {"EXACT_PREFIX", "EQUIVALENT_PREFIX"}:
            path = "prefix_incremental"
        elif prefix_alignment.kind == "ROLLBACK_AND_REENCODE_CONTEXT":
            path = "prefix_rollback"
        elif materialized_trajectory is not None:
            path = "context_split"
        self._debug_append_event(
            {
                "event": "prepare_generation",
                "path": path,
                "alignment_kind": prefix_alignment.kind,
                "split_reason": prefix_alignment.split_reason,
                "first_diff": prefix_alignment.first_diff,
                "messages": messages,
                "tools": tools,
                "history_len_before": len(self.message_history),
                "request_len": len(messages),
                "stable_prefix_len": stable_prefix_len,
                "prompt_len": len(buffer.prompt_ids),
                "response_len": len(buffer.response_ids),
                "response_mask_len": len(buffer.response_mask),
                "context_len": len(context_ids),
                "remaining_response_budget": remaining_response_budget,
                "rollback_dropped_tokens": rollback_dropped_tokens,
                "rollback_dropped_trainable_tokens": rollback_dropped_trainable_tokens,
                "rollback_from_response_len": rollback_from_response_len,
                "rollback_to_response_len": rollback_to_response_len,
                "masked_reencoded_tokens": masked_reencoded_tokens,
                "sampling_params": sampling_params,
            }
        )
        return EncodedData(
            buffer=buffer,
            context_ids=context_ids,
            sampling_params=sampling_params,
            messages=list(messages),
            tools=tools,
            image_data=image_data,
            video_data=video_data,
            materialized_trajectory=materialized_trajectory,
            length_exhausted_trajectory=None,
        )

    async def set_reward_info(self, reward_info: dict[str, Any] | None = None) -> None:
        """Store session-level reward metadata without closing the session."""
        async with self.request_lock:
            if self.phase != SessionPhase.ACTIVE:
                raise RuntimeError(f"Session {self.handle.session_id} is {self.phase.value.lower()}")
            if reward_info is not None:
                self.reward_info = dict(reward_info)
            self._touch()

    async def finalize(self) -> list[Trajectory]:
        """Close the session and return its materialized trajectories with rewards."""
        async with self.request_lock:
            if self.phase == SessionPhase.ABORTED:
                raise RuntimeError(f"Session {self.handle.session_id} is aborted")
            if self.phase == SessionPhase.FINALIZED:
                raise RuntimeError(f"Session {self.handle.session_id} is finalized")
            self._touch()
            self._materialize_active_trajectory()
            self.phase = SessionPhase.FINALIZED
            self._touch()
            return [replace(trajectory, reward_info=dict(self.reward_info)) for trajectory in self.trajectories]

    async def abort(self) -> None:
        """Abort the session and prevent further generation."""
        async with self.request_lock:
            if self.phase == SessionPhase.ABORTED:
                return
            if self.phase == SessionPhase.FINALIZED:
                raise RuntimeError(f"Session {self.handle.session_id} is finalized")
            self.phase = SessionPhase.ABORTED
            self._touch()

    def snapshot_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for actor state inspection."""
        return {
            "session_id": self.handle.session_id,
            "phase": self.phase.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "num_trajectories": len(self.trajectories),
            "has_active_trajectory": self.active_trajectory is not None,
        }

    def _copy_trajectory_buffer(self, buffer: TrajectoryBuffer) -> TrajectoryBuffer:
        return TrajectoryBuffer(
            prompt_ids=list(buffer.prompt_ids),
            response_ids=list(buffer.response_ids),
            response_mask=list(buffer.response_mask),
            response_logprobs=list(buffer.response_logprobs),
            last_turn_start=copy.copy(buffer.last_turn_start),
        )

    def _assert_response_logprob_alignment(self, buffer: TrajectoryBuffer) -> None:
        assert len(buffer.response_logprobs) in {
            0,
            len(buffer.response_ids),
        }, "response_logprobs must be empty or aligned with response_ids"

    def _snapshot_last_turn_start(
        self,
        *,
        buffer: TrajectoryBuffer,
        message_history_len: int,
        image_data: list[Any] | None,
        video_data: list[Any] | None,
    ) -> LastTurnStart:
        return LastTurnStart(
            response_ids_len=len(buffer.response_ids),
            response_mask_len=len(buffer.response_mask),
            response_logprobs_len=len(buffer.response_logprobs),
            message_history_len=message_history_len,
            image_data_len=len(image_data or []),
            video_data_len=len(video_data or []),
        )

    def _init_new_turn_start(
        self,
        *,
        buffer: TrajectoryBuffer,
        messages: list[dict[str, Any]],
        image_data: list[Any] | None,
        video_data: list[Any] | None,
    ) -> None:
        buffer.last_turn_start = self._snapshot_last_turn_start(
            buffer=buffer,
            message_history_len=len(messages),
            image_data=image_data,
            video_data=video_data,
        )

    def _materialize_active_trajectory(self) -> None:
        active = self.active_trajectory
        if active is None:
            return

        self._touch()
        self.trajectories.append(self._build_materialized_trajectory(active=active))
        self.active_trajectory = None

    def _build_materialized_trajectory(
        self,
        *,
        active: TrajectoryBuffer,
        extra_fields: dict[str, Any] | None = None,
    ) -> Trajectory:
        return Trajectory(
            prompt_ids=list(active.prompt_ids),
            response_ids=list(active.response_ids),
            response_mask=list(active.response_mask),
            response_logprobs=list(active.response_logprobs) if active.response_logprobs else None,
            reward_info={},
            num_turns=self._count_chat_turns(self.message_history),
            multi_modal_data=self._build_multi_modal_trajectory_data(self.image_data, self.video_data),
            extra_fields=dict(extra_fields) if extra_fields else {},
        )

    def _count_chat_turns(self, message_history: list[dict[str, Any]]) -> int:
        return sum(1 for m in message_history if m.get("role") in ("user", "assistant")) + 1

    def _build_multi_modal_trajectory_data(
        self,
        image_data: list[Any] | None,
        video_data: list[Any] | None,
    ) -> dict[str, Any] | None:
        multi_modal_data: dict[str, Any] = {}
        if image_data:
            multi_modal_data["images"] = list(image_data)
        if video_data:
            multi_modal_data["videos"] = list(video_data)
        return multi_modal_data or None

    def _touch(self) -> None:
        self.updated_at = time.time()
