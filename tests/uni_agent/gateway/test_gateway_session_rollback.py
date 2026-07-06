import pytest
import torch

from tests.uni_agent.support import FakeProcessor, FakeTokenizer, fake_vision_info_extractor
from uni_agent.gateway.session import GatewaySession, MessageCodec, SessionHandle, TrajectoryBuffer
from verl.workers.rollout.replica import TokenOutput


def _request(messages, *, tools=None, sampling_params=None):
    return {
        "messages": list(messages),
        "tools": tools,
        "sampling_params": dict(sampling_params or {}),
    }


class SequencedTokenBackend:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": dict(sampling_params),
                "image_data": image_data,
                "video_data": video_data,
            }
        )
        text, log_probs = self.steps.pop(0)
        token_ids = [ord(char) for char in text]
        if log_probs == "auto":
            log_probs = [-0.1] * len(token_ids)
        return TokenOutput(token_ids=token_ids, log_probs=log_probs, stop_reason="completed")


class NoImagePlaceholderProcessor(FakeProcessor):
    def __call__(self, **kwargs):
        output = super().__call__(**kwargs)
        input_ids = output["input_ids"][0].tolist()
        input_ids = [token_id for token_id in input_ids if token_id != self.image_token_id]
        output["input_ids"] = torch.tensor([input_ids], dtype=torch.long)
        return output


async def reversed_image_vision_info_extractor(messages, image_patch_size, config=None):
    image_data, video_data = await fake_vision_info_extractor(messages, image_patch_size=image_patch_size, config=config)
    if image_data is not None:
        image_data = list(reversed(image_data))
    return image_data, video_data


async def _run_two_turn_session(session: GatewaySession, backend: SequencedTokenBackend):
    first_messages = [{"role": "user", "content": "first"}]
    first = await session.run_generation(_request(first_messages), backend)
    second_messages = [
        first_messages[0],
        first.assistant_msg,
        {"role": "user", "content": "second"},
    ]
    second = await session.run_generation(_request(second_messages), backend)
    return first_messages, first, second_messages, second


@pytest.mark.asyncio
async def test_latest_turn_rewrite_rolls_back_and_reencodes_suffix_as_masked_context():
    codec = MessageCodec(FakeTokenizer())
    session = GatewaySession(SessionHandle("rollback-hit"), codec)
    events = []
    session._debug_append_event = events.append
    backend = SequencedTokenBackend(
        [
            ("FIRST", "auto"),
            ("OLD_SECOND", "auto"),
            ("RETRY", "auto"),
        ]
    )
    first_messages, first, second_messages, _second = await _run_two_turn_session(session, backend)

    turn_start = session.active_trajectory.last_turn_start
    response_before = list(session.active_trajectory.response_ids)
    mask_before = list(session.active_trajectory.response_mask)
    logprobs_before = list(session.active_trajectory.response_logprobs)
    third_messages = [
        first_messages[0],
        first.assistant_msg,
        second_messages[2],
        {"role": "user", "content": "please retry"},
    ]
    expected_suffix_ids = codec.encode_incremental(third_messages[turn_start.message_history_len :])
    expected_dropped_tokens = len(response_before) - turn_start.response_ids_len
    expected_dropped_trainable = sum(mask_before[turn_start.response_mask_len :])

    third = await session.run_generation(_request(third_messages), backend)

    prepare_events = [event for event in events if event["event"] == "prepare_generation"]
    assert prepare_events[-1]["path"] == "prefix_rollback"
    assert prepare_events[-1]["alignment_kind"] == "ROLLBACK_AND_REENCODE_CONTEXT"
    assert prepare_events[-1]["rollback_dropped_tokens"] == expected_dropped_tokens
    assert prepare_events[-1]["rollback_dropped_trainable_tokens"] == expected_dropped_trainable
    assert prepare_events[-1]["rollback_from_response_len"] == len(response_before)
    assert prepare_events[-1]["rollback_to_response_len"] == turn_start.response_ids_len
    assert prepare_events[-1]["masked_reencoded_tokens"] == len(expected_suffix_ids)

    buffer = session.active_trajectory
    assert session.trajectories == []
    assert buffer.response_ids[: turn_start.response_ids_len] == response_before[: turn_start.response_ids_len]
    assert buffer.response_mask[: turn_start.response_mask_len] == mask_before[: turn_start.response_mask_len]
    assert buffer.response_logprobs[: turn_start.response_logprobs_len] == logprobs_before[
        : turn_start.response_logprobs_len
    ]
    suffix_start = turn_start.response_ids_len
    suffix_end = suffix_start + len(expected_suffix_ids)
    assert buffer.response_ids[suffix_start:suffix_end] == expected_suffix_ids
    assert buffer.response_mask[suffix_start:suffix_end] == [0] * len(expected_suffix_ids)
    assert buffer.response_mask[suffix_end:] == [1] * len(third.assistant_msg["content"])
    assert len(buffer.response_logprobs) == len(buffer.response_ids)
    assert buffer.last_turn_start.response_ids_len == turn_start.response_ids_len
    assert buffer.last_turn_start.message_history_len == turn_start.message_history_len


@pytest.mark.asyncio
async def test_deeper_than_last_turn_change_splits_instead_of_rollback():
    session = GatewaySession(SessionHandle("rollback-deeper"), MessageCodec(FakeTokenizer()))
    backend = SequencedTokenBackend([("FIRST", "auto"), ("SECOND", "auto")])
    await _run_two_turn_session(session, backend)

    alignment = session._compute_prefix_alignment(
        [{"role": "user", "content": "changed stable prefix"}],
        tools=None,
    )

    assert alignment.kind == "SPLIT"
    assert alignment.split_reason == "deeper_than_last_turn"


@pytest.mark.asyncio
async def test_shorter_request_latest_turn_delete_enters_rollback_instead_of_history_longer_split():
    session = GatewaySession(SessionHandle("rollback-shorter"), MessageCodec(FakeTokenizer()))
    backend = SequencedTokenBackend([("FIRST", "auto"), ("SECOND", "auto")])
    await _run_two_turn_session(session, backend)
    turn_start = session.active_trajectory.last_turn_start

    alignment = session._compute_prefix_alignment(
        session.message_history[: turn_start.message_history_len],
        tools=None,
    )

    assert alignment.kind == "ROLLBACK_AND_REENCODE_CONTEXT"
    assert alignment.split_reason is None


@pytest.mark.asyncio
async def test_rollback_budget_exhaustion_returns_length_without_backend_call():
    codec = MessageCodec(FakeTokenizer())
    session = GatewaySession(SessionHandle("rollback-budget"), codec)
    backend = SequencedTokenBackend([("FIRST", "auto"), ("SECOND", "auto"), ("UNUSED", "auto")])
    first_messages, first, second_messages, _second = await _run_two_turn_session(session, backend)
    turn_start = session.active_trajectory.last_turn_start
    rollback_messages = [
        first_messages[0],
        first.assistant_msg,
        second_messages[2],
        {"role": "user", "content": "retry with too much context"},
    ]
    suffix_ids = codec.encode_incremental(rollback_messages[turn_start.message_history_len :])
    session._response_length = turn_start.response_mask_len + len(suffix_ids)

    outcome = await session.run_generation(_request(rollback_messages), backend)

    assert outcome.finish_reason == "length"
    assert len(backend.calls) == 2
    assert session.active_trajectory is None
    assert len(session.trajectories) == 1
    assert session.trajectories[0].extra_fields["materialization_reason"] == "max_response_length"


@pytest.mark.asyncio
async def test_rollback_multimodal_placeholder_mismatch_falls_back_to_split():
    processor = NoImagePlaceholderProcessor()
    session = GatewaySession(
        SessionHandle("rollback-mm-split"),
        MessageCodec(
            FakeTokenizer(),
            processor=processor,
            vision_info_extractor=fake_vision_info_extractor,
        ),
    )
    events = []
    session._debug_append_event = events.append
    backend = SequencedTokenBackend([("FIRST", "auto"), ("SECOND", "auto"), ("SPLIT", "auto")])
    first_messages, first, second_messages, _second = await _run_two_turn_session(session, backend)
    rollback_messages = [
        first_messages[0],
        first.assistant_msg,
        second_messages[2],
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "image://retry.png"}},
                {"type": "text", "text": "retry with image"},
            ],
        },
    ]

    await session.run_generation(_request(rollback_messages), backend)

    prepare_events = [event for event in events if event["event"] == "prepare_generation"]
    assert prepare_events[-1]["path"] == "context_split"
    assert prepare_events[-1]["alignment_kind"] == "SPLIT"
    assert prepare_events[-1]["split_reason"] == "multimodal_rollback_unsafe"
    assert len(session.trajectories) == 1
    assert session.image_data == ["image://retry.png"]


@pytest.mark.asyncio
async def test_rollback_multimodal_extracted_data_order_mismatch_falls_back_to_split():
    session = GatewaySession(
        SessionHandle("rollback-mm-order-split"),
        MessageCodec(
            FakeTokenizer(),
            processor=FakeProcessor(),
            vision_info_extractor=reversed_image_vision_info_extractor,
        ),
    )
    events = []
    session._debug_append_event = events.append
    backend = SequencedTokenBackend([("FIRST", "auto"), ("SECOND", "auto"), ("SPLIT", "auto")])
    first_messages, first, second_messages, _second = await _run_two_turn_session(session, backend)
    rollback_messages = [
        first_messages[0],
        first.assistant_msg,
        second_messages[2],
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "image://first.png"}},
                {"type": "image_url", "image_url": {"url": "image://second.png"}},
                {"type": "text", "text": "retry with ordered images"},
            ],
        },
    ]

    await session.run_generation(_request(rollback_messages), backend)

    prepare_events = [event for event in events if event["event"] == "prepare_generation"]
    assert prepare_events[-1]["path"] == "context_split"
    assert prepare_events[-1]["alignment_kind"] == "SPLIT"
    assert prepare_events[-1]["split_reason"] == "multimodal_rollback_unsafe"
    assert len(session.trajectories) == 1


@pytest.mark.asyncio
async def test_logprob_alignment_invariant_and_missing_backend_logprob_fail_loud():
    session = GatewaySession(SessionHandle("rollback-logprob"), MessageCodec(FakeTokenizer()))
    backend = SequencedTokenBackend([("FIRST", "auto"), ("SECOND", None)])
    first_messages = [{"role": "user", "content": "first"}]
    first = await session.run_generation(_request(first_messages), backend)
    second_messages = [
        first_messages[0],
        first.assistant_msg,
        {"role": "user", "content": "second"},
    ]

    with pytest.raises(RuntimeError, match="backend missing logprob mid-session"):
        await session.run_generation(_request(second_messages), backend)


@pytest.mark.asyncio
async def test_split_created_buffer_initializes_last_turn_start_so_next_latest_delete_can_rollback():
    session = GatewaySession(SessionHandle("rollback-after-split"), MessageCodec(FakeTokenizer()))
    events = []
    session._debug_append_event = events.append
    backend = SequencedTokenBackend([("FIRST", "auto"), ("SPLIT_BASE", "auto"), ("RETRY", "auto")])
    first_messages = [{"role": "user", "content": "first"}]
    await session.run_generation(_request(first_messages), backend)
    assert session.active_trajectory.last_turn_start.response_ids_len == 0
    assert session.active_trajectory.last_turn_start.message_history_len == len(first_messages)

    split_messages = [{"role": "user", "content": "replacement"}]
    split = await session.run_generation(_request(split_messages), backend)
    assert len(session.trajectories) == 1
    assert session.active_trajectory.last_turn_start.response_ids_len == 0
    assert session.active_trajectory.last_turn_start.message_history_len == len(split_messages)

    rollback_messages = [split_messages[0]]
    await session.run_generation(_request(rollback_messages), backend)

    prepare_events = [event for event in events if event["event"] == "prepare_generation"]
    assert prepare_events[-1]["path"] == "prefix_rollback"
    assert len(session.trajectories) == 1
    assert session.message_history == rollback_messages + [{"role": "assistant", "content": "RETRY"}]
    assert split.assistant_msg == {"role": "assistant", "content": "SPLIT_BASE"}


def test_missing_last_turn_start_splits_defensively():
    session = GatewaySession(SessionHandle("rollback-missing-lts"), MessageCodec(FakeTokenizer()))
    session.active_trajectory = TrajectoryBuffer(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
    )
    session.message_history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "old"},
    ]

    alignment = session._compute_prefix_alignment(
        [{"role": "user", "content": "first"}],
        tools=None,
    )

    assert alignment.kind == "SPLIT"
    assert alignment.split_reason == "missing_last_turn_start"
