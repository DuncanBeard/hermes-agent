"""Copilot can change opaque item IDs between events at the same output_index."""

from types import SimpleNamespace as NS

import pytest

from agent.codex_responses_adapter import _normalize_codex_response
from agent.codex_runtime import _consume_codex_event_stream


@pytest.mark.parametrize("completion", ["deltas", "arguments_done", "item_done"])
def test_changing_item_ids_produce_one_call_with_complete_arguments(completion):
    events = [
        NS(type="response.output_item.added", output_index=0, item=NS(
            type="function_call", id="opaque_added", call_id="call_skill",
            name="skill_view", arguments="", status="in_progress")),
        NS(type="response.function_call_arguments.delta", output_index=0,
           item_id="opaque_delta_1", delta='{"name":'),
        NS(type="response.function_call_arguments.delta", output_index=0,
           item_id="opaque_delta_2", delta='"humanizer"}'),
    ]
    expected = '{"name":"humanizer"}'
    if completion == "arguments_done":
        expected = '{"name":"research"}'
        events.append(NS(type="response.function_call_arguments.done", output_index=0,
                         item_id="opaque_arguments_done", arguments=expected))
    if completion == "item_done":
        expected = '{"name":"research"}'
        events.append(NS(type="response.output_item.done", output_index=0, item=NS(
            type="function_call", id="opaque_done", call_id="call_skill",
            name="skill_view", arguments=expected, status="completed")))
    events.append(NS(type="response.completed", response=NS(status="completed", output=None)))

    final = _consume_codex_event_stream(events, model="test-model")
    message, _ = _normalize_codex_response(final)
    assert [(call.id, call.function.name, call.function.arguments) for call in message.tool_calls] == [
        ("call_skill", "skill_view", expected),
    ]
    # Identity matching must not rewrite the authoritative provider item.
    if completion == "item_done":
        assert final.output[0] is events[-2].item
        assert final.output[0].id == "opaque_done"


@pytest.mark.parametrize("confirm_second_call", [True, False])
def test_interleaved_calls_keep_arguments_and_announced_order(confirm_second_call):
    events = [
        NS(type="response.output_item.added", output_index=0, item=NS(
            type="function_call", id="added_a", call_id="call_a", name="skill_view", arguments="")),
        NS(type="response.output_item.added", output_index=1, item=NS(
            type="function_call", id="added_b", call_id="call_b", name="skill_view", arguments="")),
        NS(type="response.function_call_arguments.delta", output_index=1,
           item_id="delta_b", delta='{"name":'),
        NS(type="response.function_call_arguments.delta", output_index=0,
           item_id="delta_a", delta='{"name":"humanizer"}'),
        # Missing index: retain the existing item-ID fallback.
        NS(type="response.function_call_arguments.delta", item_id="added_b", delta='"research"}'),
        NS(type="response.output_item.done", output_index=1, item=NS(
            type="function_call", id="done_b", call_id="call_b", name="skill_view",
            arguments='{"name":"research"}', status="completed")),
        # No index here forces settlement to use first-observed sequence order.
        NS(type="response.output_item.done", item=NS(type="message", content=[])),
        NS(type="response.completed", response=NS(status="completed", output=None)),
    ]
    if not confirm_second_call:
        events = [event for event in events if not (
            event.type == "response.output_item.done" and getattr(event, "output_index", None) == 1
        )]
    final = _consume_codex_event_stream(events, model="test-model")
    assert [(item.type, getattr(item, "call_id", None)) for item in final.output] == [
        ("function_call", "call_a"), ("function_call", "call_b"), ("message", None),
    ]
    assert [item.arguments for item in final.output[:2]] == [
        '{"name":"humanizer"}', '{"name":"research"}',
    ]
