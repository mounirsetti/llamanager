"""Tests for the Anthropic Messages facade in :mod:`llamanager.api_anthropic`.

The interesting surface here is the pure translation layer — both
directions (request body translation + response translation) plus the
streaming SSE translator. Each is covered without a live llama-server.

The transport-level paths (queue, upstream HTTP) are already exercised
by ``test_streaming_proxy.py`` and ``test_smoke.py``; we focus on the
shape mappings unique to the Anthropic surface.
"""
from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from llamanager.api_anthropic import (
    _AnthropicStreamTranslator,
    _build_openai_body,
    _flatten_system,
    _openai_to_anthropic_response,
    _translate_messages,
    _translate_tool_choice,
    _translate_tools,
)


# --------------------------------------------------------------------------
# request translation
# --------------------------------------------------------------------------

def test_flatten_system_string_and_blocks():
    assert _flatten_system(None) is None
    assert _flatten_system("hello") == "hello"
    assert _flatten_system([
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]) == "a\nb"


def test_flatten_system_rejects_other_types():
    with pytest.raises(HTTPException) as e:
        _flatten_system({"type": "text", "text": "x"})
    assert e.value.status_code == 400


def test_translate_messages_plain_strings_pass_through():
    out = _translate_messages([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
    ])
    assert out == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
    ]


def test_translate_messages_text_blocks_collapse_to_string():
    out = _translate_messages([
        {"role": "user", "content": [
            {"type": "text", "text": "part a"},
            {"type": "text", "text": " part b"},
        ]},
    ])
    assert out == [{"role": "user", "content": "part a part b"}]


def test_translate_messages_user_image_becomes_image_url():
    out = _translate_messages([
        {"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": "AAA",
            }},
        ]},
    ])
    assert len(out) == 1
    msg = out[0]
    assert msg["role"] == "user"
    # mixed text+image → list-shaped content (not collapsed to string)
    assert isinstance(msg["content"], list)
    kinds = [p["type"] for p in msg["content"]]
    assert kinds == ["text", "image_url"]
    assert msg["content"][1]["image_url"]["url"] == "data:image/png;base64,AAA"


def test_translate_messages_tool_result_splits_into_tool_role_message():
    """Anthropic puts tool_result blocks in the user message; OpenAI
    expects them as separate role=tool messages emitted BEFORE the
    user turn they accompany."""
    out = _translate_messages([
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "42"},
            {"type": "text", "text": "what next?"},
        ]},
    ])
    assert out == [
        {"role": "tool", "tool_call_id": "tu_1", "content": "42"},
        {"role": "user", "content": "what next?"},
    ]


def test_translate_messages_tool_result_is_error_prefixed():
    out = _translate_messages([
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1",
             "content": "bad input", "is_error": True},
        ]},
    ])
    assert out[0]["content"].startswith("[tool error]")


def test_translate_messages_assistant_tool_use_becomes_tool_calls():
    out = _translate_messages([
        {"role": "assistant", "content": [
            {"type": "text", "text": "calling now"},
            {"type": "tool_use", "id": "tu_1", "name": "get_weather",
             "input": {"city": "Paris"}},
        ]},
    ])
    assert len(out) == 1
    msg = out[0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "calling now"
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["id"] == "tu_1"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}


def test_translate_messages_assistant_tool_use_only_has_null_content():
    out = _translate_messages([
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_1", "name": "x", "input": {}},
        ]},
    ])
    assert out[0]["content"] is None
    assert len(out[0]["tool_calls"]) == 1


def test_translate_tools_wraps_in_function_envelope():
    out = _translate_tools([
        {"name": "get_weather", "description": "look up weather",
         "input_schema": {"type": "object",
                          "properties": {"city": {"type": "string"}}}},
    ])
    assert out == [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "look up weather",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}}},
        },
    }]


def test_translate_tools_rejects_server_side_tools():
    with pytest.raises(HTTPException) as e:
        _translate_tools([{"type": "web_search_20250305", "name": "web_search"}])
    assert e.value.status_code == 400
    assert "server-side" in e.value.detail


def test_translate_tool_choice_mappings():
    assert _translate_tool_choice(None) is None
    assert _translate_tool_choice({"type": "auto"}) == "auto"
    assert _translate_tool_choice({"type": "any"}) == "required"
    assert _translate_tool_choice({"type": "none"}) == "none"
    assert _translate_tool_choice({"type": "tool", "name": "x"}) == {
        "type": "function", "function": {"name": "x"},
    }


def test_build_openai_body_requires_max_tokens():
    with pytest.raises(HTTPException) as e:
        _build_openai_body({"messages": [{"role": "user", "content": "hi"}]})
    assert e.value.status_code == 400
    assert "max_tokens" in e.value.detail


def test_build_openai_body_requires_messages():
    with pytest.raises(HTTPException) as e:
        _build_openai_body({"max_tokens": 10})
    assert e.value.status_code == 400


def test_build_openai_body_prepends_system_message():
    out = _build_openai_body({
        "messages": [{"role": "user", "content": "hi"}],
        "system": "be terse",
        "max_tokens": 100,
    })
    assert out["messages"][0] == {"role": "system", "content": "be terse"}
    assert out["max_tokens"] == 100


def test_build_openai_body_forwards_sampling_params():
    out = _build_openai_body({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
        "stop_sequences": ["\n\n"],
    })
    assert out["temperature"] == 0.5
    assert out["top_p"] == 0.9
    assert out["top_k"] == 40
    assert out["stop"] == ["\n\n"]


def test_build_openai_body_streaming_requests_usage():
    out = _build_openai_body({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
        "stream": True,
    })
    assert out["stream"] is True
    assert out["stream_options"] == {"include_usage": True}


# --------------------------------------------------------------------------
# non-streaming response translation
# --------------------------------------------------------------------------

def test_openai_response_to_anthropic_text_only():
    upstream = {
        "id": "chatcmpl-1",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello!"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }
    out = _openai_to_anthropic_response(upstream, model_id="m")
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["model"] == "m"
    assert out["content"] == [{"type": "text", "text": "hello!"}]
    assert out["stop_reason"] == "end_turn"
    assert out["usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert out["id"].startswith("msg_")


def test_openai_response_length_maps_to_max_tokens():
    out = _openai_to_anthropic_response({
        "choices": [{"message": {"content": "x"},
                     "finish_reason": "length"}],
    }, model_id="m")
    assert out["stop_reason"] == "max_tokens"


def test_openai_response_tool_calls_become_tool_use_blocks():
    upstream = {
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 8},
    }
    out = _openai_to_anthropic_response(upstream, model_id="m")
    assert out["stop_reason"] == "tool_use"
    assert out["content"] == [{
        "type": "tool_use", "id": "call_1", "name": "get_weather",
        "input": {"city": "Paris"},
    }]


def test_openai_response_text_plus_tool_calls_yields_both_blocks():
    upstream = {
        "choices": [{
            "message": {
                "content": "let me check",
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "x", "arguments": "{}"},
                }],
            },
            "finish_reason": "stop",
        }],
    }
    out = _openai_to_anthropic_response(upstream, model_id="m")
    kinds = [b["type"] for b in out["content"]]
    assert kinds == ["text", "tool_use"]
    # Normalised to tool_use even though upstream said "stop".
    assert out["stop_reason"] == "tool_use"


def test_openai_response_with_no_choices_500s():
    with pytest.raises(HTTPException) as e:
        _openai_to_anthropic_response({"choices": []}, model_id="m")
    assert e.value.status_code == 502


def test_openai_response_with_malformed_tool_args_does_not_crash():
    out = _openai_to_anthropic_response({
        "choices": [{
            "message": {"content": None, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "x", "arguments": "not json"},
            }]},
            "finish_reason": "tool_calls",
        }],
    }, model_id="m")
    assert out["content"][0]["input"] == {"_raw": "not json"}


# --------------------------------------------------------------------------
# streaming translator
# --------------------------------------------------------------------------

def _parse_anthropic_events(chunks: list[bytes]) -> list[tuple[str, dict]]:
    """Parse the translator's output into (event_name, data_obj) pairs."""
    text = b"".join(chunks).decode("utf-8")
    out: list[tuple[str, dict]] = []
    current_event: str | None = None
    for line in text.splitlines():
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            payload = line.split(":", 1)[1].strip()
            out.append((current_event or "", json.loads(payload)))
            current_event = None
    return out


def _feed_chunks(translator: _AnthropicStreamTranslator,
                  openai_events: list[dict]) -> list[bytes]:
    out: list[bytes] = []
    for ev in openai_events:
        out.extend(translator.feed(
            f"data: {json.dumps(ev)}\n\n".encode("utf-8"),
        ))
    out.extend(translator.finalise())
    return out


def test_stream_translator_text_only():
    t = _AnthropicStreamTranslator(model_id="m")
    openai_events = [
        # First chunk: role marker only — no content yet.
        {"choices": [{"index": 0,
                      "delta": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "delta": {"content": "Hel"}}]},
        {"choices": [{"index": 0, "delta": {"content": "lo!"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
    ]
    events = _parse_anthropic_events(_feed_chunks(t, openai_events))
    names = [e for e, _ in events]
    assert names[0] == "message_start"
    assert "content_block_start" in names
    assert "content_block_delta" in names
    assert names[-2] == "message_delta"
    assert names[-1] == "message_stop"

    # message_delta should carry the final usage + stop_reason.
    md = [d for n, d in events if n == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "end_turn"
    assert md["usage"]["output_tokens"] == 2

    # Concatenated text deltas should equal "Hello!"
    text = "".join(d["delta"]["text"] for n, d in events
                   if n == "content_block_delta"
                   and d["delta"]["type"] == "text_delta")
    assert text == "Hello!"


def test_stream_translator_tool_use_emits_input_json_delta():
    t = _AnthropicStreamTranslator(model_id="m")
    openai_events = [
        # Tool-call kickoff with name + id.
        {"choices": [{"index": 0, "delta": {
            "tool_calls": [{"index": 0, "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather",
                                         "arguments": ""}}]}}]},
        # Partial JSON args streamed in pieces.
        {"choices": [{"index": 0, "delta": {
            "tool_calls": [{"index": 0,
                            "function": {"arguments": '{"city":'}}]}}]},
        {"choices": [{"index": 0, "delta": {
            "tool_calls": [{"index": 0,
                            "function": {"arguments": '"Paris"}'}}]}}]},
        {"choices": [{"index": 0, "delta": {},
                      "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 4, "completion_tokens": 5}},
    ]
    events = _parse_anthropic_events(_feed_chunks(t, openai_events))

    # The tool_use block_start event must carry name+id and an empty input.
    starts = [d for n, d in events if n == "content_block_start"]
    tool_starts = [s for s in starts
                   if s["content_block"]["type"] == "tool_use"]
    assert len(tool_starts) == 1
    assert tool_starts[0]["content_block"]["name"] == "get_weather"
    assert tool_starts[0]["content_block"]["id"] == "call_1"
    assert tool_starts[0]["content_block"]["input"] == {}

    # The partial_json deltas should concatenate back to valid JSON.
    json_pieces = [d["delta"]["partial_json"] for n, d in events
                   if n == "content_block_delta"
                   and d["delta"]["type"] == "input_json_delta"]
    assert json.loads("".join(json_pieces)) == {"city": "Paris"}

    # Stop reason on tool_calls → tool_use.
    md = [d for n, d in events if n == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "tool_use"


def test_stream_translator_interleaves_text_then_tool_use():
    """Text block must be closed before a tool_use block opens — blocks
    don't overlap in the Anthropic stream."""
    t = _AnthropicStreamTranslator(model_id="m")
    openai_events = [
        {"choices": [{"index": 0, "delta": {"content": "let me check"}}]},
        {"choices": [{"index": 0, "delta": {
            "tool_calls": [{"index": 0, "id": "c1",
                            "type": "function",
                            "function": {"name": "x",
                                         "arguments": "{}"}}]}}]},
        {"choices": [{"index": 0, "delta": {},
                      "finish_reason": "tool_calls"}]},
    ]
    events = _parse_anthropic_events(_feed_chunks(t, openai_events))
    # Order: message_start, text block_start, text delta, text block_stop,
    # tool_use block_start, (tool args delta), tool block_stop,
    # message_delta, message_stop.
    names = [n for n, _ in events]
    text_stop_idx = next(i for i, (n, d) in enumerate(events)
                         if n == "content_block_stop" and d["index"] == 0)
    tool_start_idx = next(i for i, (n, d) in enumerate(events)
                          if n == "content_block_start"
                          and d["content_block"]["type"] == "tool_use")
    assert text_stop_idx < tool_start_idx
    assert names[-1] == "message_stop"


def test_stream_translator_handles_upstream_error_event():
    t = _AnthropicStreamTranslator(model_id="m")
    chunk = (b'data: {"error": {"type": "upstream_error", '
             b'"message": "boom"}}\n\n')
    out = t.feed(chunk)
    assert t.upstream_error == {"type": "upstream_error", "message": "boom"}
    out.extend(t.finalise_with_error(t.upstream_error))
    events = _parse_anthropic_events(out)
    err_events = [d for n, d in events if n == "error"]
    assert len(err_events) == 1
    assert err_events[0]["error"]["message"] == "boom"
    # No message_stop after an error — the error terminates the stream.
    assert not any(n == "message_stop" for n, _ in events)


def test_stream_translator_ignores_keepalive_lines():
    t = _AnthropicStreamTranslator(model_id="m")
    out = t.feed(b": keepalive\n\ndata: [DONE]\n\n")
    assert out == []
    # finalise() still emits message_start (we never saw real content).
    finals = _parse_anthropic_events(t.finalise())
    names = [n for n, _ in finals]
    assert "message_start" in names
    assert names[-1] == "message_stop"
