"""Tests for compact, cached thread transcript retrieval."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.load import dumps
from langchain_core.messages import AIMessage, HumanMessage
from mcp.server.fastmcp.exceptions import ToolError

from opik_mcp.read_list.thread_transcript import run_thread_transcript


class FakeOpikClient:
    def __init__(self, pages: dict[int, dict[str, Any]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    async def list_traces(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.pages[kwargs["page"]]


def _trace(
    trace_id: str,
    start_time: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": trace_id,
        "project_id": "p-1",
        "start_time": start_time,
        "input": input_data,
        "output": output_data,
        "metadata": {"created_from": "langchain", "keep": "complete"},
    }


@pytest.mark.anyio
async def test_transcript_pages_caches_full_traces_and_renders_in_time_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    later = _trace(
        "tr-2",
        "2026-08-01T12:01:00Z",
        {"messages": [[{"type": "human", "content": "How are you?"}]]},
        {"generations": [[{"text": "I am well."}]]},
    )
    earlier = _trace(
        "tr-1",
        "2026-08-01T12:00:00Z",
        {"messages": [{"type": "system", "content": "Be concise."}]},
        {
            "messages": [
                {"type": "system", "content": "Be concise."},
                {"type": "human", "content": "Hello"},
                {"type": "ai", "content": "Hi!"},
            ]
        },
    )
    first_page = [earlier] + [{"id": f"ignored-{index}"} for index in range(99)]
    client = FakeOpikClient(
        {
            1: {"content": first_page, "page": 1, "size": 100, "total": 101},
            2: {"content": [later], "page": 2, "size": 1, "total": 101},
        }
    )
    cached: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(
        "opik_mcp.read_list.thread_transcript.cache_objects",
        lambda _entity_type, traces, **_kwargs: cached.append(traces),
    )

    transcript = await run_thread_transcript(
        "thread-1", project_id="p-1", client=client
    )

    assert transcript == (
        "System: Be concise.\n\nUser: Hello\n\nAssistant: Hi!\n\n"
        "User: How are you?\n\nAssistant: I am well."
    )
    assert [page[0]["id"] for page in cached] == ["tr-1", "tr-2"]
    assert cached[0][0]["metadata"] == {"created_from": "langchain", "keep": "complete"}
    assert len(client.calls) == 2
    assert json.loads(client.calls[0]["filters"]) == [
        {"field": "thread_id", "operator": "=", "value": "thread-1"}
    ]
    assert json.loads(client.calls[0]["sorting"]) == [
        {"field": "start_time", "direction": "ASC"}
    ]


@pytest.mark.anyio
async def test_transcript_skips_non_langchain_traces_and_flattens_text_blocks() -> None:
    trace = _trace(
        "tr-1",
        "2026-08-01T12:00:00Z",
        {
            "messages": [
                {
                    "type": "human",
                    "content": [
                        {"type": "text", "text": "First"},
                        {"type": "text", "text": "second"},
                    ],
                }
            ]
        },
        {
            "messages": [
                {
                    "type": "tool",
                    "name": "search",
                    "tool_call_id": "call-1",
                    "content": "result",
                }
            ]
        },
    )
    unsupported = {**trace, "id": "tr-2", "metadata": {"created_from": "custom"}}
    client = FakeOpikClient(
        {1: {"content": [trace, unsupported], "page": 1, "size": 2, "total": 2}}
    )

    transcript = await run_thread_transcript(
        "thread-1", project_name="demo", client=client
    )

    assert transcript == "User: Firstsecond\n\nTool (search): result"


@pytest.mark.anyio
async def test_transcript_restores_native_langchain_serialized_messages() -> None:
    trace = _trace(
        "tr-1",
        "2026-08-01T12:00:00Z",
        {"messages": [json.loads(dumps(HumanMessage(content="Hello")))]},
        {"messages": [json.loads(dumps(AIMessage(content="Hi")))]},
    )
    client = FakeOpikClient({1: {"content": [trace], "page": 1, "size": 1, "total": 1}})

    transcript = await run_thread_transcript(
        "thread-1", project_id="p-1", client=client
    )

    assert transcript == "User: Hello\n\nAssistant: Hi"


@pytest.mark.anyio
async def test_transcript_uses_configured_input_and_output_paths() -> None:
    trace = _trace(
        "tr-1",
        "2026-08-01T12:00:00Z",
        {"input": {"messages": [{"type": "human", "content": "Hello"}]}},
        {"messages": [{"type": "ai", "content": "Hi"}]},
    )
    client = FakeOpikClient({1: {"content": [trace], "page": 1, "size": 1, "total": 1}})

    transcript = await run_thread_transcript(
        "thread-1",
        project_name="demo",
        input_key="input.input",
        output_key="output.messages",
        client=client,
    )

    assert transcript == "User: Hello\n\nAssistant: Hi"


@pytest.mark.anyio
async def test_transcript_requires_project_scope() -> None:
    with pytest.raises(ToolError, match="requires project_id or project_name"):
        await run_thread_transcript("thread-1", client=FakeOpikClient({}))
