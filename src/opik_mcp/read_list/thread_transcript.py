"""Compact, integration-aware transcript retrieval for conversation threads."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from mcp.server.fastmcp.exceptions import ToolError

from opik_mcp.config import Settings, get_settings
from opik_mcp.opik_client import (
    OpikAuthError,
    OpikNotFoundError,
    OpikServerError,
    OpikValidationError,
    make_opik_client,
)
from opik_mcp.store import cache_objects

_PAGE_SIZE = 100


class ThreadTranscriptClient(Protocol):
    """Minimal client surface required to retrieve a thread's trace pages."""

    async def list_traces(
        self,
        *,
        project_id: str | None = None,
        project_name: str | None = None,
        filters: str | None = None,
        sorting: str | None = None,
        truncate: bool | None = None,
        exclude: list[str] | None = None,
        page: int = 1,
        size: int = 10,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class _Message:
    role: str
    content: str
    name: str | None = None


def _as_mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _value_at_path(value: object, path: str) -> object | None:
    """Return the value at a dot-separated path, or ``None`` when absent."""
    current = value
    for key in path.split("."):
        if not key:
            return None
        mapping = _as_mapping(current)
        if mapping is None or key not in mapping:
            return None
        current = mapping[key]
    return current


def _render_langchain_content(message: Any) -> str | None:
    """Return LangChain's canonical text rendering for a ``BaseMessage``."""
    text = message.text
    return text if isinstance(text, str) and text.strip() else None


def _langchain_messages(value: object) -> list[_Message]:
    """Deserialize LangChain's persisted BaseMessage form and render its text.

    ``loads`` restores native LangChain serializations directly. The tracer also
    persists ``BaseMessage.model_dump()`` records, which are not themselves
    serializable manifests; those are normalized into a messages-only LangChain
    constructor manifest before they are loaded.
    """
    if not isinstance(value, list):
        return []
    try:
        from langchain_core.load import loads
    except ModuleNotFoundError as exc:
        raise ToolError(
            "LangChain transcript rendering requires the optional dependency. "
            "Install opik-mcp[langchain]."
        ) from exc

    class_names = {
        "human": "HumanMessage",
        "user": "HumanMessage",
        "ai": "AIMessage",
        "assistant": "AIMessage",
        "system": "SystemMessage",
        "tool": "ToolMessage",
        "function": "FunctionMessage",
        "chat": "ChatMessage",
    }
    rendered: list[_Message] = []
    for item in value:
        message = _as_mapping(item)
        if message is None:
            continue
        data = _as_mapping(message.get("data"))
        payload = dict(data or message)
        message_type = payload.get("type")
        class_name = (
            class_names.get(message_type) if isinstance(message_type, str) else None
        )
        manifest: dict[str, Any] = dict(message)
        if class_name is not None and "lc" not in manifest:
            manifest = {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", class_name],
                "kwargs": payload,
            }
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=r"The function `loads` is in beta.*"
                )
                message = loads(json.dumps(manifest), allowed_objects="messages")
        except (KeyError, TypeError, ValueError):
            continue
        if not hasattr(message, "content_blocks") or not hasattr(message, "type"):
            continue
        content = _render_langchain_content(message)
        if content is None:
            continue
        name = getattr(message, "name", None)
        rendered.append(
            _Message(
                role=message.type,
                content=content,
                name=name if isinstance(name, str) else None,
            )
        )
    return rendered


def _langchain_input_messages(value: object) -> list[_Message]:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return _langchain_messages(value[0])
        return _langchain_messages(value)
    data = _as_mapping(value)
    if data is None:
        return []
    nested = _as_mapping(data.get("input"))
    if nested is not None and not isinstance(data.get("messages"), list):
        data = nested
    messages = data.get("messages")
    if not isinstance(messages, list):
        return []
    if messages and isinstance(messages[0], list):
        return _langchain_messages(messages[0])
    return _langchain_messages(messages)


def _langchain_output_messages(value: object) -> list[_Message]:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return _langchain_messages(value[0])
        return _langchain_messages(value)
    data = _as_mapping(value)
    if data is None:
        return []
    messages = data.get("messages")
    if isinstance(messages, list) and (
        not messages or not isinstance(messages[0], list)
    ):
        return _langchain_messages(messages)

    generations = data.get("generations")
    if (
        not isinstance(generations, list)
        or not generations
        or not isinstance(generations[0], list)
    ):
        return []
    output: list[_Message] = []
    for generation in generations[0]:
        item = _as_mapping(generation)
        if item is None:
            continue
        message = _as_mapping(item.get("message"))
        kwargs = _as_mapping(message.get("kwargs")) if message is not None else None
        content = (kwargs or {}).get("content", item.get("text"))
        output.extend(_langchain_messages([{"type": "ai", "content": content}]))
    return output


def _fingerprint(message: _Message) -> tuple[str, str]:
    return (message.role, message.content)


def _combine_langchain_messages(
    input_data: object, output_data: object
) -> list[_Message]:
    """Match the UI behavior: an output state superseding input is not duplicated."""
    inputs = _langchain_input_messages(input_data)
    outputs = _langchain_output_messages(output_data)
    if len(outputs) >= len(inputs) and all(
        _fingerprint(left) == _fingerprint(right)
        for left, right in zip(inputs, outputs, strict=False)
    ):
        return outputs
    return [*inputs, *outputs]


def _render_message(message: _Message) -> str:
    labels = {
        "human": "User",
        "user": "User",
        "chat": "User",
        "ai": "Assistant",
        "assistant": "Assistant",
        "system": "System",
        "tool": "Tool",
        "function": "Function",
    }
    label = labels.get(message.role, message.role.title())
    if message.role in {"tool", "function"} and message.name:
        label = f"{label} ({message.name})"
    return f"{label}: {message.content}"


def _render_trace(
    trace: Mapping[str, Any], *, input_key: str, output_key: str
) -> list[str]:
    metadata = _as_mapping(trace.get("metadata"))
    if metadata is None or metadata.get("created_from") != "langchain":
        return []
    input_data = _value_at_path(trace, input_key)
    output_data = _value_at_path(trace, output_key)
    messages = _combine_langchain_messages(input_data, output_data)
    return [_render_message(message) for message in messages]


def _format_error(exc: BaseException) -> str:
    if isinstance(exc, OpikNotFoundError):
        return (
            "Thread traces were not found. Verify the thread and project identifiers."
        )
    if isinstance(exc, OpikAuthError):
        return "Permission denied while retrieving thread traces."
    if isinstance(exc, OpikValidationError):
        return "The backend rejected the thread trace query. Verify the project identifier."
    if isinstance(exc, OpikServerError):
        return (
            "The Opik backend failed while retrieving thread traces; retry the request."
        )
    return "Failed to retrieve thread traces."


async def run_thread_transcript(
    thread_id: str,
    *,
    project_id: str | None = None,
    project_name: str | None = None,
    input_key: str = "input",
    output_key: str = "output",
    settings: Settings | None = None,
    client: ThreadTranscriptClient | None = None,
) -> str:
    """Fetch, cache, and render every LangChain trace in a thread as text."""
    if project_id is None and project_name is None:
        raise ToolError("get_thread_transcript requires project_id or project_name.")
    if not input_key or not output_key:
        raise ToolError(
            "input_key and output_key must be non-empty dot-separated paths."
        )

    opik = (
        client if client is not None else make_opik_client(settings or get_settings())
    )
    filters = json.dumps([{"field": "thread_id", "operator": "=", "value": thread_id}])
    traces: list[dict[str, Any]] = []
    page = 1

    while True:
        try:
            response = await opik.list_traces(
                project_id=project_id,
                project_name=project_name,
                filters=filters,
                sorting=json.dumps([{"field": "start_time", "direction": "ASC"}]),
                page=page,
                size=_PAGE_SIZE,
            )
        except (
            OpikAuthError,
            OpikNotFoundError,
            OpikValidationError,
            OpikServerError,
        ) as exc:
            raise ToolError(_format_error(exc)) from exc

        content = response.get("content") or []
        page_traces = [item for item in content if isinstance(item, dict)]
        cache_objects("trace", page_traces, settings=settings)
        traces.extend(page_traces)

        total = response.get("total")
        if not page_traces or (isinstance(total, int) and page * _PAGE_SIZE >= total):
            break
        if len(page_traces) < _PAGE_SIZE:
            break
        page += 1

    lines: list[str] = []
    for trace in sorted(traces, key=lambda item: item.get("start_time") or ""):
        lines.extend(_render_trace(trace, input_key=input_key, output_key=output_key))
    return "\n\n".join(lines) if lines else "No renderable transcript messages found."


__all__ = ["run_thread_transcript"]
