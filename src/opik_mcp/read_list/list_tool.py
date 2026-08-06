"""``list`` tool — paginated discovery of Opik entities.

Ported from ollie-assist's ``tools/list.py``. Output is a pipe-delimited
table (mirrors ollie's format) — easier for the LLM to scan than nested
JSON and lossless for the columns we care about (id, name, plus a few
entity-specific fields like ``created_at`` / ``dataset_name``).

Project-scoped lists (``trace``, ``test_suite_item``, ``prompt_version``)
require their parent id via ``project_id`` / ``test_suite_id`` /
``prompt_id`` — enforced via the registry's ``list_required_kwargs``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp.exceptions import ToolError

from opik_mcp.config import Settings, get_settings
from opik_mcp.opik_client import (
    OpikAuthError,
    OpikListClient,
    OpikNotFoundError,
    OpikServerError,
    OpikValidationError,
    make_opik_client,
)
from opik_mcp.read_list.errors import EntityArgValidationError
from opik_mcp.read_list.registry import ENTITY_REGISTRY, LISTABLE_TYPES, EntityHandler
from opik_mcp.store import cache_objects

logger = logging.getLogger("opik_mcp.read_list.list")

_MAX_SIZE = 100
_TRUNCATE_AT = 60


def _as_json_str(value: str | list[dict[str, Any]]) -> str:
    """Normalize a filters/sorting arg to the JSON string the client expects.

    MCP clients may send these clauses either as a JSON-encoded string (the
    REST convention) or as a native array (many clients coerce JSON-looking
    text into structured values). Accept both and forward a JSON string.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value)


async def run_list(
    entity_type: str,
    *,
    name: str | None = None,
    page: int = 1,
    size: int = 25,
    project_id: str | None = None,
    project_name: str | None = None,
    test_suite_id: str | None = None,
    prompt_id: str | None = None,
    filters: str | list[dict[str, Any]] | None = None,
    sorting: str | list[dict[str, Any]] | None = None,
    search: str | None = None,
    from_time: str | None = None,
    to_time: str | None = None,
    settings: Settings | None = None,
    client: OpikListClient | None = None,
) -> str:
    """List tool entrypoint. See ``server.py`` for the registered tool."""
    handler = ENTITY_REGISTRY.get(entity_type)
    if handler is None or handler.list_fn is None:
        valid = ", ".join(sorted(LISTABLE_TYPES))
        err = EntityArgValidationError(f"Cannot list {entity_type!r}. Listable types: {valid}")
        raise ToolError(str(err)) from err

    size = max(1, min(size, _MAX_SIZE))
    page = max(1, page)

    kw: dict[str, Any] = {"page": page, "size": size}
    if name:
        kw["name"] = name
    if project_id is not None:
        kw["project_id"] = project_id
    # Only project-scoped lists (trace, thread) take project_name; forwarding it
    # to a workspace-wide list_fn (projects/experiments/…) would be an unexpected
    # kwarg. Gate on the same signal the required-check uses.
    if project_name is not None and "project_id" in handler.list_required_kwargs:
        kw["project_name"] = project_name
    if test_suite_id is not None:
        kw["test_suite_id"] = test_suite_id
    if prompt_id is not None:
        kw["prompt_id"] = prompt_id
    # Server-side narrowing params. Both trace and thread lists accept
    # ``filters`` + ``sorting`` (see OpikListClient.list_traces/list_threads);
    # ``search`` and ``from_time``/``to_time`` are backed only by the threads
    # endpoint. Forward each only for the entities that accept it — other
    # list_fns would reject the unexpected kwargs.
    if entity_type in ("thread", "trace"):
        if filters is not None:
            kw["filters"] = _as_json_str(filters)
        if sorting is not None:
            kw["sorting"] = _as_json_str(sorting)
    if entity_type == "thread":
        if search is not None:
            kw["search"] = search
        if from_time is not None:
            kw["from_time"] = from_time
        if to_time is not None:
            kw["to_time"] = to_time

    for required in handler.list_required_kwargs:
        if kw.get(required) is None:
            # project_name is an accepted alternative to project_id for the
            # project-scoped lists (trace, thread) — the client methods take
            # either, so don't force the UUID when a name was given.
            if required == "project_id" and kw.get("project_name"):
                continue
            hint = f"{required} (or project_name)" if required == "project_id" else required
            err = EntityArgValidationError(
                f"list({entity_type!r}) requires {hint}. "
                f"E.g. list({entity_type!r}, {required}='<uuid>', …)."
            )
            raise ToolError(str(err)) from err

    opik = client if client is not None else make_opik_client(settings or get_settings())

    try:
        page_body = await handler.list_fn(opik, **kw)
    except (
        OpikAuthError,
        OpikNotFoundError,
        OpikValidationError,
        OpikServerError,
    ) as e:
        raise ToolError(f"Failed to list {entity_type}s: {e}") from e

    content_raw = page_body.get("content") or []
    content: list[dict[str, Any]] = [it for it in content_raw if isinstance(it, dict)]
    total_raw = page_body.get("total")
    total = total_raw if isinstance(total_raw, int) and total_raw >= 0 else len(content)

    if not content:
        if name:
            return f"No {entity_type}s matching {name!r} found."
        return f"No {entity_type}s found."

    # Write-through cache (best-effort; never breaks the list path).
    cache_objects(entity_type, content, settings=settings)

    return _format_table(entity_type, handler, content, total, page, size, name)


def _format_table(
    entity_type: str,
    handler: EntityHandler,
    content: list[dict[str, Any]],
    total: int,
    page: int,
    size: int,
    name: str | None,
) -> str:
    """Pipe-delimited table — mirrors ollie's ``_format_table``."""
    columns: tuple[str, ...] = ("id", "name", *handler.list_extra_fields)
    count = len(content)
    if name:
        header = (
            f"Found {total} {entity_type}s matching {name!r} "
            f"(page {page}, showing {count} of {total}):"
        )
    else:
        header = f"Found {total} {entity_type}s (page {page}, showing {count} of {total}):"

    col_header = " | ".join(columns)
    rows: list[str] = []
    for item in content:
        values: list[str] = []
        for col in columns:
            val = item.get(col)
            s = "" if val is None else str(val)
            # Never truncate identifiers — they're needed verbatim for
            # follow-up read() calls (e.g. thread ids are 64 chars).
            if col != "id" and len(s) > _TRUNCATE_AT:
                s = s[: _TRUNCATE_AT - 3] + "..."
            values.append(s)
        rows.append(" | ".join(values))

    lines = [header, "", col_header, *rows]
    if page * size < total:
        lines.append("")
        lines.append(f"Use page={page + 1} for next {size} results.")
    return "\n".join(lines)


__all__ = ["run_list"]
