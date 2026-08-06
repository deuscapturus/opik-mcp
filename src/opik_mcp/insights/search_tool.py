"""``search`` MCP tool — regex across cached string values of an entity.

Scans every cached object of an entity type (the write-through parquet cache)
and returns matches with the JSON path where the pattern was found. Complements
``query``: use ``search`` when you don't know *where* a value lives, ``query``
when you do.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mcp.server.fastmcp.exceptions import ToolError

from opik_mcp.config import Settings
from opik_mcp.store import (
    LOCAL_ONLY_HINT,
    AnalyticsExtraMissing,
    cached_entity_types,
    local_persistence_enabled,
    read_dataset,
)

_MAX_MATCHES = 100
_SNIPPET_PAD = 60


def _walk_strings(obj: Any, prefix: str) -> list[tuple[str, str]]:
    """Yield ``(json_path, string_value)`` for every string leaf in ``obj``."""
    out: list[tuple[str, str]] = []
    if isinstance(obj, str):
        out.append((prefix, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_walk_strings(v, f"{prefix}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_walk_strings(v, f"{prefix}[{i}]"))
    return out


def _snippet(value: str, match: re.Match[str]) -> str:
    start = max(0, match.start() - _SNIPPET_PAD)
    end = min(len(value), match.end() + _SNIPPET_PAD)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(value) else ""
    return f"{prefix}{value[start:end]}{suffix}"


def run_search(
    entity_type: str,
    pattern: str,
    *,
    ignore_case: bool = True,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Regex-search cached objects of ``entity_type``.

    Returns matches with the JSON path (usable as a ``query`` breadcrumb) and a
    surrounding snippet.
    """
    if not local_persistence_enabled(settings):
        raise ToolError(LOCAL_ONLY_HINT)

    try:
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)
    except re.error as e:
        raise ToolError(f"Invalid regex: {e}") from e

    try:
        df = read_dataset(entity_type, settings)
    except AnalyticsExtraMissing as e:
        raise ToolError(str(e)) from e

    if df.empty:
        cached = ", ".join(cached_entity_types(settings)) or "(nothing cached yet)"
        raise ToolError(f"No cached {entity_type!r} data. Run read/list first. Cached: {cached}.")

    matches: list[dict[str, Any]] = []
    for raw in df.get("_raw", []):
        if not isinstance(raw, str):
            continue
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        root = obj.get("id") if isinstance(obj, dict) else None
        for path, value in _walk_strings(obj, f".{entity_type}"):
            m = regex.search(value)
            if m is None:
                continue
            matches.append({"id": root, "path": path, "snippet": _snippet(value, m)})
            if len(matches) >= _MAX_MATCHES:
                break
        if len(matches) >= _MAX_MATCHES:
            break

    return {
        "entity_type": entity_type,
        "pattern": pattern,
        "match_count": len(matches),
        "truncated": len(matches) >= _MAX_MATCHES,
        "matches": matches,
    }


__all__ = ["run_search"]
