"""``query`` MCP tool — duckdb SQL over the parquet write-through cache.

Folds two capabilities into one tool:

1. **scan** — run arbitrary read-only SQL across everything ``read``/``list``
   have cached this session. Each cached entity type is a view (e.g.
   ``SELECT * FROM trace``). A ``_raw`` column on every view carries the full
   JSON of the object.
2. **jq / breadcrumb resolution** — the compression layer truncates large
   string values and leaves a breadcrumb like
   ``[TRUNCATED 40123 chars — full value at .trace.output]``. Passing that
   ``path`` (``.trace.output``) here extracts the *complete* value from the
   cache via ``json_extract`` — no re-fetch, no truncation.

Only ``SELECT`` / ``WITH`` statements are permitted.
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
    query_sql,
)

_READONLY_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_MAX_ROWS = 200


def _resolve_path(path: str, settings: Settings | None) -> str:
    """Resolve a ``.entity.field[.field...]`` breadcrumb path against the cache."""
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise ToolError("Empty path. Expected e.g. '.trace.output'.")
    entity = parts[0]
    fields = parts[1:]
    cached = cached_entity_types(settings)
    if entity not in cached:
        avail = ", ".join(cached) or "(nothing cached yet)"
        raise ToolError(f"No cached data for {entity!r}. Run read/list first. Cached: {avail}.")
    if not fields:
        sql = f'SELECT _raw AS value FROM "{entity}" LIMIT {_MAX_ROWS}'  # noqa: S608
    else:
        json_path = "$." + ".".join(fields)
        escaped = json_path.replace("'", "''")
        sql = (  # noqa: S608
            f"SELECT json_extract(_raw, '{escaped}') AS value "
            f"FROM \"{entity}\" WHERE json_extract(_raw, '{escaped}') IS NOT NULL "
            f"LIMIT {_MAX_ROWS}"
        )
    return sql


def run_query(
    *,
    sql: str | None = None,
    path: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Query the parquet cache.

    Provide exactly one of:
    - ``sql``: a read-only SELECT/WITH statement over cached entity views.
    - ``path``: a breadcrumb like ``.trace.output`` to extract a full value.
    """
    if not local_persistence_enabled(settings):
        raise ToolError(LOCAL_ONLY_HINT)

    if (sql is None) == (path is None):
        raise ToolError("Provide exactly one of 'sql' or 'path'.")

    if path is not None:
        sql = _resolve_path(path, settings)
    else:
        assert sql is not None
        if not _READONLY_RE.match(sql):
            raise ToolError("Only read-only SELECT/WITH queries are allowed.")

    try:
        df = query_sql(sql, settings)
    except AnalyticsExtraMissing as e:
        raise ToolError(str(e)) from e
    except Exception as e:  # duckdb parse/binder errors → actionable message
        avail = ", ".join(cached_entity_types(settings)) or "(nothing cached yet)"
        raise ToolError(f"Query failed: {e}\nCached entity views: {avail}.") from e

    truncated = len(df) > _MAX_ROWS
    rows = df.head(_MAX_ROWS).to_dict(orient="records")
    return {
        "row_count": int(len(df)),
        "returned": len(rows),
        "truncated": truncated,
        "columns": [str(c) for c in df.columns],
        "rows": json.loads(json.dumps(rows, default=str)),
    }


__all__ = ["run_query"]
