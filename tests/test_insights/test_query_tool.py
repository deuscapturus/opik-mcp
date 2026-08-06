"""Tests for the ``query`` insights tool (``insights.query_tool.run_query``).

Covers the two folded capabilities: arbitrary read-only SQL over the parquet
write-through cache, and breadcrumb-path resolution that extracts a full value
from the ``_raw`` JSON column.
"""

from __future__ import annotations

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from opik_mcp.config import Settings
from opik_mcp.insights.query_tool import run_query

from .conftest import requires_analytics

pytestmark = requires_analytics


def test_query_sql_over_cached_view(cache: Settings) -> None:
    """A SELECT against the ``trace`` view returns cached rows."""
    result = run_query(sql="SELECT id, name FROM trace ORDER BY id", settings=cache)

    assert result["row_count"] == 2
    assert result["truncated"] is False
    assert {"id", "name"} <= set(result["columns"])
    ids = [row["id"] for row in result["rows"]]
    assert ids == ["t1", "t2"]


def test_query_path_resolves_full_value(cache: Settings) -> None:
    """A breadcrumb path extracts the full field value from ``_raw``."""
    result = run_query(path=".trace.output", settings=cache)

    values = [row["value"] for row in result["rows"]]
    # duckdb json_extract returns JSON-encoded scalars (quoted strings).
    assert any("quick brown fox" in str(v) for v in values)


def test_query_requires_exactly_one_arg(cache: Settings) -> None:
    """Passing neither or both of sql/path is a validation error."""
    with pytest.raises(ToolError):
        run_query(settings=cache)
    with pytest.raises(ToolError):
        run_query(sql="SELECT 1", path=".trace.output", settings=cache)


def test_query_rejects_non_readonly_sql(cache: Settings) -> None:
    """Only SELECT/WITH statements are permitted."""
    with pytest.raises(ToolError, match="read-only"):
        run_query(sql="DELETE FROM trace", settings=cache)


def test_query_path_unknown_entity_lists_cached(cache: Settings) -> None:
    """An unresolvable entity path reports what is actually cached."""
    with pytest.raises(ToolError, match="No cached"):
        run_query(path=".span.output", settings=cache)


def test_query_bad_sql_reports_available_views(cache: Settings) -> None:
    """A binder error surfaces an actionable message with the cached views."""
    with pytest.raises(ToolError, match="Query failed"):
        run_query(sql="SELECT * FROM no_such_table", settings=cache)
