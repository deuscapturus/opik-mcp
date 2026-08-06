"""Tests for the ``search`` insights tool (``insights.search_tool.run_search``).

Regex-scans cached string leaves of an entity type and returns the JSON path
(a ``query`` breadcrumb) plus a surrounding snippet for each match.
"""

from __future__ import annotations

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from opik_mcp.config import Settings
from opik_mcp.insights.search_tool import run_search

from .conftest import requires_analytics

pytestmark = requires_analytics


def test_search_finds_match_with_path_and_snippet(cache: Settings) -> None:
    """A hit reports the owning id, a dotted JSON path, and a snippet."""
    result = run_search("trace", "quick brown", settings=cache)

    assert result["match_count"] == 1
    match = result["matches"][0]
    assert match["id"] == "t1"
    assert match["path"] == ".trace.output"
    assert "quick brown" in match["snippet"]


def test_search_ignore_case_default(cache: Settings) -> None:
    """Search is case-insensitive by default."""
    assert run_search("trace", "QUICK", settings=cache)["match_count"] == 1
    assert run_search("trace", "QUICK", ignore_case=False, settings=cache)["match_count"] == 0


def test_search_no_matches_returns_empty(cache: Settings) -> None:
    """A pattern that matches nothing yields an empty, non-truncated result."""
    result = run_search("trace", "zzz-not-present", settings=cache)
    assert result["match_count"] == 0
    assert result["truncated"] is False
    assert result["matches"] == []


def test_search_invalid_regex_is_tool_error(cache: Settings) -> None:
    """A malformed pattern is reported as a validation error."""
    with pytest.raises(ToolError, match="Invalid regex"):
        run_search("trace", "(unclosed", settings=cache)


def test_search_uncached_entity_is_tool_error(cache: Settings) -> None:
    """Searching an entity with no cached data lists what is cached."""
    with pytest.raises(ToolError, match="No cached"):
        run_search("span", "anything", settings=cache)
