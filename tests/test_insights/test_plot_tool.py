"""Tests for the ``plot`` insights tool (``insights.plot_tool.run_plot``).

Runs read-only SQL over the cache and writes a self-contained HTML report to
``settings.reports_dir``. Tests assert the file lands where expected and that
validation guards fire before any file is written.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from opik_mcp.config import Settings
from opik_mcp.insights.plot_tool import run_plot

from .conftest import requires_analytics

pytestmark = requires_analytics


def test_plot_writes_html_report(cache: Settings) -> None:
    """A valid query writes an HTML file under reports_dir and returns its uri."""
    result = run_plot(
        "SELECT name, duration_ms FROM trace ORDER BY name",
        title="Trace durations",
        kind="bar",
        settings=cache,
    )

    out = Path(result["path"])
    assert out.exists()
    assert out.parent == cache.reports_dir
    assert out.suffix == ".html"
    assert result["uri"] == out.as_uri()
    assert result["row_count"] == 2
    assert result["plotted_rows"] == 2
    document = out.read_text(encoding="utf-8")
    assert "Trace durations" in document
    # The chart payload is embedded in the document.
    assert "duration_ms" in document


def test_plot_rejects_unknown_kind(cache: Settings) -> None:
    """An unsupported chart kind is rejected before running the query."""
    with pytest.raises(ToolError, match="kind must be one of"):
        run_plot("SELECT name, duration_ms FROM trace", kind="pie", settings=cache)


def test_plot_rejects_non_readonly_sql(cache: Settings) -> None:
    """Only SELECT/WITH statements are permitted."""
    with pytest.raises(ToolError, match="read-only"):
        run_plot("UPDATE trace SET name='x'", settings=cache)


def test_plot_unknown_column_is_tool_error(cache: Settings) -> None:
    """Naming a column not in the result set is a validation error."""
    with pytest.raises(ToolError, match="not in result columns"):
        run_plot("SELECT name, duration_ms FROM trace", y="missing", settings=cache)


def test_plot_empty_result_is_tool_error(cache: Settings) -> None:
    """A query that returns no rows has nothing to plot."""
    with pytest.raises(ToolError, match="no rows"):
        run_plot("SELECT name, duration_ms FROM trace WHERE 1=0", settings=cache)
