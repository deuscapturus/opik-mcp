"""Tests for ``opik_docs`` and ``read_skill`` (``insights.docs_tool``).

Both tools fetch text at call time over httpx. Tests mock the network with
``respx`` and assert URL construction (base-join, ``SKILL.md`` defaulting),
truncation, and error mapping. They do not require the analytics extra.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from mcp.server.fastmcp.exceptions import ToolError

from opik_mcp.config import Settings
from opik_mcp.insights.docs_tool import run_opik_docs, run_read_skill

_DOCS_BASE = "https://docs.test/opik"
_SKILLS_BASE = "https://skills.test/tree/main/skills"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        opik_mcp_docs_base_url=_DOCS_BASE,
        opik_mcp_skills_base_url=_SKILLS_BASE,
    )


@pytest.mark.anyio
@respx.mock
async def test_opik_docs_fetches_page(settings: Settings) -> None:
    route = respx.get(f"{_DOCS_BASE}/tracing/log_traces").mock(
        return_value=httpx.Response(200, text="# Log traces")
    )
    result = await run_opik_docs("tracing/log_traces", settings=settings)

    assert route.called
    assert result["url"] == f"{_DOCS_BASE}/tracing/log_traces"
    assert result["truncated"] is False
    assert result["content"] == "# Log traces"


@pytest.mark.anyio
@respx.mock
async def test_opik_docs_truncates_long_content(settings: Settings) -> None:
    body = "x" * 70_000
    respx.get(f"{_DOCS_BASE}/big").mock(return_value=httpx.Response(200, text=body))
    result = await run_opik_docs("big", settings=settings)

    assert result["truncated"] is True
    assert len(result["content"]) == 60_000


@pytest.mark.anyio
@respx.mock
async def test_opik_docs_404_is_tool_error(settings: Settings) -> None:
    respx.get(f"{_DOCS_BASE}/missing").mock(return_value=httpx.Response(404))
    with pytest.raises(ToolError, match="404"):
        await run_opik_docs("missing", settings=settings)


@pytest.mark.anyio
async def test_opik_docs_requires_path(settings: Settings) -> None:
    with pytest.raises(ToolError, match="path is required"):
        await run_opik_docs("  ", settings=settings)


@pytest.mark.anyio
@respx.mock
async def test_read_skill_appends_skill_md(settings: Settings) -> None:
    """A name with no file extension is treated as a skill dir → SKILL.md."""
    route = respx.get(f"{_SKILLS_BASE}/chronicle/SKILL.md").mock(
        return_value=httpx.Response(200, text="skill body")
    )
    result = await run_read_skill("chronicle", settings=settings)

    assert route.called
    assert result["url"] == f"{_SKILLS_BASE}/chronicle/SKILL.md"
    assert result["content"] == "skill body"


@pytest.mark.anyio
@respx.mock
async def test_read_skill_explicit_file_kept(settings: Settings) -> None:
    """A name that already names a file is fetched verbatim."""
    route = respx.get(f"{_SKILLS_BASE}/chronicle/SKILL.md").mock(
        return_value=httpx.Response(200, text="skill body")
    )
    await run_read_skill("chronicle/SKILL.md", settings=settings)
    assert route.called


@pytest.mark.anyio
async def test_read_skill_requires_name(settings: Settings) -> None:
    with pytest.raises(ToolError, match="name is required"):
        await run_read_skill("", settings=settings)
