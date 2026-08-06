"""``opik_docs`` and ``read_skill`` MCP tools — runtime remote fetch.

Both tools fetch text at call time from a configured base URL (nothing is
bundled with the server). ``opik_docs`` reaches the public Opik docs site;
``read_skill`` pulls a skill markdown file from the opik-mcp repo's skills
tree. Base URLs are configurable via ``Settings`` for pinning/mirroring.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

import httpx
from mcp.server.fastmcp.exceptions import ToolError

from opik_mcp.config import Settings, get_settings

_TIMEOUT_S = 20.0
_MAX_CHARS = 60_000


def _clean_path(path: str) -> str:
    """Strip a leading slash so ``urljoin`` treats ``base`` as a directory."""
    return path.strip().lstrip("/")


async def _fetch(url: str) -> str:
    async with httpx.AsyncClient(timeout=_TIMEOUT_S, follow_redirects=True) as client:
        resp = await client.get(url, headers={"Accept": "text/plain, text/markdown, text/html"})
    if resp.status_code == 404:
        raise ToolError(f"Not found (404): {url}")
    if resp.status_code >= 400:
        raise ToolError(f"Fetch failed ({resp.status_code}) for {url}")
    return resp.text


def _base(base: str) -> str:
    """Ensure a trailing slash so ``urljoin`` appends rather than replaces."""
    return base if base.endswith("/") else base + "/"


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= _MAX_CHARS:
        return text, False
    return text[:_MAX_CHARS], True


async def run_opik_docs(
    path: str,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Fetch an Opik docs page by ``path`` (e.g. ``tracing/log_traces``)."""
    settings = settings or get_settings()
    if not path or not path.strip():
        raise ToolError("path is required (e.g. 'tracing/log_traces').")
    url = urljoin(_base(settings.opik_mcp_docs_base_url), _clean_path(path))
    text, truncated = _truncate(await _fetch(url))
    return {"url": url, "truncated": truncated, "content": text}


async def run_read_skill(
    name: str,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Fetch a skill markdown file by ``name`` (e.g. ``chronicle/SKILL.md``)."""
    settings = settings or get_settings()
    if not name or not name.strip():
        raise ToolError("name is required (e.g. 'chronicle/SKILL.md').")
    rel = _clean_path(name)
    if "." not in rel.rsplit("/", 1)[-1]:
        rel = f"{rel}/SKILL.md"
    url = urljoin(_base(settings.opik_mcp_skills_base_url), rel)
    text, truncated = _truncate(await _fetch(url))
    return {"url": url, "truncated": truncated, "content": text}


__all__ = ["run_opik_docs", "run_read_skill"]
