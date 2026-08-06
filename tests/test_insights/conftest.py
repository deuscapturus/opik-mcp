"""Shared fixtures for the insights-tool tests.

Every insights test drives real code paths against a throwaway parquet cache
rooted at a ``tmp_path`` so nothing touches ``~/.opik-mcp``. The ``cache``
fixture yields a ``Settings`` pointed at that temp dir and pre-seeds it via the
production write-through path (``store.cache_objects``), so tests exercise the
exact serialisation the server uses at runtime.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from opik_mcp import store
from opik_mcp.config import Settings

# The query/search/plot tools need pandas + duckdb (the optional ``analytics``
# extra). Skip the whole module cleanly when it is absent rather than erroring.
requires_analytics = pytest.mark.skipif(
    not store.analytics_available(),
    reason="analytics extra (pandas/pyarrow/duckdb) not installed",
)


@pytest.fixture
def settings(tmp_path) -> Settings:  # type: ignore[no-untyped-def]
    """A ``Settings`` whose parquet cache lives under a per-test temp dir."""
    return Settings(opik_mcp_data_dir=str(tmp_path / "cache"))


@pytest.fixture
def cache(settings: Settings) -> Iterator[Settings]:
    """Seed the cache with a couple of trace objects and return the settings.

    Uses the production write-through entry point so the parquet layout, the
    ``_raw`` JSON column, and view names all match runtime exactly.
    """
    traces = [
        {
            "id": "t1",
            "name": "alpha",
            "duration_ms": 120,
            "output": "the quick brown fox jumps over the lazy dog",
        },
        {
            "id": "t2",
            "name": "beta",
            "duration_ms": 340,
            "output": "lorem ipsum dolor sit amet",
        },
    ]
    store.cache_objects("trace", traces, settings=settings)
    yield settings
