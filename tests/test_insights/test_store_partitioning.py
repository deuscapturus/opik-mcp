"""Cache path-partitioning by workspace and project_id.

The local parquet cache nests parts under
``entities/<entity>/workspace=<ws>/project_id=<pid>/`` so a single user pointing
the same ``data_dir`` at different workspaces (or projects) over time can never
commingle rows. These tests drive the production write-through entry point and
assert both the on-disk layout and the read-side isolation boundary.
"""

from __future__ import annotations

from urllib.parse import quote

import pytest

from opik_mcp import store
from opik_mcp.config import DEFAULT_WORKSPACE, Settings

from .conftest import requires_analytics

pytestmark = requires_analytics


def _settings(tmp_path, workspace: str | None = None) -> Settings:
    return Settings(
        opik_mcp_data_dir=str(tmp_path / "cache"),
        comet_workspace=workspace,
    )


def test_parts_written_under_workspace_and_project_partitions(tmp_path):
    settings = _settings(tmp_path, workspace="acme")
    store.cache_objects(
        "trace",
        [{"id": "t1", "project_id": "p1"}, {"id": "t2", "project_id": "p1"}],
        settings=settings,
    )

    leaf = (
        store.entity_dir("trace", settings)
        / f"workspace={quote('acme', safe='')}"
        / f"project_id={quote('p1', safe='')}"
    )
    assert list(leaf.glob("part-*.parquet")), "expected a part under the partition"


def test_missing_project_id_falls_into_none_bucket(tmp_path):
    settings = _settings(tmp_path)  # workspace -> DEFAULT_WORKSPACE
    store.cache_objects("trace", [{"id": "t1"}], settings=settings)

    leaf = (
        store.entity_dir("trace", settings)
        / f"workspace={quote(DEFAULT_WORKSPACE, safe='')}"
        / "project_id=__none__"
    )
    assert list(leaf.glob("part-*.parquet"))
    # still readable in the current workspace
    df = store.read_dataset("trace", settings)
    assert set(df["id"]) == {"t1"}


def test_single_write_splits_mixed_project_ids(tmp_path):
    settings = _settings(tmp_path)
    store.cache_objects(
        "trace",
        [
            {"id": "t1", "project_id": "p1"},
            {"id": "t2", "project_id": "p2"},
            {"id": "t3"},
        ],
        settings=settings,
    )

    ws_dir = store.entity_dir("trace", settings) / (
        f"workspace={quote(DEFAULT_WORKSPACE, safe='')}"
    )
    project_dirs = {d.name for d in ws_dir.iterdir() if d.is_dir()}
    assert project_dirs == {"project_id=p1", "project_id=p2", "project_id=__none__"}
    # a read in this workspace unions across all its projects
    df = store.read_dataset("trace", settings)
    assert set(df["id"]) == {"t1", "t2", "t3"}


def test_reads_are_isolated_per_workspace(tmp_path):
    ws_a = _settings(tmp_path, workspace="ws-a")
    ws_b = _settings(tmp_path, workspace="ws-b")  # same data_dir, different workspace

    store.cache_objects("trace", [{"id": "a1", "project_id": "p1"}], settings=ws_a)
    store.cache_objects("trace", [{"id": "b1", "project_id": "p1"}], settings=ws_b)

    assert set(store.read_dataset("trace", ws_a)["id"]) == {"a1"}
    assert set(store.read_dataset("trace", ws_b)["id"]) == {"b1"}


def test_cached_entity_types_scoped_to_workspace(tmp_path):
    ws_a = _settings(tmp_path, workspace="ws-a")
    ws_b = _settings(tmp_path, workspace="ws-b")

    store.cache_objects("trace", [{"id": "a1"}], settings=ws_a)

    assert store.cached_entity_types(ws_a) == ["trace"]
    assert store.cached_entity_types(ws_b) == []


def test_query_sql_isolated_per_workspace(tmp_path):
    ws_a = _settings(tmp_path, workspace="ws-a")
    ws_b = _settings(tmp_path, workspace="ws-b")

    store.cache_objects("trace", [{"id": "a1", "project_id": "p1"}], settings=ws_a)
    store.cache_objects("trace", [{"id": "b1", "project_id": "p1"}], settings=ws_b)

    df_a = store.query_sql("SELECT id FROM trace ORDER BY id", ws_a)
    df_b = store.query_sql("SELECT id FROM trace ORDER BY id", ws_b)
    assert list(df_a["id"]) == ["a1"]
    assert list(df_b["id"]) == ["b1"]
