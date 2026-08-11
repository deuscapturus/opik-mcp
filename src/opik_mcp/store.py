"""Local parquet-backed cache for read/list results.

Every ``read``/``list`` call writes its result here (fail-soft, best-effort) so
the ``query`` / ``search`` / ``plot`` tools can operate over the accumulated
data with duckdb — including resolving the truncation breadcrumbs the
compression layer emits (a full value lives in the cache even when the
LLM-facing string was truncated).

Layout under ``settings.data_dir``::

    <data_dir>/
        state.json                 # bookkeeping (last write ts per entity)
        entities/<entity>/         # one dir per entity_type
            workspace=<ws>/        # per-workspace partition (isolation boundary)
                project_id=<pid>/  # per-project partition (pid, or __none__)
                    part-<uuid12>.parquet  # append-only parts, one row/obj

Workspace is the isolation boundary: reads never cross into another workspace's
partition, so a single user pointing the same ``data_dir`` at different
workspaces (or projects) over time can never commingle rows.

Uses atomic writes (tmp file + ``os.replace``),
glob-and-concat reads, and a small JSON state file updated via read-merge-write.

pandas/pyarrow/duckdb are an OPTIONAL dependency (``opik-mcp[analytics]``). The
module imports them lazily so the base server never pays for them; helpers raise
:class:`AnalyticsExtraMissing` with an actionable install hint when absent.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opik_mcp.config import DEFAULT_WORKSPACE, Settings, get_settings

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger("opik_mcp.store")

_INSTALL_HINT = (
    "This tool needs the optional analytics extra. Install it with:\n"
    "    pip install 'opik-mcp[analytics]'\n"
    "(provides pandas, pyarrow and duckdb)."
)


class AnalyticsExtraMissing(RuntimeError):
    """Raised when a cache/query helper is used without ``opik-mcp[analytics]``."""


LOCAL_ONLY_HINT = (
    "The query/search/plot insight tools read a local parquet cache that only "
    "exists in local (stdio) mode. This server is running in hosted (HTTP) "
    "mode, where the cache is disabled — it is a single unpartitioned directory "
    "with no per-tenant isolation, so it is unsafe to share. Run opik-mcp "
    "locally over stdio to use these tools."
)


def local_persistence_enabled(settings: Settings | None = None) -> bool:
    """True when the local parquet cache is active for this transport."""
    return (settings or get_settings()).local_persistence_enabled


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as e:  # pragma: no cover - exercised via extra-missing test
        raise AnalyticsExtraMissing(_INSTALL_HINT) from e
    return pd


def _require_duckdb() -> Any:
    try:
        import duckdb
    except ModuleNotFoundError as e:  # pragma: no cover
        raise AnalyticsExtraMissing(_INSTALL_HINT) from e
    return duckdb


def analytics_available() -> bool:
    """True when pandas, pyarrow and duckdb can all be imported."""
    try:
        import duckdb  # noqa: F401
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _data_dir(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).data_dir


def _entities_dir(settings: Settings | None = None) -> Path:
    return _data_dir(settings) / "entities"


def entity_dir(entity_type: str, settings: Settings | None = None) -> Path:
    """Common ancestor of all cached parquet parts for one entity type.

    Parts themselves live deeper, under nested
    ``workspace=<ws>/project_id=<pid>`` partitions.
    """
    return _entities_dir(settings) / entity_type


_NO_PROJECT = "__none__"
"""Partition bucket for cached objects that carry no ``project_id``."""


def _enc(value: str) -> str:
    """Encode a workspace / project_id into a single safe path segment.

    ``quote(safe="")`` is injective (distinct inputs never collide) and emits no
    path separators or glob metacharacters, so distinct workspaces can never
    share a directory and no value can smuggle a wildcard into a read glob.
    """
    from urllib.parse import quote

    return quote(value, safe="")


def _current_workspace(settings: Settings | None = None) -> str:
    """Resolve the workspace the cache partitions under.

    The cache only runs in local (stdio) mode, where no inbound HTTP context
    exists, so this matches ``resolve_opik_config``'s local resolution:
    ``COMET_WORKSPACE`` else the ``default`` convention.
    """
    s = settings or get_settings()
    return s.comet_workspace or DEFAULT_WORKSPACE


def _workspace_dir(entity_type: str, workspace: str, settings: Settings | None = None) -> Path:
    """Per-workspace subtree for one entity type (the isolation boundary)."""
    return entity_dir(entity_type, settings) / f"workspace={_enc(workspace)}"


def _partition_dir(
    entity_type: str,
    workspace: str,
    project_id: str,
    settings: Settings | None = None,
) -> Path:
    """Leaf directory holding parts for one (workspace, project_id) pair."""
    return _workspace_dir(entity_type, workspace, settings) / (f"project_id={_enc(project_id)}")


def _state_path(settings: Settings | None = None) -> Path:
    return _data_dir(settings) / "state.json"


# --------------------------------------------------------------------------- #
# state.json
# --------------------------------------------------------------------------- #
def load_state(settings: Settings | None = None) -> dict[str, Any]:
    path = _state_path(settings)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.debug("state.json unreadable (%s); treating as empty", e)
        return {}


def save_state(state: dict[str, Any], settings: Settings | None = None) -> None:
    path = _state_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, path)


def update_state(key: str, value: Any, settings: Settings | None = None) -> None:
    """Read-merge-write a single key so concurrent writers don't clobber."""
    state = load_state(settings)
    state[key] = value
    save_state(state, settings)


# --------------------------------------------------------------------------- #
# parquet write / read
# --------------------------------------------------------------------------- #
def _write_parquet_atomic(df: "pd.DataFrame", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _append_part(
    entity_type: str,
    df: "pd.DataFrame",
    workspace: str,
    project_id: str,
    settings: Settings | None = None,
) -> Path:
    part = (
        _partition_dir(entity_type, workspace, project_id, settings)
        / f"part-{uuid.uuid4().hex[:12]}.parquet"
    )
    _write_parquet_atomic(df, part)
    return part


def _partition_of(row: dict[str, Any]) -> str:
    """The ``project_id`` a cached object belongs to, or the no-project bucket."""
    pid = row.get("project_id")
    return pid if isinstance(pid, str) and pid else _NO_PROJECT


def _rows_to_frame(rows: list[dict[str, Any]]) -> "pd.DataFrame":
    """Normalize cached objects into a one-column-per-scalar frame.

    Nested/compound values are JSON-encoded into a string column so parquet
    stays flat and duckdb can ``json_extract`` them later. A ``_raw`` column
    always carries the full JSON of the object so the ``query`` tool can resolve
    truncation breadcrumbs against the complete value.
    """
    pd = _require_pandas()
    flat: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = {"_raw": json.dumps(row, default=str)}
        for k, v in row.items():
            if isinstance(v, (dict, list)):
                out[k] = json.dumps(v, default=str)
            else:
                out[k] = v
        flat.append(out)
    return pd.DataFrame(flat)


def cache_objects(
    entity_type: str,
    objects: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
) -> None:
    """Persist cached objects for ``entity_type`` (best-effort, never raises).

    Called write-through from ``run_read``/``run_list``. Any failure — missing
    optional deps, unwritable disk, un-serializable payload — is swallowed and
    logged at debug so it can never break the user-facing read/list path.

    No-ops entirely in hosted (HTTP) mode: the cache is a single local directory
    with no per-tenant isolation, so it is only written in local (stdio) mode.
    """
    if not objects:
        return
    settings = settings or get_settings()
    if not local_persistence_enabled(settings):
        return
    try:
        workspace = _current_workspace(settings)
        # A single list response can span multiple projects; group by project_id
        # so every part file lands under exactly one partition.
        groups: dict[str, list[dict[str, Any]]] = {}
        for obj in objects:
            groups.setdefault(_partition_of(obj), []).append(obj)
        for project_id, rows in groups.items():
            _append_part(entity_type, _rows_to_frame(rows), workspace, project_id, settings)
        update_state(
            f"entities.{entity_type}.last_write_utc",
            _now_iso(),
            settings,
        )
    except AnalyticsExtraMissing:
        logger.debug("analytics extra missing; skipping cache for %s", entity_type)
    except Exception as e:  # pragma: no cover - defensive; write-through must not break reads
        logger.debug("cache write-through failed for %s: %s", entity_type, e)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def read_dataset(entity_type: str, settings: Settings | None = None) -> "pd.DataFrame":
    """Concat cached parquet parts for ``entity_type`` in the current workspace.

    Unions every project partition under the current workspace, but never crosses
    into another workspace — the on-disk partition is the isolation boundary.
    """
    pd = _require_pandas()
    ws_dir = _workspace_dir(entity_type, _current_workspace(settings), settings)
    parts = sorted(ws_dir.glob("project_id=*/part-*.parquet"))
    if not parts:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in parts]
    return pd.concat(frames, ignore_index=True)


def cached_entity_types(settings: Settings | None = None) -> list[str]:
    """Entity types with at least one cached part in the current workspace."""
    root = _entities_dir(settings)
    if not root.exists():
        return []
    ws_segment = f"workspace={_enc(_current_workspace(settings))}"
    out = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        ws_dir = d / ws_segment
        if ws_dir.is_dir() and any(ws_dir.glob("project_id=*/part-*.parquet")):
            out.append(d.name)
    return sorted(out)


# --------------------------------------------------------------------------- #
# duckdb query surface
# --------------------------------------------------------------------------- #
def query_sql(sql: str, settings: Settings | None = None) -> "pd.DataFrame":
    """Run a read-only SQL query over the parquet cache with duckdb.

    Each cached entity type is exposed as a view named after it (e.g.
    ``trace``, ``span``) plus a ``cache_<entity>`` alias, so callers can write
    ``SELECT * FROM trace`` directly. Views only cover entity types that have
    cached parts.
    """
    duckdb = _require_duckdb()
    con = duckdb.connect(database=":memory:")
    try:
        workspace = _current_workspace(settings)
        for entity in cached_entity_types(settings):
            pattern = str(
                _workspace_dir(entity, workspace, settings) / "project_id=*" / "part-*.parquet"
            )
            escaped = pattern.replace("'", "''")
            for view in (entity, f"cache_{entity}"):
                con.execute(
                    f'CREATE VIEW "{view}" AS '
                    f"SELECT * FROM read_parquet('{escaped}', union_by_name=true)"
                )
        return con.execute(sql).df()
    finally:
        con.close()
