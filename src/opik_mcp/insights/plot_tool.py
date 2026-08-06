"""``plot`` MCP tool — self-contained HTML report from a SQL query.

Runs a read-only duckdb query over the parquet cache, then writes a single,
dependency-free HTML file (inline CSS + vanilla-JS canvas chart, dark/light
aware) to ``settings.reports_dir``. Returns the file path/URI. No external
assets, no network — the report opens straight from disk.
"""

from __future__ import annotations

import html
import json
import math
import re
import uuid
from typing import Any

from mcp.server.fastmcp.exceptions import ToolError

from opik_mcp.config import Settings, get_settings
from opik_mcp.store import (
    LOCAL_ONLY_HINT,
    AnalyticsExtraMissing,
    cached_entity_types,
    local_persistence_enabled,
    query_sql,
)

_READONLY_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_MAX_POINTS = 500
_CHART_KINDS = ("bar", "line", "scatter")

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  background: #ffffff; color: #1a1a1a;
}
@media (prefers-color-scheme: dark) {
  body { background: #14151a; color: #e6e6e6; }
  th { background: #1f2029 !important; }
  tr:nth-child(even) td { background: #1a1b22 !important; }
  canvas { background: #1a1b22; }
}
main { max-width: 960px; margin: 0 auto; padding: 24px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.meta { opacity: .65; font-size: 12px; margin-bottom: 20px; }
canvas { width: 100%; height: 360px; background: #f7f7f9; border-radius: 8px; }
table { border-collapse: collapse; width: 100%; margin-top: 24px; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #8883; }
th { background: #f0f0f3; position: sticky; top: 0; }
tr:nth-child(even) td { background: #fafafb; }
code { background: #8882; padding: 1px 5px; border-radius: 4px; }
</style>
</head>
<body>
<main>
<h1>__TITLE__</h1>
<div class="meta">__META__</div>
<canvas id="chart" width="900" height="360"></canvas>
<div id="table"></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const payload = JSON.parse(document.getElementById("data").textContent);
const {rows, columns, chart} = payload;

function renderTable() {
  if (!rows.length) return;
  const t = document.createElement("table");
  const thead = document.createElement("thead");
  const htr = document.createElement("tr");
  columns.forEach(c => { const th = document.createElement("th"); th.textContent = c; htr.appendChild(th); });
  thead.appendChild(htr); t.appendChild(thead);
  const tb = document.createElement("tbody");
  rows.forEach(r => {
    const tr = document.createElement("tr");
    columns.forEach(c => { const td = document.createElement("td"); td.textContent = r[c] ?? ""; tr.appendChild(td); });
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  document.getElementById("table").appendChild(t);
}

function renderChart() {
  const cv = document.getElementById("chart");
  if (!chart || !chart.x) { cv.style.display = "none"; return; }
  const ctx = cv.getContext("2d");
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const fg = dark ? "#e6e6e6" : "#1a1a1a";
  const accent = "#5b8def";
  const W = cv.width, H = cv.height, pad = 48;
  const xs = rows.map(r => r[chart.x]);
  const ys = rows.map(r => Number(r[chart.y]) || 0);
  const maxY = Math.max(1, ...ys), minY = Math.min(0, ...ys);
  const plotW = W - pad * 2, plotH = H - pad * 2;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = fg + "44"; ctx.fillStyle = fg; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad); ctx.lineTo(W - pad, H - pad); ctx.stroke();
  const yToPx = v => (H - pad) - ((v - minY) / (maxY - minY || 1)) * plotH;
  ctx.font = "11px system-ui"; ctx.fillText(String(maxY), 4, pad + 4); ctx.fillText(String(minY), 4, H - pad);
  const n = rows.length;
  if (chart.kind === "bar") {
    const bw = plotW / n * 0.7;
    ctx.fillStyle = accent;
    ys.forEach((v, i) => {
      const x = pad + (i + 0.5) * (plotW / n) - bw / 2;
      const y = yToPx(v);
      ctx.fillRect(x, y, bw, (H - pad) - y);
    });
  } else if (chart.kind === "line") {
    ctx.strokeStyle = accent; ctx.lineWidth = 2; ctx.beginPath();
    ys.forEach((v, i) => {
      const x = pad + (i + 0.5) * (plotW / n), y = yToPx(v);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  } else {
    ctx.fillStyle = accent;
    ys.forEach((v, i) => {
      const x = pad + (i + 0.5) * (plotW / n), y = yToPx(v);
      ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
    });
  }
  ctx.fillStyle = fg;
  const step = Math.ceil(n / 12);
  xs.forEach((v, i) => {
    if (i % step) return;
    const x = pad + (i + 0.5) * (plotW / n);
    ctx.save(); ctx.translate(x, H - pad + 14); ctx.rotate(-0.5);
    ctx.fillText(String(v).slice(0, 14), 0, 0); ctx.restore();
  });
}

renderChart();
renderTable();
</script>
</body>
</html>
"""


def run_plot(
    sql: str,
    *,
    title: str = "Opik report",
    kind: str = "bar",
    x: str | None = None,
    y: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run ``sql`` over the cache and write a self-contained HTML report.

    ``kind`` is one of ``bar``, ``line``, ``scatter``. ``x``/``y`` name the
    columns to chart; if omitted, the first two columns are used.
    """
    settings = settings or get_settings()

    if not local_persistence_enabled(settings):
        raise ToolError(LOCAL_ONLY_HINT)

    if kind not in _CHART_KINDS:
        raise ToolError(f"kind must be one of {_CHART_KINDS}, got {kind!r}.")
    if not _READONLY_RE.match(sql):
        raise ToolError("Only read-only SELECT/WITH queries are allowed.")

    try:
        df = query_sql(sql, settings)
    except AnalyticsExtraMissing as e:
        raise ToolError(str(e)) from e
    except Exception as e:
        avail = ", ".join(cached_entity_types(settings)) or "(nothing cached yet)"
        raise ToolError(f"Query failed: {e}\nCached entity views: {avail}.") from e

    if df.empty:
        raise ToolError("Query returned no rows; nothing to plot.")

    columns = [str(c) for c in df.columns]
    x_col = x or columns[0]
    y_col = y or (columns[1] if len(columns) > 1 else columns[0])
    for col, label in ((x_col, "x"), (y_col, "y")):
        if col not in columns:
            raise ToolError(f"{label}={col!r} not in result columns {columns}.")

    def _clean(v: Any) -> Any:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    raw_rows = json.loads(json.dumps(df.head(_MAX_POINTS).to_dict(orient="records"), default=str))
    rows = [{k: _clean(v) for k, v in r.items()} for r in raw_rows]
    payload = {
        "rows": rows,
        "columns": columns,
        "chart": {"kind": kind, "x": x_col, "y": y_col},
    }

    meta = (
        f"{len(df)} rows · {html.escape(kind)} chart · {html.escape(x_col)} × {html.escape(y_col)}"
    )
    document = (
        _HTML_TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__META__", meta)
        .replace("__DATA__", json.dumps(payload, allow_nan=False).replace("</", "<\\/"))
    )

    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "report"
    out_path = reports_dir / f"{slug}-{uuid.uuid4().hex[:8]}.html"
    out_path.write_text(document, encoding="utf-8")

    return {
        "path": str(out_path),
        "uri": out_path.as_uri(),
        "row_count": int(len(df)),
        "plotted_rows": min(len(df), _MAX_POINTS),
        "title": title,
    }


__all__ = ["run_plot"]
