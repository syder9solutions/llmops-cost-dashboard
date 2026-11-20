"""Zero-dependency HTML dashboard (single page, inline SVG charts, fetch→JSON).

Served at /dashboard by the FastAPI app. Data comes from /stats, /spans,
/alerts, /attribution — the same JSON any external dashboard would consume.
"""
from __future__ import annotations

import json
from typing import Any, Dict

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>LLMOps Cost &amp; Latency Dashboard</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: ui-monospace, "SF Mono", Menlo, monospace;
         background: #0d1117; color: #c9d1d9; margin: 0; padding: 24px; }
  h1 { font-size: 20px; color: #58a6ff; margin: 0 0 4px; }
  .sub { color: #8b949e; font-size: 12px; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
          gap: 12px; margin-bottom: 22px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 12px 14px; }
  .card .label { font-size: 11px; color: #8b949e; text-transform: uppercase;
                 letter-spacing: .05em; }
  .card .value { font-size: 22px; color: #e6edf3; margin-top: 4px; }
  .card .value.good { color: #3fb950; } .card .value.bad { color: #f85149; }
  section { margin-bottom: 26px; }
  h2 { font-size: 14px; color: #8b949e; text-transform: uppercase;
       letter-spacing: .05em; border-bottom: 1px solid #21262d;
       padding-bottom: 6px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #21262d; }
  th { color: #8b949e; font-weight: normal; }
  tr.firing td { color: #f85149; }
  .chart { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 10px; }
  .alert-chip { display:inline-block; padding:2px 8px; border-radius:10px;
                font-size:11px; margin-left:8px; }
  .chip-fire { background:#3d1a1a; color:#f85149; }
  .chip-ok { background:#12261a; color:#3fb950; }
  #refresh { float:right; font-size:12px; color:#58a6ff; cursor:pointer;
             background:none; border:none; }
  .tiny { color:#8b949e; font-size:11px; }
</style></head>
<body>
<h1>💸 LLMOps Cost &amp; Latency Dashboard
  <button id="refresh" onclick="load()">reload</button></h1>
<div class="sub">project 12 · traces from local SQLite store · p50/p95 nearest-rank ·
cost = tokens × price table (<span id="prices_ver">-</span>)</div>

<div class="grid" id="kpis"></div>

<section><h2>Cost &amp; latency over time</h2><div class="chart" id="ts_chart">
</div><div class="tiny">bars: requests (green=cached, red=error) ·
line: $ per bucket</div></section>

<section><h2>Latency distribution (live calls)</h2><div class="chart"
id="lat_chart"></div><div class="tiny">green p50 · yellow p95 · red p99 (ms)</div></section>

<section><h2>Attribution — top features by cost</h2>
<table id="feat_table"><thead><tr>
<th>feature</th><th>$</th><th>req</th><th>cached</th><th>tok in</th>
<th>tok out</th><th>$/1k req</th></tr></thead><tbody></tbody></table></section>

<section><h2>Cost by model</h2>
<table id="model_table"><thead><tr>
<th>model</th><th>$</th><th>req</th><th>cached</th><th>tok in</th><th>tok out</th>
</tr></thead><tbody></tbody></table></section>

<section><h2>Alerts</h2><table id="alert_table"><thead><tr>
<th>rule</th><th>state</th><th>value</th><th>threshold</th><th>reason</th>
</tr></thead><tbody></tbody></table></section>

<section><h2>Recent traces</h2><div class="tiny" id="recent">loading…</div></section>

<script>
const fmtUsd = x => '$' + (x ?? 0).toFixed(4);
const fmtMs = x => x == null ? '—' : Math.round(x) + ' ms';
const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

function kpis(s) {
  const m = s.metrics['*'] || {};
  const cards = [
    ['requests', m.requests ?? 0, ''],
    ['cost (total)', fmtUsd(m.cost_usd), ''],
    ['$/1k requests', fmtUsd(m.cost_per_1k_requests), ''],
    ['p50 latency', fmtMs(m.latency_p50_ms), ''],
    ['p95 latency', fmtMs(m.latency_p95_ms), m.latency_p95_ms > 3000 ? 'bad' : ''],
    ['cache hit-rate', ((m.cache_hit_rate ?? 0) * 100).toFixed(1) + '%',
      (m.cache_hit_rate ?? 0) >= 0.2 ? 'good' : ''],
    ['error rate', ((m.error_rate ?? 0) * 100).toFixed(1) + '%',
      (m.error_rate ?? 0) > 0.1 ? 'bad' : 'good'],
    ['tracing overhead (median)', fmtMs(s.overhead_ms_p50), ''],
  ];
  document.getElementById('kpis').innerHTML = cards.map(c =>
    `<div class="card"><div class="label">${c[0]}</div>
     <div class="value ${c[2]}">${c[1]}</div></div>`).join('');
}

function sparkline(rows, id, w, h) {
  if (!rows.length) { document.getElementById(id).innerHTML =
    '<div class="tiny">no data</div>'; return; }
  const maxR = Math.max(...rows.map(r => r.reqs), 1);
  const maxC = Math.max(...rows.map(r => r.cost_usd), 1e-9);
  const bw = Math.max(2, Math.min(28, (w - 40) / rows.length - 2));
  let bars = '', pts = [];
  rows.forEach((r, i) => {
    const x = 34 + i * (bw + 2);
    const hh = (r.reqs / maxR) * (h - 30);
    const cached_h = (r.cached / maxR) * (h - 30);
    const err_h = (r.errors / maxR) * (h - 30);
    bars += `<rect x="${x}" y="${h - 18 - (hh - cached_h - err_h)}" width="${bw}"
      height="${Math.max(0, hh - cached_h - err_h)}" fill="#2ea043" opacity="0.55"/>`;
    if (cached_h) bars += `<rect x="${x}" y="${h - 18 - cached_h}" width="${bw}"
      height="${cached_h}" fill="#58a6ff" opacity="0.8"/>`;
    if (err_h) bars += `<rect x="${x}" y="${h - 18 - err_h}" width="${bw}"
      height="${err_h}" fill="#f85149"/>`;
    pts.push([x + bw / 2, h - 18 - (r.cost_usd / maxC) * (h - 30)]);
  });
  const path = 'M' + pts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join('L');
  document.getElementById(id).innerHTML =
    `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">
       ${bars}<path d="${path}" fill="none" stroke="#d29922" stroke-width="1.5"/>
     </svg>`;
}

function latBars(m, id) {
  const vals = [['p50', m.latency_p50_ms, '#3fb950'],
                ['p95', m.latency_p95_ms, '#d29922'],
                ['p99', m.latency_p99_ms, '#f85149']];
  const mx = Math.max(...vals.map(v => v[1] ?? 0), 1);
  const w = 460, h = 120;
  const rects = vals.map((v, i) => {
    const bw = 90, x = 40 + i * 140;
    const bh = ((v[1] ?? 0) / mx) * (h - 40);
    return `<rect x="${x}" y="${h - 24 - bh}" width="${bw}" height="${bh}"
      fill="${v[2]}"/><text x="${x + bw / 2}" y="${h - 8}" fill="#8b949e"
      font-size="11" text-anchor="middle">${v[0]}</text>
      <text x="${x + bw / 2}" y="${h - 30 - bh}" fill="#e6edf3" font-size="11"
      text-anchor="middle">${v[1] == null ? '—' : Math.round(v[1])}</text>`;
  }).join('');
  document.getElementById(id).innerHTML =
    `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">${rects}</svg>`;
}

function tables(s) {
  const ft = document.querySelector('#feat_table tbody');
  ft.innerHTML = (s.top_features || []).map(f =>
    `<tr><td>${esc(f.feature)}</td><td>${fmtUsd(f.cost)}</td>
     <td>${f.requests}</td><td>${f.cached}</td><td>${f.tokens_in}</td>
     <td>${f.tokens_out}</td>
     <td>${fmtUsd(f.requests ? f.cost / f.requests * 1000 : 0)}</td></tr>`).join('');
  const mt = document.querySelector('#model_table tbody');
  mt.innerHTML = Object.entries(s.by_model || {}).map(([m, v]) =>
    `<tr><td>${esc(m)}</td><td>${fmtUsd(v.cost)}</td><td>${v.requests}</td>
     <td>${v.cached}</td><td>${v.tokens_in}</td><td>${v.tokens_out}</td></tr>`).join('');
  const at = document.querySelector('#alert_table tbody');
  at.innerHTML = Object.entries(s.alerts?.rules || {}).map(([k, r]) =>
    `<tr class="${r.firing ? 'firing' : ''}"><td>${esc(k)}</td>
     <td><span class="alert-chip ${r.firing ? 'chip-fire' : 'chip-ok'}">
     ${r.firing ? 'FIRING' : 'ok'}</span></td>
     <td>${typeof r.value === 'number' ? (+r.value).toFixed(3) : '—'}</td>
     <td>${typeof r.threshold === 'number' ? (+r.threshold).toFixed(3) : '—'}</td>
     <td>${esc(r.reason)}</td></tr>`).join('');
  document.getElementById('recent').innerHTML = (s.recent_traces || []).map(t =>
    `▸ ${esc(t.ts)} ${esc(t.name)} ${esc(t.model || '')} ` +
    `${t.cache_hit ? '<b style="color:#58a6ff">[' + esc(t.cache_hit) + ' hit]</b>' : ''}` +
    ` ${fmtMs(t.dur_ms)} ${fmtUsd(t.cost_usd)}`).join('<br>');
  document.getElementById('prices_ver').textContent = s.prices_version || '-';
}

async function load() {
  try {
    const r = await fetch('/stats?window_min=120&bucket_s=60');
    const s = await r.json();
    kpis(s); sparkline(s.time_buckets || [], 'ts_chart', 900, 150);
    latBars(s.metrics['*'] || {}, 'lat_chart'); tables(s);
  } catch (e) {
    document.getElementById('recent').textContent = 'load error: ' + e;
  }
}
load(); setInterval(load, 15000);
</script>
</body></html>
"""


def render() -> str:
    return PAGE


def stats_payload_json(store, window_min: float = 120.0) -> str:
    """Server-side equivalent of the dashboard's /stats fetch (used in tests)."""
    from . import pricing
    from .cost import top_cost_features
    from .metrics import llm_metrics, time_buckets
    import time as _t
    now = _t.time()
    spans = store.llm_spans(since_ts=now - window_min * 60)
    return json.dumps({
        "metrics": llm_metrics(spans),
        "time_buckets": time_buckets(spans[-400:]),
        "top_features": top_cost_features(spans),
        "prices_version": pricing.PRICES_VERSION,
    })
