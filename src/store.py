"""Append-only SQLite trace store. WAL mode, indexed, thread-safe via a lock.

Rows are spans; `attrs` is JSON. The store is the single source of truth for
every metric, dashboard number and alert decision.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any, Dict, Iterable, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
  span_id    TEXT PRIMARY KEY,
  trace_id   TEXT NOT NULL,
  parent_id  TEXT,
  name       TEXT NOT NULL,
  kind       TEXT NOT NULL,
  ts         REAL NOT NULL,
  dur_ms     REAL NOT NULL DEFAULT 0,
  status     TEXT NOT NULL DEFAULT 'ok',
  error_type TEXT,
  attrs      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_kind_ts ON spans(kind, ts);
CREATE INDEX IF NOT EXISTS idx_spans_name_ts ON spans(name, ts);
CREATE INDEX IF NOT EXISTS idx_spans_ts ON spans(ts);
"""

SCHEMA_VERSION = 1


class TraceStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- write ----------------------------------------------------------
    def add_span_row(self, row: Dict[str, Any]) -> None:
        attrs = row.get("attrs", {})
        if not isinstance(attrs, str):
            attrs = json.dumps(attrs, sort_keys=True, default=str)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO spans VALUES (?,?,?,?,?,?,?,?,?,?)",
                (row["span_id"], row["trace_id"], row.get("parent_id"),
                 row["name"], row["kind"], row["ts"], row.get("dur_ms", 0.0),
                 row.get("status", "ok"), row.get("error_type"),
                 attrs),
            )

    def add_spans(self, rows: Iterable[Dict[str, Any]]) -> None:
        rows = list(rows)
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO spans VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(r["span_id"], r["trace_id"], r.get("parent_id"), r["name"],
                  r["kind"], r["ts"], r.get("dur_ms", 0.0),
                  r.get("status", "ok"), r.get("error_type"),
                  json.dumps(r.get("attrs", {}), sort_keys=True, default=str))
                 for r in rows],
            )
            self._conn.commit()

    def flush(self) -> None:
        with self._lock:
            self._conn.commit()

    # -- read -----------------------------------------------------------
    def _row_to_dict(self, r: sqlite3.Row) -> Dict[str, Any]:
        d = dict(r)
        try:
            d["attrs"] = json.loads(d.get("attrs") or "{}")
        except json.JSONDecodeError:
            d["attrs"] = {}
        return d

    def query(self, kind: Optional[str] = None, name: Optional[str] = None,
              status: Optional[str] = None, trace_id: Optional[str] = None,
              since_ts: Optional[float] = None, until_ts: Optional[float] = None,
              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM spans WHERE 1=1"
        args: List[Any] = []
        if kind:
            sql += " AND kind=?"; args.append(kind)
        if name:
            sql += " AND name=?"; args.append(name)
        if status:
            sql += " AND status=?"; args.append(status)
        if trace_id:
            sql += " AND trace_id=?"; args.append(trace_id)
        if since_ts is not None:
            sql += " AND ts>=?"; args.append(since_ts)
        if until_ts is not None:
            sql += " AND ts<=?"; args.append(until_ts)
        sql += " ORDER BY ts ASC"
        if limit:
            sql += " LIMIT ?"; args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def llm_spans(self, since_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        return self.query(kind="llm_call", since_ts=since_ts)

    def count(self, kind: Optional[str] = None) -> int:
        sql = "SELECT COUNT(*) FROM spans"
        args: List[Any] = []
        if kind:
            sql += " WHERE kind=?"; args.append(kind)
        with self._lock:
            return self._conn.execute(sql, args).fetchone()[0]

    def raw_spans_export(self) -> List[Dict[str, Any]]:
        """Full export for verification scripts (column order = schema)."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM spans ORDER BY ts").fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -- lifecycle ------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()

    def wipe(self) -> None:
        with self._lock:
            self._conn.executescript("DELETE FROM spans;")
            self._conn.commit()
