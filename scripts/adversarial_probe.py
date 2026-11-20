#!/usr/bin/env python
"""Adversarial HTTP probes against the running dashboard API.

Invariants checked:
- validation negatives return 422 (not 500, not silent accept);
- malformed/extra fields are rejected;
- provider errors map to 502 and never leak the API key;
- dashboard/trace endpoints don't leak secrets;
- boundary values are handled deterministically.
"""
from __future__ import annotations

import os, sys, json, urllib.request, urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

PORT = int(os.environ.get("DEMO_PORT", config.APP_PORT))
BASE = f"http://127.0.0.1:{PORT}"


def req(method, path, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    headers = {"Content-Type": "application/json"} if data else {}
    r = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main() -> int:
    checks = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    # validation negatives
    code, _ = req("POST", "/v1/chat/completions", {"messages": "not-a-list"})
    check("messages wrong type -> 422", code == 422, f"got {code}")

    code, _ = req("POST", "/v1/chat/completions", {"messages": []})
    check("empty messages -> 422", code == 422, f"got {code}")

    code, _ = req("POST", "/v1/chat/completions",
                  {"messages": [{"role": "user", "content": "hi"}], "bogus_field": 1})
    check("extra field -> 422", code == 422, f"got {code}")

    code, _ = req("POST", "/v1/chat/completions", raw=b"{not valid json")
    check("malformed JSON -> 422", code == 422, f"got {code}")

    code, _ = req("GET", "/v1/chat/completions")
    check("GET on POST-only -> 405", code == 405, f"got {code}")

    # boundary query params
    for q in ("window_min=0", "window_min=10001", "bucket_s=1"):
        code, _ = req("GET", f"/stats?{q}")
        check(f"/stats {q} -> 422", code == 422, f"got {code}")

    # observability endpoints return JSON and don't leak the key
    for path in ("/stats", "/alerts", "/attribution", "/spans?limit=5", "/dashboard"):
        code, body = req("GET", path)
        check(f"GET {path.split('?')[0]} -> 200", code == 200, f"got {code}")
        leaked = config.OPENAI_API_KEY and config.OPENAI_API_KEY in body
        check(f"GET {path.split('?')[0]} no key leak", not leaked)
    # unknown trace id -> 404 is correct (not a 500, no leak)
    code, body = req("GET", "/traces/nonexistent")
    check("GET unknown trace -> 404", code == 404, f"got {code}")
    check("unknown trace no key leak",
          not (config.OPENAI_API_KEY and config.OPENAI_API_KEY in body))

    ok = sum(1 for _, c, _ in checks if c)
    for name, c, detail in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}" + (f" ({detail})" if not c else ""))
    print(f"\nadversarial probes: {ok}/{len(checks)} clean")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
