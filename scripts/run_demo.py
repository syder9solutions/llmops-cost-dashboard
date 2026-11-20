#!/usr/bin/env python
"""Boot the API server, health-check it, exercise key endpoints over real
HTTP, print results, and shut down. Never daemonizes (boot_check style)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402

PORT = int(os.environ.get("DEMO_PORT", config.APP_PORT))


def req(method, path, body=None, timeout=60):
    url = f"http://127.0.0.1:{PORT}{path}"
    data = None
    headers = {}
    if body is not None:
        import json
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def main() -> int:
    env = dict(os.environ)
    proc = subprocess.Popen(
        [".venv/bin/python", "-m", "uvicorn", "app.main:app",
         "--host", config.APP_HOST, "--port", str(PORT), "--log-level", "warning"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        # wait for boot
        for _ in range(60):
            try:
                code, _ = req("GET", "/health", timeout=2)
                if code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            print("BOOT FAILED")
            return 1
        print("boot: OK")

        code, body = req("GET", "/health")
        print(f"GET /health → {code} {body[:120].decode()}")

        code, body = req("GET", "/dashboard")
        ok_dash = code == 200 and b"LLMOps" in body
        print(f"GET /dashboard → {code} html={len(body)}B contains-title={ok_dash}")

        code, body = req("GET", "/stats?window_min=60")
        print(f"GET /stats → {code} bytes={len(body)}")

        code, body = req("GET", "/prices")
        print(f"GET /prices → {code} bytes={len(body)}")

        # a real live completion through the instrumented gateway
        code, body = req("POST", "/v1/chat/completions", {
            "model": config.TEXT_MODEL,
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 16, "feature": "boot_check", "user": "system",
            "cache": "on"})
        print(f"POST /v1/chat/completions → {code} bytes={len(body)}")
        import json
        if code == 200:
            j = json.loads(body)
            tr = j.get("tracing", {})
            print(f"  trace={tr.get('trace_id','')[:8]}… cost=${tr.get('cost_usd')} "
                  f"tok={tr.get('prompt_tokens')}/{tr.get('completion_tokens')}")
            # cache hit second time
            code2, body2 = req("POST", "/v1/chat/completions", {
                "model": config.TEXT_MODEL,
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 16, "feature": "boot_check", "user": "system",
                "cache": "on"})
            tr2 = json.loads(body2).get("tracing", {})
            print(f"  repeat → cache={tr2.get('cache')} cost=${tr2.get('cost_usd')}")

        # validation negatives
        code, _ = req("POST", "/v1/chat/completions", {"messages": []})
        print(f"POST empty messages → {code} (expect 422/502)")
        code, _ = req("POST", "/v1/chat/completions", {
            "model": "antigravity/gemini-3-flash",
            "messages": [{"role": "user", "content": "x"}]})
        print(f"POST banned model → {code} (expect 422)")

        print("DEMO OK" if ok_dash else "DASHBOARD BROKEN")
        return 0 if ok_dash else 1
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
