#!/usr/bin/env python
"""Boot server, run adversarial probe, shut down. Non-daemonizing."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src import config

PORT = int(os.environ.get("DEMO_PORT", config.APP_PORT))


def wait_ready(proc, timeout=25):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def main() -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    proc = subprocess.Popen(
        [".venv/bin/python", "-m", "uvicorn", "app.main:app",
         "--host", config.APP_HOST, "--port", str(PORT), "--log-level", "warning"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    try:
        if not wait_ready(proc):
            print("server failed to become ready", file=sys.stderr)
            return 1
        print("server ready")
        rc = subprocess.call([".venv/bin/python", "scripts/adversarial_probe.py"], env=env)
        return rc
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        print("server stopped")


if __name__ == "__main__":
    raise SystemExit(main())
