"""Env + .env loader (zero deps). Reads workspace .env then os.environ."""
from __future__ import annotations

import os
from pathlib import Path


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_here = Path(__file__).resolve().parent
_load_env_file(_here.parent / ".env")

BANNED_MODEL_PREFIX = "antigravity/"  # orchestrator directive (project 11)

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:20129/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

TEXT_MODEL = os.environ.get("TEXT_MODEL", "glm/glm-5.3-flash")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini/gemini-embedding-2")
STRONG_MODEL = os.environ.get("STRONG_MODEL", "glm/glm-5.3")
# gateway-verified cheap model (groq/llama-3.1-8b-instant 400s on this gateway)
CHEAP_MODEL = os.environ.get("CHEAP_MODEL", "groq/openai/gpt-oss-20b")

DATA_DIR = os.environ.get("DATA_DIR", str(_here.parent / "data"))
TRACE_DB = os.environ.get("TRACE_DB", os.path.join(DATA_DIR, "traces.db"))
EVALS_DIR = str(_here.parent / "evals")

MAX_TOKENS_DEFAULT = int(os.environ.get("MAX_TOKENS_DEFAULT", "300"))
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "120"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
# shared minimum interval between outgoing provider requests (seconds).
# admission control: prevents our own concurrent workload from tripping the
# provider's per-org rate limit. 0 disables throttling.
MIN_CALL_INTERVAL_S = float(os.environ.get("MIN_CALL_INTERVAL_S", "0.75"))
# max wall-clock a single _post may spend retrying (incl. sleeps). A 429 whose
# retry_after/reset exceeds this is a hard quota wall and is raised immediately.
MAX_RETRY_WAIT_S = float(os.environ.get("MAX_RETRY_WAIT_S", "20"))

# alert baseline (tunable; scripts/run_alert_eval.py measures precision)
ALERT_BASELINE = {
    "error_rate": 0.10,
    "p95_ms": 3000.0,
    "p95_ratio": 1.5,
    "cost_per_1k": None,       # set after baseline phase
    "cost_ratio": 1.5,
    "psi_threshold": 0.25,
    "min_hit_rate": 0.05,
    "user_daily_budget": None,
}

# gateway listen address (loopback only by default)
APP_HOST = os.environ.get("APP_HOST", "127.0.0.1")
APP_PORT = int(os.environ.get("APP_PORT", "8124"))


def check_banned(model: str) -> None:
    if model and model.startswith(BANNED_MODEL_PREFIX):
        raise ValueError(
            f"model '{model}' uses banned prefix '{BANNED_MODEL_PREFIX}'")


def summary() -> dict:
    return {
        "openai_base_url": OPENAI_BASE_URL,
        "text_model": TEXT_MODEL, "embed_model": EMBED_MODEL,
        "strong_model": STRONG_MODEL, "cheap_model": CHEAP_MODEL,
        "trace_db": TRACE_DB, "app_host": APP_HOST, "app_port": APP_PORT,
        "has_api_key": bool(OPENAI_API_KEY),
    }
