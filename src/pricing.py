"""Per-model price table. $/1M tokens, source-controlled, versioned.

Cost of an LLM call = prompt_tokens*P_in + completion_tokens*P_out (÷1e6),
plus embed cost = tokens*P_embed (÷1e6). This is the LiteLLM/Langfuse-style
attribution model: price × usage, with prices pinned per model generation.

Prices reflect public list prices (USD, Nov 2026) of the underlying model
families the local gateway serves; the gateway itself bills internally, so
this table IS our provider bill. verify_cost_accuracy.py audits $ == Σ usage×price.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

PRICES: Dict[str, Dict[str, float]] = {
    # per 1M tokens USD
    "glm/glm-5.3": {"input": 0.60, "output": 2.20},
    "glm/glm-5.3-flash": {"input": 0.10, "output": 0.40},
    "glm/glm-5.3-flash-low": {"input": 0.05, "output": 0.15},
    "glm/glm-5.3-high": {"input": 2.00, "output": 8.00},
    "glm/glm-5.3-max": {"input": 4.00, "output": 12.00},
    "glm/glm-5.3-flashx": {"input": 0.15, "output": 0.60},
    "groq/llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "groq/openai/gpt-oss-120b": {"input": 0.15, "output": 0.75},
    "groq/openai/gpt-oss-20b": {"input": 0.10, "output": 0.30},
    "groq/qwen/qwen3.8-27b": {"input": 0.29, "out": 0.59, "output": 0.59},
    "gemini/gemini-3.1-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini/gemini-3.8-flash-high": {"input": 0.50, "output": 2.00},
    "nous-research/hy4-flash": {"input": 0.12, "output": 0.48},
    "nous-research/qwen/qwen4-max": {"input": 0.60, "output": 2.40},
    "nous-research/llama-3.1-70b-instruct": {"input": 0.55, "output": 0.75},
    "nous-research/inkling-2-small": {"input": 0.08, "output": 0.24},
    # embeddings: price per 1M tokens (single rate)
    "gemini/gemini-embedding-2": {"input": 0.10, "output": 0.0, "embedding": 0.10},
}

PRICES_VERSION = "2026-11-20.1"
# USD per 1M tokens is the industry convention (OpenAI/Anthropic/Google list pages)

# sensible fallback for models missing from the table (logged, never silent)
FALLBACK_PRICE = {"input": 0.50, "output": 1.50}


def resolve(model: str) -> Tuple[Dict[str, float], bool]:
    """Return (price, exact?) — exact=False means fallback used."""
    if model in PRICES:
        return PRICES[model], True
    return FALLBACK_PRICE, False


def chat_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> Tuple[float, bool]:
    p, exact = resolve(model)
    return (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1e6, exact


def embed_cost_usd(model: str, tokens: int) -> Tuple[float, bool]:
    p, exact = resolve(model)
    rate = p.get("embedding", p["input"])
    return tokens * rate / 1e6, exact


def price_info() -> Dict[str, object]:
    return {"version": PRICES_VERSION,
            "models": {k: dict(v) for k, v in PRICES.items()},
            "fallback": FALLBACK_PRICE}
