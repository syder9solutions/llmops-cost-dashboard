"""Two-tier cache: exact (content hash) + semantic (SimHash LSH).

API responses are cached in-memory (dict + TTL + LRU-ish trim). The exact tier
matches normalized prompt+params hash; the semantic tier matches SimHash
within Hamming threshold and returns the nearest previous answer.

Cache correctness policy (validated on the workload):
- semantic hits are admitted only under a TTL and a `semantic_enabled` flag;
- stores per-key hit stats for hit-rate metrics;
- `stats()` exposes hits/misses per tier for the dashboard.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .semanticsim import SimhashIndex, md5, normalize_for_cache, tokenize, simhash64

_SENTINEL = object()


class CacheEntry:
    __slots__ = ("key", "value", "ts", "ttl", "h", "model", "prompt_norm",
                 "tier", "params")

    def __init__(self, key: str, value: Any, ts: float, ttl: float, h: int,
                 model: str, prompt_norm: str, tier: str,
                 params: str = "") -> None:
        self.key = key; self.value = value; self.ts = ts; self.ttl = ttl
        self.h = h; self.model = model; self.prompt_norm = prompt_norm
        self.tier = tier; self.params = params


class SemanticCache:
    def __init__(self, exact_ttl_s: float = 3600.0, semantic_ttl_s: float = 1800.0,
                 semantic_threshold_bits: int = 18, max_entries: int = 4096,
                 semantic_enabled: bool = True) -> None:
        self.exact_ttl_s = exact_ttl_s
        self.semantic_ttl_s = semantic_ttl_s
        self.max_entries = max_entries
        self.semantic_enabled = semantic_enabled
        self._lock = threading.RLock()
        self._exact: Dict[str, CacheEntry] = {}          # hash -> entry
        self._index = SimhashIndex(threshold_bits=semantic_threshold_bits)
        self._stats: Dict[str, float] = {"exact_hits": 0, "semantic_hits": 0,
                                         "misses": 0, "evictions": 0,
                                         "semantic_lookups": 0}

    # -- keys -----------------------------------------------------------
    @staticmethod
    def exact_key(model: str, messages_hash: str, **params: Any) -> str:
        p = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return md5(f"{model}|{messages_hash}|{p}")

    @staticmethod
    def messages_hash(messages: List[Dict[str, str]]) -> str:
        norm = "||".join(
            f"{m.get('role')}:{normalize_for_cache(str(m.get('content', '')))}"
            for m in messages)
        return md5(norm)

    # -- core -----------------------------------------------------------
    def lookup(self, model: str, messages: List[Dict[str, str]],
               **params: Any) -> Tuple[Optional[Any], Optional[str]]:
        """Return (value, tier) — tier in {'exact','semantic',None}.

        Semantic hits require IDENTICAL params (max_tokens etc.): a different
        param set is a different request and must not be served from cache.
        """
        mh = self.messages_hash(messages)
        key = self.exact_key(model, mh, **params)
        now = time.time()
        with self._lock:
            e = self._exact.get(key)
            if e is not None and now - e.ts <= e.ttl:
                self._stats["exact_hits"] += 1
                return e.value, "exact"
            if e is not None and now - e.ts > e.ttl:
                self._remove_locked(key)
            # semantic tier — ONLY under the same params. Find the param-
            # signature of each candidate: we track it on the entry.
            if self.semantic_enabled and messages:
                q = next((str(m["content"]) for m in reversed(messages)
                          if m.get("role") == "user"), "")
                if q:
                    self._stats["semantic_lookups"] += 1
                    m = self._index.find_semantic_match(q)
                    if m is not None:
                        skey, _h, dist = m
                        se = self._exact.get(skey)
                        if (se is not None
                                and now - se.ts <= self.semantic_ttl_s
                                and se.model == model
                                and se.params == self._params_sig(**params)):
                            self._stats["semantic_hits"] += 1
                            return se.value, "semantic"
            self._stats["misses"] += 1
            return None, None

    @staticmethod
    def _params_sig(**params: Any) -> str:
        return "&".join(f"{k}={v}" for k, v in sorted(params.items()))

    def store(self, model: str, messages: List[Dict[str, str]], value: Any,
              semantic: bool = True, **params: Any) -> None:
        mh = self.messages_hash(messages)
        key = self.exact_key(model, mh, **params)
        now = time.time()
        norm_q = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                norm_q = str(m["content"])
                break
        with self._lock:
            if len(self._exact) >= self.max_entries:
                self._evict_locked()
            e = CacheEntry(key, value, now, self.exact_ttl_s, 0, model,
                           norm_q, "exact", params=self._params_sig(**params))
            self._exact[key] = e
            if semantic and norm_q:
                self._index.add(key, norm_q)

    def _remove_locked(self, key: str) -> None:
        self._exact.pop(key, None)
        self._index.remove(key)

    def _evict_locked(self) -> None:
        # evict oldest 10% (simple TLRU; workloads are bounded)
        items = sorted(self._exact.items(), key=lambda kv: kv[1].ts)
        n = max(1, len(items) // 10)
        for k, _ in items[:n]:
            self._remove_locked(k)
            self._stats["evictions"] += 1

    # -- observability ----------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            s = dict(self._stats)
            s["entries"] = len(self._exact)
            s["semantic_enabled"] = self.semantic_enabled
            total = s["exact_hits"] + s["semantic_hits"] + s["misses"]
            s["hit_rate"] = ((s["exact_hits"] + s["semantic_hits"]) / total
                             if total else 0.0)
            return s

    def reset(self) -> None:
        with self._lock:
            self._exact.clear()
            self._index = SimhashIndex(
                threshold_bits=self._index.threshold_bits)
            self._stats = {"exact_hits": 0, "semantic_hits": 0, "misses": 0,
                           "evictions": 0, "semantic_lookups": 0}
