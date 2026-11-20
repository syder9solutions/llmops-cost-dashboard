"""64-bit SimHash + band-LSH semantic fingerprinting (zero deps).

Semantic caching needs "are these two prompts about the same thing?" without
an embedding API. We tokenize → weight tokens by informativeness (stopword
down-weighting + length) → 64-bit SimHash → Hamming-band LSH for candidate
retrieval → exact Hamming threshold decision.

SimHash with token-unigram features approximates cosine similarity on the
hashed bag-of-words space: Hamming distance ≤ T ⇒ near-duplicate meaning for
our workload classes (paraphrases, casing/punctuation variants). The exact
similarity bound is workload-dependent; we measure it on the real workload
(evals/semantic_quality.json) rather than assume it.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_SPACE_RE = re.compile(r"\s+")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "at", "by", "from", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "as", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "do", "does", "did", "can", "could", "would", "should", "will", "shall",
    "what", "which", "who", "whom", "how", "when", "where", "why", "there",
}


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit — deterministic across runs (unlike hash())."""
    h = 0xcbf29ce484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def simhash64(tokens: Iterable[str]) -> int:
    """Weighted SimHash. Weight: tf × stopword-downweight × length bonus."""
    counts = Counter(tokens)
    if not counts:
        return 0
    bits = [0.0] * 64
    max_tf = max(counts.values())
    for tok, tf in counts.items():
        if tok in _STOPWORDS:
            w = 0.1
        else:
            w = 1.0 + 0.25 * min(len(tok), 8) / 8.0
        w *= 0.5 + 0.5 * (tf / max_tf)
        h = _fnv1a64(tok.encode("utf-8"))
        for i in range(64):
            if h & (1 << i):
                bits[i] += w
            else:
                bits[i] -= w
    out = 0
    for i in range(64):
        if bits[i] > 0:
            out |= (1 << i)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def normalize_for_cache(text: str) -> str:
    """Lowercase, collapse whitespace, strip punct-only noise."""
    t = _SPACE_RE.sub(" ", text.strip().lower())
    t = re.sub(r"[^\w\s'-]", " ", t)
    return _SPACE_RE.sub(" ", t).strip()


class SimhashIndex:
    """Band-LSH: split 64 bits into B bands; identical band-bits collide in
    bucket (b, band_value). Candidates sharing ≥1 band are checked exactly."""

    def __init__(self, bands: int = 8, threshold_bits: int = 18) -> None:
        assert 64 % bands == 0
        self.bands = bands
        self.band_width = 64 // bands
        self.threshold_bits = threshold_bits
        self._buckets: Dict[Tuple[int, int], List[str]] = {}
        self._items: Dict[str, Tuple[str, int]] = {}  # key -> (text_norm, hash)

    def _band_values(self, h: int) -> List[int]:
        mask = (1 << self.band_width) - 1
        return [(h >> (i * self.band_width)) & mask for i in range(self.bands)]

    def add(self, key: str, text: str) -> int:
        norm = normalize_for_cache(text)
        h = simhash64(tokenize(norm))
        self._items[key] = (norm, h)
        for b, v in enumerate(self._band_values(h)):
            self._buckets.setdefault((b, v), []).append(key)
        return h

    def remove(self, key: str) -> None:
        item = self._items.pop(key, None)
        if item is None:
            return
        _, h = item
        for b, v in enumerate(self._band_values(h)):
            lst = self._buckets.get((b, v))
            if lst and key in lst:
                lst.remove(key)
                if not lst:
                    del self._buckets[(b, v)]

    def candidates(self, text: str) -> List[str]:
        h = simhash64(tokenize(normalize_for_cache(text)))
        out: set = set()
        for b, v in enumerate(self._band_values(h)):
            out.update(self._buckets.get((b, v), ()))
        return list(out)

    def find_semantic_match(self, text: str) -> Optional[Tuple[str, int, int]]:
        """Return (key, hash, hamming_dist) of best candidate within threshold."""
        h = simhash64(tokenize(normalize_for_cache(text)))
        best: Optional[Tuple[str, int, int]] = None
        for key in self.candidates(text):
            _, kh = self._items[key]
            d = hamming(h, kh)
            if d <= self.threshold_bits and (best is None or d < best[2]):
                best = (key, kh, d)
        return best

    def __len__(self) -> int:
        return len(self._items)


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()  # noqa: S324 - cache key, not security
