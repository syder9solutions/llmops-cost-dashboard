"""Deterministic synthetic support-assistant workload (seed 20261212).

Simulates a multi-feature LLM app (chat_support, doc_summary, code_explain,
billing_qa) with realistic repetition structure:
  - ~30% exact repeats (same user asks the same thing again)
  - ~20% paraphrases (semantic cache tier target)
  - ~50% novel queries
Deterministic: same seed → same workload, so experiments are comparable.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

SEED = 20261212

_TOPICS = [
    ("password reset", "How do I reset my password if I no longer have access to my registered email?"),
    ("refund policy", "What is your policy on refunds for annual subscriptions after the trial period ends?"),
    ("api limits", "What are the current rate limits for the paid API tier and how are they calculated?"),
    ("invoice download", "Where can I download a PDF copy of invoices from previous billing cycles?"),
    ("team seats", "How do I add or remove team seats and when does the proration take effect?"),
    ("sso setup", "How do I configure SAML single sign-on for our organization's identity provider?"),
    ("data export", "Can I export all of my account data in a machine-readable format for compliance?"),
    ("uptime status", "Where can I find the status page and historical uptime statistics for the service?"),
    ("mobile app", "Is there an offline mode in the mobile app and how does sync conflict resolution work?"),
    ("webhook retries", "What is the retry schedule for failed webhook deliveries and can I replay them?"),
    ("audit logs", "How long are audit logs retained on the enterprise plan and who can access them?"),
    ("integrations", "Which third-party integrations are officially supported and maintained this quarter?"),
]

_PARAPHRASE_TEMPLATES = [
    "Hey, quick question — {q}",
    "{q} Also, is there a doc for this?",
    "Sorry if this is obvious, but {q_lower}",
    "{q} (asked before, still unclear)",
    "Hi team, {q_lower} Thanks!",
    "Could you help me with this: {q_lower}",
]

_FEATURES = ["chat_support", "doc_summary", "code_explain", "billing_qa"]
_USERS = ["alice", "bob", "carol", "dave", "erin"]

_SYSTEM = {
    "chat_support": "You are a concise customer-support assistant. Answer in 2-3 sentences.",
    "doc_summary": "Summarize the user's document request in 2 sentences.",
    "code_explain": "You are a senior engineer. Explain briefly.",
    "billing_qa": "You are a billing specialist. Be precise and brief.",
}


def build_workload(n: int = 300, seed: int = SEED,
                   mix: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """n requests: mix = {repeat: 0.30, paraphrase: 0.20, novel: 0.50}."""
    mix = mix or {"repeat": 0.30, "paraphrase": 0.20, "novel": 0.50}
    rng = random.Random(seed)
    seen: List[str] = []          # questions already emitted (repeat pool)
    para_pool: List[str] = []     # paraphrase targets (paraphrase tier)
    out: List[Dict[str, Any]] = []

    def novel_question() -> str:
        # topic + a distinguishing detail so each novel ask is unique
        topic, base = rng.choice(_TOPICS)
        flavor = rng.choice([
            "Context: we are a 40-person company.",
            "This is for our staging environment.",
            "Asking on behalf of a customer.",
            "We hit this yesterday around 4pm.",
            "Second time this month.",
            "Our account id is #{}.".format(rng.randint(10000, 99999)),
            "Priority question from leadership.",
            "It worked last week.",
            "We are evaluating vendors.",
            "Need an answer before renewal.",
        ])
        return f"{base} {flavor}"

    for i in range(n):
        feature = rng.choices(_FEATURES, weights=[40, 25, 20, 15])[0]
        user = rng.choice(_USERS)
        r = rng.random()
        if seen and r < mix["repeat"]:
            q = rng.choice(seen)
        elif para_pool and r < mix["repeat"] + mix["paraphrase"]:
            base_q = rng.choice(para_pool)
            tpl = rng.choice(_PARAPHRASE_TEMPLATES)
            q = tpl.format(q=base_q, q_lower=base_q[0].lower() + base_q[1:])
            if q not in seen:
                seen.append(q)
        else:
            q = novel_question()
            seen.append(q)
            para_pool.append(q)
            if len(para_pool) > 60:
                para_pool.pop(0)
        out.append({
            "i": i, "feature": feature, "user": user,
            "messages": [
                {"role": "system", "content": _SYSTEM[feature]},
                {"role": "user", "content": q},
            ],
            "max_tokens": int(os.environ.get("MAX_TOKENS", "120")),
        })
    return out


def paraphrase_pairs(n: int = 60, seed: int = SEED + 1) -> List[Dict[str, Any]]:
    """(base, paraphrase) pairs for semantic-cache quality measurement."""
    rng = random.Random(seed)
    pairs = []
    for j in range(n):
        topic, base = _TOPICS[j % len(_TOPICS)]
        flavor = "Ref #{}.".format(rng.randint(1000, 9999))
        b = f"{base} {flavor}"
        tpl = rng.choice(_PARAPHRASE_TEMPLATES)
        p = tpl.format(q=b, q_lower=b[0].lower() + b[1:])
        pairs.append({"topic": topic, "base": b, "paraphrase": p})
    return pairs


if __name__ == "__main__":
    import json
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    wl = build_workload(n)
    uniq = len({w["messages"][1]["content"] for w in wl})
    print(json.dumps({"n": n, "unique_prompts": uniq,
                      "repeat_share": round(1 - uniq / n, 3)}, indent=1))
