#!/usr/bin/env python
"""Wipe the trace store (between experiment phases). Requires --yes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.store import TraceStore  # noqa: E402

if "--yes" not in sys.argv:
    print("refusing to wipe without --yes")
    raise SystemExit(2)

st = TraceStore(config.TRACE_DB)
n = st.count()
st.wipe()
st.close()
print(f"wiped {n} spans from {config.TRACE_DB}")
