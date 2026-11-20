"""Offline test suite — no network anywhere. PYTHONPATH unset; conftest adds
repo root to sys.path (same convention as sibling projects)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# keep test runs from touching the real trace DB
os.environ.setdefault("TRACE_DB",
                      os.path.join(os.path.dirname(os.path.dirname(
                          os.path.abspath(__file__))), "data", "traces_test.db"))
