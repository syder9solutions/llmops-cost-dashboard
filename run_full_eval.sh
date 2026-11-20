#!/bin/bash
# Full eval pipeline, sequential phases. One client at a time (shared throttle).
# Each phase writes its JSON artifact; a phase's nonzero exit (small residual
# provider error budget) is recorded but does NOT abort the remaining phases.
cd . || exit 1
export PYTHONPATH=""
PHASES="baseline cache_ab ablation alert_eval cost_audit report"

echo "=== PHASE baseline ==="
.venv/bin/python scripts/reset_traces.py --yes
N=200 WORKERS=2 .venv/bin/python scripts/run_baseline.py; echo "baseline rc=$?"

echo "=== PHASE cache_ab ==="
.venv/bin/python scripts/reset_traces.py --yes
N=150 WORKERS=2 .venv/bin/python scripts/run_cache_ab.py; echo "cache_ab rc=$?"

echo "=== PHASE ablation ==="
.venv/bin/python scripts/reset_traces.py --yes
N=60 WORKERS=2 .venv/bin/python scripts/run_ablation.py; echo "ablation rc=$?"

echo "=== PHASE alert_eval (offline) ==="
.venv/bin/python scripts/run_alert_eval.py; echo "alert_eval rc=$?"

echo "=== PHASE cost_audit ==="
.venv/bin/python scripts/verify_cost_accuracy.py; echo "cost_audit rc=$?"

echo "=== PHASE report ==="
.venv/bin/python scripts/make_report.py; echo "report rc=$?"

echo "=== PIPELINE DONE ==="
