#!/usr/bin/env bash
# scripts/bench.sh — the bench ladder, repo-owned.
#
# Replaces the retired .github/workflows/bench.yml: nightly + on-demand
# sweeps driven by the mlopt-nightly peregrin campaign. Every stage is
# non-blocking by design (machine-sensitive benches are a signal to
# look at, not a merge blocker) — the script always exits 0 unless the
# environment itself is broken (no uv, sync failure), and prints a
# per-stage summary at the end.
#
# Usage:
#     scripts/bench.sh
#
# Env knobs:
#     MLOPT_BENCH_QUICK   "0" runs the full sweeps (default "1" — the
#                         schedule lane; set 0 for a deep manual run)
#     MLOPT_BENCH_SUITES  comma-separated suite filter for
#                         `bench run-all --suites` (default: all)
set -uo pipefail

cd "$(dirname "$0")/.."

QUICK="${MLOPT_BENCH_QUICK:-1}"
SUITES="${MLOPT_BENCH_SUITES:-}"

echo "== bench: uv sync (dev + bench groups) =="
uv sync --frozen --group dev --group bench || { echo "bench: uv sync failed"; exit 1; }

echo "== bench: force CPU torch build =="
uv pip install --torch-backend cpu --reinstall torch || { echo "bench: torch cpu install failed"; exit 1; }

fails=0
run() {  # run <name> -- <cmd...>
    local name="$1"; shift
    [[ "${1:-}" == "--" ]] && shift
    echo "== bench: $name =="
    if "$@"; then echo "  [ok] $name"; else echo "  [FAIL] $name"; fails=$((fails+1)); fi
}

quick_arg=""
[ "$QUICK" = "1" ] && quick_arg="--quick"
suites_arg=()
[ -n "$SUITES" ] && suites_arg=(--suites "$SUITES")

run "run-all" -- .venv/bin/python -m bench run-all --device cpu $quick_arg "${suites_arg[@]}"
run "dashboard" -- .venv/bin/python -m bench dashboard
run "gate"      -- .venv/bin/python -m bench gate

echo "== bench done — $fails stage(s) failed (non-blocking) =="
exit 0
