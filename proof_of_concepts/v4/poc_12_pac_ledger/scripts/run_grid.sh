#!/usr/bin/env bash
# The exp_29 grid. Run ONLY after the dawn-field-theory registration is sealed.
#   run_grid.sh proxy   kappa in {0, 0.5, 1, 2, inf} x seeds 1-3 at n = 1000; aggregate
#   run_grid.sh full    the same at n = 4000; aggregate
# Results go to results/<phase>/ — one JSON + one _pos.npz per run, append-only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; POC="$(dirname "$HERE")"; REPO="$(cd "$POC/../../.." && pwd)"
PY="${PY:-}"
if [ -z "$PY" ]; then for c in "$REPO/.venv/bin/python" "$(command -v python)" "$(command -v python3)"; do
  [ -n "$c" ] && "$c" -c "import torch" 2>/dev/null && PY="$c" && break; done; fi
[ -n "$PY" ] || { echo "run_grid.sh: no python with torch found — set PY=" >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
RUN="$PY $HERE/exp_03_ledger_arms.py"; AGG="$PY $HERE/exp_04_aggregate.py"
SEEDS="${SEEDS:-1 2 3}"; KAPPAS="${KAPPAS:-0 0.5 1 2 inf}"
phase="${1:?phase: proxy | full}"
case "$phase" in
  proxy|full)
    OUT="$POC/results/$phase"; mkdir -p "$OUT"
    for s in $SEEDS; do for k in $KAPPAS; do $RUN --size "$phase" --kappa "$k" --seed "$s" --out-dir "$OUT"; done; done
    $AGG --size "$phase" --results-dir "$OUT" ;;
  *) echo "unknown phase $phase" >&2; exit 2 ;;
esac
echo "phase $phase done"
