#!/usr/bin/env bash
# exp_32b (R2b) — the three grids of the registration SEALED at dawn-field-theory 3dbc0304.
# Fine sweep kappa 1.00..1.30 step 0.05; gravity arms g = 0.75 / 3.0 at kappa {1.00, 1.25};
# size arm n = 8000 at kappa {1.00, 1.25, 1.30}. Seeds 16 17 18, fresh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; POC="$(dirname "$HERE")"
export PY="${PY:?set PY to a python with torch}"
S="16 17 18"
echo "=== [1/4] fine sweep: 21 runs ==="
SEEDS="$S" KAPPAS="1.0 1.05 1.1 1.15 1.2 1.25 1.3" OUT="$POC/results/full_r2b_fine" "$HERE/run_grid.sh" full
echo "=== [2/4] gravity arm g = 0.75: 6 runs ==="
SEEDS="$S" KAPPAS="1.0 1.25" EXTRA="--g 0.75" OUT="$POC/results/full_r2b_g" "$HERE/run_grid.sh" full
echo "=== [3/4] gravity arm g = 3.0: 6 runs ==="
SEEDS="$S" KAPPAS="1.0 1.25" EXTRA="--g 3.0" OUT="$POC/results/full_r2b_g" "$HERE/run_grid.sh" full
echo "=== [4/4] size arm n = 8000: 9 runs ==="
SEEDS="$S" KAPPAS="1.0 1.25 1.3" OUT="$POC/results/full_r2b_n8000" "$HERE/run_grid.sh" double
echo "=== ALL FOUR PHASES DONE ==="
