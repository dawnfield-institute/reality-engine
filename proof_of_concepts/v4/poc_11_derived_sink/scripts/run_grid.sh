#!/usr/bin/env bash
# The exp_28 grid, phase by phase. Run ONLY after the dawn-field-theory registration is sealed.
#
#   run_grid.sh proxy            B0, S(tau) x 3 tau, R per S, L, S+L(tau=10), S(tau=10) at md 0.90/0.98 -- x 3 seeds; aggregate
#   run_grid.sh proxy_d  <tau*>  D x 3 seeds matched to S(tau*); re-aggregate            (tau* from the dft scorer's proxy pass)
#   run_grid.sh full     <tau*>  B0, S(tau*), R, D x 3 seeds at n = 4000; aggregate
#
# Results go to results/<phase-dir>/ (proxy/ or full/), one JSON + one _pos.npz per run, append-only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POC="$(dirname "$HERE")"; REPO="$(cd "$POC/../../.." && pwd)"
PY="${PY:-}"                      # a python with torch: the repo venv, else the current one; or set PY=
if [ -z "$PY" ]; then for c in "$REPO/.venv/bin/python" "$(command -v python)" "$(command -v python3)"; do
  [ -n "$c" ] && "$c" -c "import torch" 2>/dev/null && PY="$c" && break; done; fi
[ -n "$PY" ] || { echo "run_grid.sh: no python with torch found — set PY=" >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
RUN="$PY $HERE/exp_03_sink_arms.py"; AGG="$PY $HERE/exp_04_aggregate.py"
SEEDS="${SEEDS:-1 2 3}"; TAUS="${TAUS:-5 10 20}"
phase="${1:?phase: proxy | proxy_d <tau*> | full <tau*>}"

s_file() {  # size tau seed md -> the S run json (exactly one expected)
  local f; f=$(ls "$OUT"/exp_03_sink_arms_"$1"_S_tau"$2"_s"$3"_md"$4"_*.json 2>/dev/null | tail -1)
  [ -n "$f" ] || { echo "no S run for $1 tau=$2 seed=$3 md=$4" >&2; exit 1; }; echo "$f"; }
b0_file() {  # size seed -> the B0 run json
  local f; f=$(ls "$OUT"/exp_03_sink_arms_"$1"_B0_s"$2"_md0.95_*.json 2>/dev/null | tail -1)
  [ -n "$f" ] || { echo "no B0 run for $1 seed=$2" >&2; exit 1; }; echo "$f"; }

case "$phase" in
  proxy)
    OUT="$POC/results/proxy"; mkdir -p "$OUT"
    for s in $SEEDS; do $RUN --size proxy --arm B0 --seed "$s" --out-dir "$OUT"; done
    for t in $TAUS; do for s in $SEEDS; do
      $RUN --size proxy --arm S --tau "$t" --seed "$s" --out-dir "$OUT"
      $RUN --size proxy --arm R --seed "$s" --schedule-from "$(s_file proxy "$t" "$s" 0.95)" --out-dir "$OUT"
    done; done
    for s in $SEEDS; do
      $RUN --size proxy --arm L --seed "$s" --out-dir "$OUT"
      $RUN --size proxy --arm SL --tau 10 --seed "$s" --out-dir "$OUT"
      for md in 0.9 0.98; do $RUN --size proxy --arm S --tau 10 --seed "$s" --md "$md" --out-dir "$OUT"; done
    done
    $AGG --size proxy --results-dir "$OUT" ;;
  proxy_d)
    TS="${2:?tau*}"; OUT="$POC/results/proxy"
    for s in $SEEDS; do $RUN --size proxy --arm D --seed "$s" --match-from "$(s_file proxy "$TS" "$s" 0.95)" --baseline "$(b0_file proxy "$s")" --out-dir "$OUT"; done
    $AGG --size proxy --results-dir "$OUT" ;;
  full)
    TS="${2:?tau*}"; OUT="$POC/results/full"; mkdir -p "$OUT"
    for s in $SEEDS; do
      $RUN --size full --arm B0 --seed "$s" --out-dir "$OUT"
      $RUN --size full --arm S --tau "$TS" --seed "$s" --out-dir "$OUT"
      $RUN --size full --arm R --seed "$s" --schedule-from "$(s_file full "$TS" "$s" 0.95)" --out-dir "$OUT"
      $RUN --size full --arm D --seed "$s" --match-from "$(s_file full "$TS" "$s" 0.95)" --baseline "$(b0_file full "$s")" --out-dir "$OUT"
    done
    $AGG --size full --results-dir "$OUT" ;;
  *) echo "unknown phase $phase" >&2; exit 2 ;;
esac
echo "phase $phase done"
