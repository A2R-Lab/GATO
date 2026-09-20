#!/usr/bin/env bash
# ============================================================================
# Contact-wipe pool: all three arms over the 24-scenario grid.
#
#   bash examples/contact-task/run_wipe_pool.sh [DEPTH_M]
#
# pos/ucone run on the DEFAULT modules; the fc arm runs on the fc module
# VARIANT (bsqpN*_iiwa14_fc.so, side by side in python/gato — no .so swap).
#
# Correctness-class runs (fixed pacing, bit-deterministic) — busy-box safe.
# Solve-time stats are recorded but are NOT quiet-box numbers.
# ============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PY=${PY:-$REPO/.venv/bin/python}     # the project venv (tools/install.sh --test: pinocchio + mujoco)
DEPTH=${1:-0.002}
OUT=$HERE/data/wipe_$(date -u +%Y%m%d_%H%M%S)

mkdir -p "$OUT"
echo "pool -> $OUT  (depth ${DEPTH} m)"

echo "== arm: pos =="
"$PY" "$HERE/run_wipe_cell.py" --arm pos --depth "$DEPTH" --out "$OUT" | tee "$OUT/pos.log"

echo "== arm: ucone =="
"$PY" "$HERE/run_wipe_cell.py" --arm ucone --depth "$DEPTH" --out "$OUT" | tee "$OUT/ucone.log"

echo "== arm: fc (the fc module variant, bsqpN*_iiwa14_fc) =="
"$PY" "$HERE/run_wipe_cell.py" --arm fc --depth "$DEPTH" --out "$OUT" | tee "$OUT/fc.log"

"$PY" "$HERE/summarize_wipe.py" "$OUT" | tee "$OUT/SUMMARY.txt"
echo "done -> $OUT"
