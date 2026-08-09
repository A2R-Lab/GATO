#!/usr/bin/env bash
# ============================================================================
# Contact-wipe pool: all three arms over the 24-scenario grid.
#
#   bash examples/contact-task/run_wipe_pool.sh [DEPTH_M]
#
# pos/ucone run on the DEFAULT modules in python/gato; the fc arm runs with the
# build_fc/modules .so swapped in (fc and default modules cannot co-load in one
# process — PyInit name collision), restored by trap on ANY exit.
#
# Correctness-class runs (fixed pacing, bit-deterministic) — busy-box safe.
# Solve-time stats are recorded but are NOT quiet-box numbers.
# ============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PY=${PY:-/home/plancher/Desktop/GRiD/.venv/bin/python}
DEPTH=${1:-0.002}
OUT=$HERE/data/wipe_$(date -u +%Y%m%d_%H%M%S)
PKG=$REPO/python/gato
FCMOD=$REPO/build_fc/modules
SUF=cpython-312-x86_64-linux-gnu.so

mkdir -p "$OUT"
echo "pool -> $OUT  (depth ${DEPTH} m)"

echo "== arm: pos =="
"$PY" "$HERE/run_wipe_cell.py" --arm pos --depth "$DEPTH" --out "$OUT" | tee "$OUT/pos.log"

echo "== arm: ucone =="
"$PY" "$HERE/run_wipe_cell.py" --arm ucone --depth "$DEPTH" --out "$OUT" | tee "$OUT/ucone.log"

echo "== arm: fc (module swap) =="
BK=$(mktemp -d)
restore() {
  for f in "$BK"/*.$SUF; do [[ -e $f ]] && cp -f "$f" "$PKG/"; done
  rm -rf "$BK"
}
trap restore EXIT
for f in "$FCMOD"/*.$SUF; do
  base=$(basename "$f")
  [[ -f $PKG/$base ]] && cp -f "$PKG/$base" "$BK/"
  cp -f "$f" "$PKG/"
done
"$PY" "$HERE/run_wipe_cell.py" --arm fc --out "$OUT" | tee "$OUT/fc.log"
restore
trap - EXIT

"$PY" "$HERE/summarize_wipe.py" "$OUT" | tee "$OUT/SUMMARY.txt"
echo "done -> $OUT"
