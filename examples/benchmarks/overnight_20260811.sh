#!/usr/bin/env bash
# ============================================================================
# Overnight 2026-08-11 — the POST-GRiD-FOLD timing re-baseline (one night).
#
#   bash examples/benchmarks/overnight_20260811.sh
#
# SOLO GPU slot required. Sequential legs, each logged; a failed leg is
# recorded and the night continues. Everything lands under
# night_logs/overnight_<stamp>/.
#
# WHY TONIGHT: the a580308 fold changed every module binary (vendored regen
# incl. launch-config stem-match: the arms now BAKE their tuned tier/thread
# tables where they silently fell back to conservative before) — all held
# timing items re-baseline against this exact tree (@dae4022, CI green):
#   1  slot   the full timing slot (fig3-fair NxB, 3way, ablinsys, so_cost,
#             r2quote, fig4/5/7 stats; ~6-7 h; own logs in night_logs/<stamp>)
#             -> compare vs the 08-04 night (20260804_064916, CSVs @2c95b15);
#             this also prices the launch-config change + the +45-70%
#             batched-throughput regression standing from 08-04.
#   2  w36    go2 STATE_SIZE=36 pcg-vs-bdsv crossover over batch (W3.6)
#   3  wipe   contact-wipe quiet-box solve-time quote (fc + pos arms, one
#             scenario each; solve_ms in the pkls + leg log)
# ============================================================================
set -uo pipefail

REPO=/home/plancher/Desktop/GATO
PY=/home/plancher/Desktop/GRiD/.venv/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$REPO/night_logs/overnight_$STAMP
mkdir -p "$OUT"
cd "$REPO"

declare -A RC
leg() {  # leg <name> <cmd...>
    local name=$1; shift
    echo "=== LEG $name start $(date +%H:%M:%S) ===" | tee -a "$OUT/night.log"
    if "$@" > "$OUT/$name.log" 2>&1; then RC[$name]=0; else RC[$name]=$?; fi
    echo "=== LEG $name done rc=${RC[$name]} $(date +%H:%M:%S) ===" | tee -a "$OUT/night.log"
}

leg slot bash examples/benchmarks/run_gato_timing_slot.sh
leg w36  $PY examples/benchmarks/w36_go2_linsys_sweep.py --out "$OUT/w36_go2_linsys.json"
leg wipe_fc  $PY examples/contact-task/run_wipe_cell.py --arm fc  --scenarios 0 --out "$OUT/wipe_quote"
leg wipe_pos $PY examples/contact-task/run_wipe_cell.py --arm pos --scenarios 0 --out "$OUT/wipe_quote"

echo "=== NIGHT SUMMARY ===" | tee -a "$OUT/night.log"
fail=0
for k in "${!RC[@]}"; do
    echo "  $k rc=${RC[$k]}" | tee -a "$OUT/night.log"
    [[ ${RC[$k]} -ne 0 ]] && fail=1
done
exit $fail
