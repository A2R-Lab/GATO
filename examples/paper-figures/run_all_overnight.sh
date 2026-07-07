#!/usr/bin/env bash
# One-shot OVERNIGHT regeneration of all GATO paper data on a QUIET box (2026-07-07 queue).
# Stages run SEQUENTIALLY (timing-sensitive first; a stage failure is logged and the chain
# continues — nothing later depends on an earlier stage's success):
#   1. fig3-fair : GATO NxB sweep (N in {8..128}, B up to 512) + BatchThneed B-sweep +
#                  MPCGPU per-solve (N=64), then table/heatmap assembly. TIMING — must be first.
#   2. gates     : MPCGPU tools/run_gates.sh + GBD-PCG test/run_gates.sh. First execution of
#                  the authored gate runners: REVIEW the log before committing those files.
#   3. fig5      : disturbance sweep REGEN (all pre-2026-07-07 fig5 data is invalid — the sim
#                  applied the world f_ext in the wrong frame; fixed in commit 6466349).
#   4. fig4      : hparam sweep regen.
#   5. fig7      : FULL 100-scenario pick-place sweep (~7 h — the long pole, so it goes last;
#                  everything before it completes in the first ~2 h).
# Total ~9-10 h. Per-stage logs + SUMMARY.txt in examples/paper-figures/overnight_logs/<stamp>/.
#
# Usage (from the GATO repo root):  examples/paper-figures/run_all_overnight.sh
#   FORCE=1 skips the GPU-quiet preflight check.
set -uo pipefail
REPO=/home/plancher/Desktop/GATO
MPCGPU=/home/plancher/Desktop/MPCGPU
PY=/home/plancher/Desktop/GRiD/.venv/bin/python
cd "$REPO" || exit 1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOGDIR=$REPO/examples/paper-figures/overnight_logs/$STAMP
mkdir -p "$LOGDIR"
SUMMARY=$LOGDIR/SUMMARY.txt
echo "overnight run $STAMP (UTC)" > "$SUMMARY"

# ---- preflight: the box must be quiet (stage 1 is a timing run) ----
if [[ "${FORCE:-0}" != "1" ]]; then
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  load=$(awk '{print int($1)}' /proc/loadavg)
  if (( util > 5 || apps > 0 || load > 2 )); then
    echo "ABORT: box not quiet (gpu_util=${util}% compute_apps=${apps} load=${load})." \
         "Re-run with FORCE=1 to override." | tee -a "$SUMMARY"
    exit 1
  fi
fi

stage() {  # stage <name> <log> <cmd...>
  local name=$1 log=$2; shift 2
  local t0=$SECONDS
  echo "==== [$name] $(date -u +%H:%M:%S) $*" | tee -a "$SUMMARY"
  if "$@" >"$log" 2>&1; then
    echo "[$name] PASS  ($(( (SECONDS-t0)/60 )) min)  log=$log" | tee -a "$SUMMARY"
  else
    echo "[$name] FAIL  ($(( (SECONDS-t0)/60 )) min)  log=$log  <-- review" | tee -a "$SUMMARY"
  fi
}

# 1. fig3 fair data (timing: GATO grid, BT, MPCGPU — the orchestrator runs them sequentially)
stage fig3 "$LOGDIR/fig3.log" \
  "$PY" examples/paper-figures/reproduce_fig3_fair.py --run-gato --run-bt --run-mpcgpu

# 2. correctness gate runners (authored 2026-07-07, first execution — review before committing)
stage gates-mpcgpu "$LOGDIR/gates_mpcgpu.log" bash -c "cd $MPCGPU && bash tools/run_gates.sh"
stage gates-gbdpcg "$LOGDIR/gates_gbdpcg.log" bash -c "cd $MPCGPU/GBD-PCG && bash test/run_gates.sh"

# 3-5. figure regens (correctness/statistics, not timing)
stage fig5 "$LOGDIR/fig5.log" "$PY" examples/paper-figures/reproduce_fig5_disturbance.py --regen
stage fig4 "$LOGDIR/fig4.log" "$PY" examples/paper-figures/reproduce_fig4_hparam.py --regen
stage fig7 "$LOGDIR/fig7.log" "$PY" examples/paper-figures/reproduce_fig7_pickplace.py --regen

# ---- digest: pull the headline lines into the summary ----
{
  echo; echo "---- key results ----"
  grep -E "GATOvsBT|GATOvsMPCGPU|=== Fig-3" -m 12 "$LOGDIR/fig3.log" 2>/dev/null | head -14
  grep -E "PASS|FAIL" "$LOGDIR/gates_mpcgpu.log" 2>/dev/null | tail -8
  grep -E "PASS|FAIL" "$LOGDIR/gates_gbdpcg.log" 2>/dev/null | tail -6
  grep -E "TABLE I|^ *Batch|^ *[0-9]+ +[0-9.]+" "$LOGDIR/fig7.log" 2>/dev/null | tail -12
} >> "$SUMMARY"
echo "DONE $(date -u +%H:%M:%S). Summary: $SUMMARY"
cat "$SUMMARY"
