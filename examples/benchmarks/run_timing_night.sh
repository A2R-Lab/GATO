#!/usr/bin/env bash
# ONE staged quiet-box TIMING night for the GATO stack (Phase 3 of the 2026-07-30
# plan). Every pre-07-30 tracking/timing number is stale (named-target EE-frame
# regen + GLASS/GRiD bump + trajfile regen); this runner regenerates the whole
# quotable set in one serial pass. Legs run SEQUENTIALLY, timing-sensitive first,
# so a cut-short window still yields the timing numbers; a leg failure is logged
# and the chain continues (nothing later depends on an earlier leg's success —
# EXCEPT leg 3way regenerating the trajfiles fig3 loads; its failure is flagged).
#
#   0. rebuild : defensive module rebuild at THIS HEAD (default build/ tree +
#                build_eh/ exact modules) — all later legs are MODULE-DEP.
#   1. fence   : correctness sanity (MPCGPU run_gates 4/4 + GATO gpu pytest).
#   2. 3way    : MPCGPU tools/run_3way_iiwa.sh — regenerates trajfiles (EE frame)
#                + 3-way tracking parity. Feeds fig3's goal inputs.
#   3. fig3    : reproduce_fig3_fair.py full (GATO NxB sweep + BT + MPCGPU).
#   4. ablinsys: ADMM inner-loop linsys A/B — bdsv-factor-reuse (default) vs
#                warm PCG (_lpcg cells), box/cone/collision ADMM families.
#   5. so_cost : SO-SQP per-iter cost — +bdsv control arm on default modules,
#                then .so-swap build_eh exact modules, +ex arm, restore.
#   6. r2quote : quiet-box solve_us re-runs of the R2/2b bound-default cells
#                (cone family on press/press_mild, collision family on pillars).
#   7. figs    : fig5 (~1 min) + fig4 (~85 min) + fig7 (~200 min) — statistics
#                regen, GPU-hours but not timing-sensitive; last in case the
#                window is cut short.
#
# Total ~6-7 h. Per-leg logs + SUMMARY.txt in examples/benchmarks/night_logs/<stamp>/.
# Timing rows land where their harnesses put them (sweep CSVs under
# examples/benchmarks/data/, constraint_eval results.jsonl, paper-figure pkls);
# quiet-box provenance = this run's SUMMARY (preflight recorded) + git SHA.
#
# Usage (repo root, quiet box):  examples/benchmarks/run_timing_night.sh
#   FORCE=1  skip the GPU-quiet preflight (smoke-testing the plumbing only)
#   QUICK=1  tiny subsets everywhere (plumbing smoke; numbers are GARBAGE)
#   JOBS=n   leg-0 rebuild parallelism (default 4 — the quiet-box cap; use 1
#            when smoke-testing while other agents are on the box)
set -uo pipefail
REPO=/home/plancher/Desktop/GATO
MPCGPU=/home/plancher/Desktop/MPCGPU
PY=/home/plancher/Desktop/GRiD/.venv/bin/python
CEV="examples/benchmarks/constraint_eval.py"
cd "$REPO" || exit 1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOGDIR=$REPO/examples/benchmarks/night_logs/$STAMP
mkdir -p "$LOGDIR"
SUMMARY=$LOGDIR/SUMMARY.txt
{ echo "timing night $STAMP (UTC)  HEAD=$(git rev-parse --short HEAD)"
  echo "preflight: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1), load=$(cut -d' ' -f1 /proc/loadavg)"
} > "$SUMMARY"

# ---- preflight: the box must be QUIET (legs 2-6 are timing runs) ----
if [[ "${FORCE:-0}" != "1" ]]; then
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  load=$(awk '{print int($1)}' /proc/loadavg)
  if (( util > 5 || apps > 0 || load > 2 )); then
    echo "ABORT: box not quiet (gpu_util=${util}% compute_apps=${apps} load=${load})." \
         "FORCE=1 overrides (plumbing smoke only)." | tee -a "$SUMMARY"
    exit 1
  fi
fi

leg() {  # leg <name> <cmd...>  — per-leg log, rc capture, orphan check after
  local name=$1; shift
  local log=$LOGDIR/$name.log t0=$SECONDS
  echo "==== [$name] $(date -u +%H:%M:%S) $*" | tee -a "$SUMMARY"
  if "$@" >"$log" 2>&1; then
    echo "[$name] PASS  ($(( (SECONDS-t0)/60 )) min)  log=$log" | tee -a "$SUMMARY"
  else
    echo "[$name] FAIL rc=$? ($(( (SECONDS-t0)/60 )) min)  log=$log  <-- review" | tee -a "$SUMMARY"
  fi
  # orphan fence: nothing of ours may outlive its leg (GPU must be EMPTY between
  # timing legs — bench-orchestration trap: pipes/timeouts can orphan children)
  local orph
  orph=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  if (( orph > 0 )); then
    echo "[$name] WARNING: $orph compute app(s) still on the GPU post-leg" | tee -a "$SUMMARY"
    nvidia-smi --query-compute-apps=pid,name --format=csv,noheader >> "$SUMMARY"
  fi
}

QUICKQ=(); [[ "${QUICK:-0}" == "1" ]] && QUICKQ=(--quick)

# ---- 0. defensive rebuild at THIS HEAD (all timing legs are MODULE-DEP) ----
JOBS=${JOBS:-4}
leg rebuild bash -c "cmake --build build --parallel $JOBS && { [[ -d build_eh ]] && cmake --build build_eh --parallel $JOBS || true; }"

# ---- 1. correctness fence (~5 min) ----
leg fence-mpcgpu bash -c "cd $MPCGPU && bash tools/run_gates.sh"
leg fence-pytest "$PY" -m pytest test -m "gpu and not slow" -q

# ---- 2. 3-way tracking parity (regenerates the EE-frame trajfiles fig3 loads) ----
leg 3way bash -c "cd $MPCGPU && bash tools/run_3way_iiwa.sh"
grep -q "\[3way\] PASS" "$SUMMARY" || echo "NOTE: 3way failed — fig3 falls back to the formula goal (verified-equal); numbers remain valid." | tee -a "$SUMMARY"

# ---- 3. fig3-fair: the N x B / 3-solver timing grid ----
leg fig3 "$PY" examples/paper-figures/reproduce_fig3_fair.py --run-gato --run-bt --run-mpcgpu "${QUICKQ[@]}"

# ---- 4. ADMM inner-loop linsys A/B (factor-reuse vs warm PCG) ----
AB_CELLS="indy7-admm-fig8,iiwa14-admm-fig8,indy7-admm_lpcg-fig8,iiwa14-admm_lpcg-fig8"
AB_CELLS+=",indy7-cone_soc_admm-press_mild,iiwa14-cone_soc_admm-press_mild"
AB_CELLS+=",indy7-cone_soc_admm_lpcg-press_mild,iiwa14-cone_soc_admm_lpcg-press_mild"
AB_CELLS+=",indy7-cc_admm-pillars,iiwa14-cc_admm-pillars"
AB_CELLS+=",indy7-cc_admm_lpcg-pillars,iiwa14-cc_admm_lpcg-pillars"
[[ "${QUICK:-0}" == "1" ]] && AB_CELLS="indy7-cc_admm-pillars,indy7-cc_admm_lpcg-pillars"
leg ablinsys "$PY" "$CEV" --run --cells "$AB_CELLS"

# ---- 5. SO-SQP per-iter cost (+bdsv control vs +ex exact, .so-swap) ----
SO_CELLS="indy7-al-reach,iiwa14-al-reach,indy7-cone_soc_al-press_mild,iiwa14-cone_soc_al-press_mild"
[[ "${QUICK:-0}" == "1" ]] && SO_CELLS="indy7-al-reach"
leg so-control "$PY" "$CEV" --run --cells "$SO_CELLS" --bdsv
GATO_PKG=$REPO/python/gato
EH_MODS=$REPO/build_eh/modules
if [[ -d "$EH_MODS" ]]; then
  SWAP_BAK=$LOGDIR/so_swap_bak && mkdir -p "$SWAP_BAK"
  swap_ok=1
  for so in "$EH_MODS"/*.so; do
    b=$(basename "$so")
    sha256sum "$GATO_PKG/$b" > "$SWAP_BAK/$b.sha" || swap_ok=0
    cp -p "$GATO_PKG/$b" "$SWAP_BAK/$b" && cp -p "$so" "$GATO_PKG/$b" || swap_ok=0
  done
  if (( swap_ok )); then
    leg so-exact "$PY" "$CEV" --run --cells "$SO_CELLS" --exact
  else
    echo "[so-exact] SKIP: .so swap failed — default modules NOT touched further" | tee -a "$SUMMARY"
  fi
  # restore + verify (the R2 discipline: sha256-checked restore, always runs)
  for so in "$EH_MODS"/*.so; do
    b=$(basename "$so"); cp -p "$SWAP_BAK/$b" "$GATO_PKG/$b"
  done
  if ( cd "$SWAP_BAK" && sha256sum -c ./*.sha --quiet ); then
    echo "[so-restore] default modules restored, sha256 VERIFIED" | tee -a "$SUMMARY"
  else
    echo "[so-restore] FAIL: restored .so sha256 MISMATCH — rebuild build/ before trusting later legs" | tee -a "$SUMMARY"
  fi
else
  echo "[so-exact] SKIP: no build_eh/modules (build with -DGATO_EXACT_HESSIAN=ON first)" | tee -a "$SUMMARY"
fi

# ---- 6. R2/2b bound-default solve_us quotes (quiet-box re-runs) ----
RQ_CELLS="indy7-baseline-fig8,iiwa14-baseline-fig8"
for pl in indy7 iiwa14; do
  for pr in press press_mild; do
    RQ_CELLS+=",${pl}-cone_off-${pr},${pl}-cone_soc_admm-${pr},${pl}-cone_soc_al-${pr},${pl}-cone_rb-${pr}"
  done
  RQ_CELLS+=",${pl}-cc_off-pillars,${pl}-cc_al-pillars,${pl}-cc_rb-pillars"
done
[[ "${QUICK:-0}" == "1" ]] && RQ_CELLS="indy7-baseline-fig8"
leg r2quote "$PY" "$CEV" --run --cells "$RQ_CELLS"

# ---- 7. long statistics tail (not timing-sensitive; last on purpose) ----
if [[ "${QUICK:-0}" != "1" ]]; then
  leg fig5 "$PY" examples/paper-figures/reproduce_fig5_disturbance.py --regen
  leg fig4 "$PY" examples/paper-figures/reproduce_fig4_hparam.py --regen
  leg fig7 "$PY" examples/paper-figures/reproduce_fig7_pickplace.py --regen
else
  echo "[figs] SKIP (QUICK=1)" | tee -a "$SUMMARY"
fi

# ---- digest ----
{
  echo; echo "---- key results ----"
  grep -E "PASS|FAIL" "$LOGDIR/fence-mpcgpu.log" 2>/dev/null | tail -6
  grep -E "RESULT" "$LOGDIR/3way.log" 2>/dev/null | head -6
  grep -E "GATOvsBT|GATOvsMPCGPU|=== Fig-3" -m 12 "$LOGDIR/fig3.log" 2>/dev/null | head -14
  # solve_us lives in results.jsonl (the --run driver logs only [ok] lines);
  # last row per (cell, exact-flag) wins — exactly the harness supersede rule
  echo "-- quiet-box solve_us_median (last row per cell+arm, this run's legs) --"
  "$PY" - <<'PYEOF'
import json
rows = {}
with open("examples/benchmarks/data/constraint_eval/results.jsonl") as f:
    for line in f:
        r = json.loads(line)
        rows[r["cell"]] = r  # arm tags (+ex/+bdsv) are part of the cell name
want = [c for c in sorted(rows) if ("lpcg" in c or "+ex" in c or "+bdsv" in c
        or "admm" in c or "cone_" in c or "cc_" in c or "baseline" in c)]
for c in want:
    r = rows[c]
    print(f'{c:48s} solve_us={r.get("solve_us_median", float("nan")):10.1f}  '
          f'track={r.get("track_mean", float("nan")):.4f}')
PYEOF
  grep -E "TABLE I|^ *Batch|^ *[0-9]+ +[0-9.]+" "$LOGDIR/fig7.log" 2>/dev/null | tail -12
} >> "$SUMMARY"
echo "DONE $(date -u +%H:%M:%S). Summary: $SUMMARY"
cat "$SUMMARY"
