#!/usr/bin/env bash
# THE staged quiet-box TIMING night for the GATO stack — the single entry
# point for a solo GPU slot (drop this one line into a batch queue):
#
#   examples/benchmarks/run_timing_night.sh
#
# Legs run SEQUENTIALLY, timing-sensitive first, so a cut-short window still
# yields the timing numbers; a leg failure is logged and the chain continues
# (nothing later depends on an earlier leg's success — EXCEPT leg 3way
# regenerating the trajfiles fig3 loads; its failure is flagged). Exit 0 only
# if every leg passed, so a queue can tell success from partial.
#
#   0. rebuild : defensive module rebuild at THIS HEAD in build/ (all later
#                legs are MODULE-DEP; the receipt profile carries the fc/eh
#                variant modules — ./tools/build.sh --profile receipt).
#   1. fence   : correctness sanity (MPCGPU run_gates 4/4 + GATO gpu pytest).
#   2. 3way    : MPCGPU tools/run_3way_iiwa.sh — regenerates trajfiles (EE frame)
#                + 3-way tracking parity. Feeds fig3's goal inputs.
#   3. fig3    : reproduce_fig3_fair.py full (GATO NxB sweep + BT + MPCGPU).
#   4. ablinsys: ADMM inner-loop linsys A/B — warm PCG (default since
#                2026-08-01) vs bdsv-factor-reuse (_lbdsv cells),
#                box/cone/collision ADMM families.
#   5. so_cost : SO-SQP per-iter cost — +bdsv control arm on default modules,
#                then the +ex arm on the eh variant modules.
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
# ENV (all optional)
#   PY=path        python to run with (default: the project .venv — the ONLY
#                  python for this repo; tools/install.sh --test gives it pinocchio)
#   MPCGPU=path    sibling MPCGPU checkout (default <repo>/../MPCGPU; legs 1-3)
#   SETTLE=secs    wait up to this long for the GPU to go quiet before starting
#                  (queues where the previous job lingers; default 600, 0 = off)
#   FORCE=1        skip the GPU-quiet preflight (plumbing smoke only)
#   QUICK=1        tiny subsets everywhere (plumbing smoke; numbers are GARBAGE)
#   JOBS=n         leg-0 rebuild parallelism (default 4 — the RAM cap: each
#                  module compile peaks ~6-7 GB; use 1 when others are on the box)
#   SKIP_FIGS=1    run the timing legs only, skip the fig5/fig4/fig7 tail
#   DRYRUN=1       print the plan and exit without touching the GPU
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PY=${PY:-$REPO/.venv/bin/python}
MPCGPU=${MPCGPU:-$(cd "$REPO/.." && pwd)/MPCGPU}
CEV="examples/benchmarks/constraint_eval.py"
JOBS=${JOBS:-4}
SETTLE=${SETTLE:-600}
# shellcheck source=night_lib.sh
source "$HERE/night_lib.sh"

# ---- preflight: fail FAST and LOUD on anything that would waste the slot ----
fatal() { echo "FATAL: $*" >&2; exit 2; }
[[ -x $PY ]]          || fatal "python not found: $PY (./tools/install.sh --test; or PY=...)"
[[ -d $MPCGPU ]]      || fatal "MPCGPU not found at $MPCGPU (legs 1-3 need it; MPCGPU=... overrides)"
[[ -f $REPO/build/CMakeCache.txt ]] || fatal "no configured build tree at $REPO/build (./tools/build.sh --profile receipt)"
command -v nvidia-smi >/dev/null    || fatal "nvidia-smi not on PATH"
command -v cmake      >/dev/null    || fatal "cmake not on PATH"
cd "$REPO" || fatal "cannot cd $REPO"
"$PY" -c "import gato, pinocchio" 2>/dev/null || fatal "$PY cannot import gato + pinocchio"
# the eh/fc variant modules are separate side-by-side ABIs: the build tree must
# be the receipt profile for legs 5 (so-exact) to have anything to run
grep -q '^GATO_RECEIPT_PROFILE:BOOL=ON' build/CMakeCache.txt \
  || echo "WARNING: build/ is not the receipt profile (./tools/build.sh --profile receipt) — variant legs may SKIP"

HEAD_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
DIRTY=$(git status --porcelain --ignore-submodules=untracked 2>/dev/null | grep -c . || true)
cat <<EOF
=========================================================================
GATO timing night
  repo      $REPO @ $HEAD_SHA$([[ ${DIRTY:-0} -gt 0 ]] && echo "  (tree has $DIRTY modified paths)")
  python    $PY
  mpcgpu    $MPCGPU
  jobs      $JOBS      settle  ${SETTLE}s      quick  ${QUICK:-0}      skip_figs  ${SKIP_FIGS:-0}
  logs      $REPO/examples/benchmarks/night_logs/<UTC stamp>/
=========================================================================
EOF
if [[ "${DRYRUN:-0}" == "1" ]]; then echo "DRYRUN=1 — exiting before any GPU work."; exit 0; fi

# ---- settle (queue stragglers), then the hard quiet-box preflight ----
night_settle "$SETTLE" || exit 3

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOGDIR=$REPO/examples/benchmarks/night_logs/$STAMP
mkdir -p "$LOGDIR"
SUMMARY=$LOGDIR/SUMMARY.txt
{ echo "timing night $STAMP (UTC)  HEAD=$HEAD_SHA"
  echo "preflight: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1), load=$(cut -d' ' -f1 /proc/loadavg)"
} > "$SUMMARY"
night_preflight || exit 3

QUICKQ=(); [[ "${QUICK:-0}" == "1" ]] && QUICKQ=(--quick)

# ---- 0. defensive rebuild at THIS HEAD (all timing legs are MODULE-DEP) ----
leg rebuild bash -c "cmake --build build --parallel $JOBS"

# ---- 1. correctness fence (~5 min) ----
leg fence-mpcgpu bash -c "cd $MPCGPU && bash tools/run_gates.sh"
leg fence-pytest "$PY" -m pytest test -m "gpu and not slow" -q

# ---- 2. 3-way tracking parity (regenerates the EE-frame trajfiles fig3 loads) ----
leg 3way bash -c "cd $MPCGPU && bash tools/run_3way_iiwa.sh"
grep -q "\[3way\] PASS" "$SUMMARY" || echo "NOTE: 3way failed — fig3 falls back to the formula goal (verified-equal); numbers remain valid." | tee -a "$SUMMARY"

# ---- 3. fig3-fair: the N x B / 3-solver timing grid ----
leg fig3 "$PY" examples/paper-figures/reproduce_fig3_fair.py --run-gato --run-bt --run-mpcgpu "${QUICKQ[@]}"

# ---- 4. ADMM inner-loop linsys A/B (factor-reuse vs warm PCG) ----
AB_CELLS="indy7-admm-fig8,iiwa14-admm-fig8,indy7-admm_lbdsv-fig8,iiwa14-admm_lbdsv-fig8"
AB_CELLS+=",indy7-cone_soc_admm-press_mild,iiwa14-cone_soc_admm-press_mild"
AB_CELLS+=",indy7-cone_soc_admm_lbdsv-press_mild,iiwa14-cone_soc_admm_lbdsv-press_mild"
AB_CELLS+=",indy7-cc_admm-pillars,iiwa14-cc_admm-pillars"
AB_CELLS+=",indy7-cc_admm_lbdsv-pillars,iiwa14-cc_admm_lbdsv-pillars"
[[ "${QUICK:-0}" == "1" ]] && AB_CELLS="indy7-cc_admm-pillars,indy7-cc_admm_lbdsv-pillars"
leg ablinsys "$PY" "$CEV" --run --cells "$AB_CELLS"

# ---- 5. SO-SQP per-iter cost (+bdsv control vs +ex exact, eh variant modules) ----
SO_CELLS="indy7-al-reach,iiwa14-al-reach,indy7-cone_soc_al-press_mild,iiwa14-cone_soc_al-press_mild"
[[ "${QUICK:-0}" == "1" ]] && SO_CELLS="indy7-al-reach"
leg so-control "$PY" "$CEV" --run --cells "$SO_CELLS" --bdsv
# exact Hessian = the eh module VARIANT (bsqpN*_{plant}_eh.so, side by side — no .so swap)
if compgen -G "$REPO/python/gato/bsqpN*_indy7_eh.*.so" > /dev/null; then
  leg so-exact "$PY" "$CEV" --run --cells "$SO_CELLS" --exact
else
  echo "[so-exact] SKIP: no eh variant modules (./tools/build.sh --profile receipt)" | tee -a "$SUMMARY"
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
# SKIP_FIGS=1 drops it for a time-boxed slot: every TIMING number is already in
# by this point, so this is the safe thing to cut when the window is short.
if [[ "${QUICK:-0}" != "1" && "${SKIP_FIGS:-0}" != "1" ]]; then
  leg fig5 "$PY" examples/paper-figures/reproduce_fig5_disturbance.py --regen
  leg fig4 "$PY" examples/paper-figures/reproduce_fig4_hparam.py --regen
  leg fig7 "$PY" examples/paper-figures/reproduce_fig7_pickplace.py --regen
else
  echo "[figs] SKIP (QUICK=${QUICK:-0} SKIP_FIGS=${SKIP_FIGS:-0})" | tee -a "$SUMMARY"
fi

# ---- digest ----
{
  echo; echo "---- key results ----"
  grep -E "PASS|FAIL" "$LOGDIR/fence-mpcgpu.log" 2>/dev/null | tail -6
  grep -E "RESULT" "$LOGDIR/3way.log" 2>/dev/null | head -6
  # the tables sit at the end of the log; grep for column names only ever hit the
  # header line (rows are numbers), so take the range and drop progress lines
  sed -n '/=== Fig-3 (left/,$p' "$LOGDIR/fig3.log" 2>/dev/null \
    | grep -vE '^\[(fig3-fair|paper-figures)\]' | head -32
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
want = [c for c in sorted(rows) if ("lbdsv" in c or "lpcg" in c or "+ex" in c or "+bdsv" in c
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

# ---- exit code for the queue: a failed leg is logged and the chain continues
# by design, so surface it here ----
fails=$(grep -c '^\[.*\] FAIL'  "$SUMMARY" || true)
passes=$(grep -c '^\[.*\] PASS' "$SUMMARY" || true)
echo "legs passed: ${passes:-0}   failed: ${fails:-0}"
if (( ${passes:-0} == 0 )); then echo "EXIT 1: no leg reported PASS"; exit 1; fi
if (( ${fails:-0} > 0 )); then echo "EXIT 1: review the per-leg logs in $LOGDIR/"; exit 1; fi
echo "all legs PASS"
exit 0
