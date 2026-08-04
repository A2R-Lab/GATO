#!/usr/bin/env bash
# ============================================================================
# GATO timing slot — SINGLE-FILE ENTRY POINT for a batched multi-agent night.
#
#   bash /home/plancher/Desktop/GATO/examples/benchmarks/run_gato_timing_slot.sh
#
# Drop this one line into the batch queue. It needs a SOLO slot: legs 2-6 are
# timing runs and any other GPU work on the box corrupts them. It runs to
# completion unattended, writes everything under night_logs/<stamp>/, and exits
# 0 only if every leg passed (so the queue can tell success from partial).
#
# Runtime ~6-7 h (timing legs first, ~1.5 h; the fig4/fig7 statistics tail is
# the rest and is NOT timing-sensitive, so a cut-short slot still yields every
# timing number). No input, no prompts, no network.
#
# ENV OVERRIDES (all optional)
#   JOBS=n        rebuild parallelism for leg 0 (default 4 = the RAM cap:
#                 each module compile peaks ~6-7 GB)
#   SETTLE=secs   wait up to this long for the GPU to go quiet before starting,
#                 for queues where the previous job's process lingers
#                 (default 600; SETTLE=0 disables the wait entirely)
#   SKIP_TAIL=1   stop after the timing legs, skip fig5/fig4/fig7 (~1.5 h total)
#   DRYRUN=1      print the plan and exit without touching the GPU
#
# WHAT IT DOES
#   0  rebuild   defensive module rebuild at HEAD (all legs are MODULE-DEP)
#   1  fence     correctness sanity: MPCGPU gates 4/4 + GATO gpu pytest
#   2  3way      MPCGPU 3-way tracking parity; regenerates fig3's trajfiles
#   3  fig3      fig3-fair N x B grid, GATO + BatchThneed + MPCGPU
#   4  ablinsys  ADMM inner linsys A/B: warm-PCG (default) vs bdsv factor-reuse
#   5  so_cost   SO-SQP per-iter cost, +bdsv control vs +ex exact (.so swap,
#                sha256-verified restore)
#   6  r2quote   quiet-box solve_us for the R2/2b bound-default cells
#   7  figs      fig5 + fig4 + fig7 statistics regen (not timing-sensitive)
#
# WHY THIS RUN EXISTS: the last full timing night was 2026-08-01 at dbb5ec3.
# Since then the device side changed materially — deterministic merit (two-pass
# reduction replacing atomicAdd, a per-solve cost on EVERY solve), admm_linsys
# default flipped to pcg, GRiD bumped to b91964a with a vendored regen, and the
# iiwa14 Drake inertial swap. Those numbers are stale.
# ============================================================================
set -uo pipefail

REPO=/home/plancher/Desktop/GATO
RUNNER=$REPO/examples/benchmarks/run_timing_night.sh
MPCGPU=/home/plancher/Desktop/MPCGPU
PY=/home/plancher/Desktop/GRiD/.venv/bin/python

JOBS=${JOBS:-4}
SETTLE=${SETTLE:-600}

# ---- preflight: fail FAST and LOUD on anything that would waste the slot ----
fatal() { echo "FATAL: $*" >&2; exit 2; }

[[ -d $REPO ]]        || fatal "GATO repo not found at $REPO"
[[ -x $RUNNER ]]      || fatal "night runner missing/not executable: $RUNNER"
[[ -x $PY ]]          || fatal "python not found: $PY (the GRiD venv supplies pinocchio)"
[[ -d $MPCGPU ]]      || fatal "MPCGPU not found at $MPCGPU (legs 1-3 need it)"
[[ -f $REPO/build/CMakeCache.txt ]] || fatal "no configured build tree at $REPO/build"
command -v nvidia-smi >/dev/null    || fatal "nvidia-smi not on PATH"
command -v cmake      >/dev/null    || fatal "cmake not on PATH"

cd "$REPO" || fatal "cannot cd $REPO"

# The build cache must be the STANDARD configure. test_build.py dogfoods
# gato.build and leaves PLANT/KNOTS pointing at a throwaway plant; leg 0 then
# tries to build a target whose generated headers are gone and the whole slot
# dies at minute one. This exact state was found on 2026-08-03.
cache_plant=$(sed -n 's/^PLANT:STRING=//p'  build/CMakeCache.txt)
cache_knots=$(sed -n 's/^KNOTS:STRING=//p'  build/CMakeCache.txt)
cache_type=$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' build/CMakeCache.txt)
if [[ "$cache_plant" != "indy7;iiwa14" || "$cache_knots" != "8;16;32;64;128" \
      || "$cache_type" != "Release" ]]; then
  echo "WARNING: non-standard cmake configure (PLANT='$cache_plant'" \
       "KNOTS='$cache_knots' TYPE='$cache_type') — restoring the documented one."
  cmake -S . -B build -DPLANT="indy7;iiwa14" -DKNOTS="8;16;32;64;128" \
        -DCMAKE_BUILD_TYPE=Release >/dev/null \
    || fatal "cmake reconfigure failed"
fi
# fc/exact flags must be OFF: they change the module ABI, so timing the default
# solver against them is meaningless.
grep -q '^GATO_CONTACT_FORCES:BOOL=OFF' build/CMakeCache.txt \
  || fatal "GATO_CONTACT_FORCES is ON in build/ — timing needs the default ABI"
grep -q '^GATO_EXACT_HESSIAN:BOOL=OFF'  build/CMakeCache.txt \
  || fatal "GATO_EXACT_HESSIAN is ON in build/ — exact modules belong in build_eh/"

$PY -c "import sys; sys.path.insert(0,'python'); import gato, pinocchio" 2>/dev/null \
  || fatal "python cannot import gato + pinocchio"

HEAD_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
DIRTY=$(git status --porcelain 2>/dev/null | grep -vc 'baselines/sqpcpu' || true)

cat <<EOF
=========================================================================
GATO timing slot
  repo      $REPO @ $HEAD_SHA$([[ ${DIRTY:-0} -gt 0 ]] && echo "  (tree has $DIRTY modified paths)")
  runner    $RUNNER
  python    $PY
  jobs      $JOBS      settle  ${SETTLE}s      skip_tail  ${SKIP_TAIL:-0}
  logs      $REPO/examples/benchmarks/night_logs/<UTC stamp>/
=========================================================================
EOF

if [[ "${DRYRUN:-0}" == "1" ]]; then echo "DRYRUN=1 — exiting before any GPU work."; exit 0; fi

# ---- settle: let a previous queue entry's stragglers clear ----
# Not a substitute for a solo slot; it only absorbs the seconds between jobs.
if (( SETTLE > 0 )); then
  waited=0
  while (( waited < SETTLE )); do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    (( apps == 0 && util <= 5 )) && break
    (( waited == 0 )) && echo "waiting for the GPU to clear (apps=$apps util=${util}%)..."
    sleep 20; waited=$((waited + 20))
  done
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  if (( apps > 0 )); then
    echo "ABORT: GPU still busy after ${SETTLE}s — this slot must be SOLO or the" \
         "timing numbers are worthless. Nothing was run." >&2
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv >&2
    exit 3
  fi
  (( waited > 0 )) && echo "GPU clear after ${waited}s."
fi

# ---- run ----
# The runner does its own quiet preflight, per-leg logs, rc capture, orphan
# fence between legs, and the sha256-verified .so swap/restore for leg 5.
export JOBS
t0=$SECONDS
if [[ "${SKIP_TAIL:-0}" == "1" ]]; then
  SKIP_FIGS=1 bash "$RUNNER"
else
  bash "$RUNNER"
fi
runner_rc=$?
mins=$(( (SECONDS - t0) / 60 ))

LOGDIR=$(ls -dt "$REPO"/examples/benchmarks/night_logs/*/ 2>/dev/null | head -1)
SUMMARY="${LOGDIR%/}/SUMMARY.txt"

echo
echo "========================= SLOT COMPLETE ($mins min) ========================="
if [[ ! -f "$SUMMARY" ]]; then
  echo "no SUMMARY found — the runner did not start cleanly (rc=$runner_rc)"
  exit "${runner_rc:-1}"
fi

echo "SUMMARY: $SUMMARY"; echo
grep -E '^\[.*\] (PASS|FAIL)' "$SUMMARY" || true
fails=$(grep -c '^\[.*\] FAIL'  "$SUMMARY" || true)
passes=$(grep -c '^\[.*\] PASS' "$SUMMARY" || true)
warns=$(grep -c 'WARNING'        "$SUMMARY" || true)
echo
echo "legs passed: ${passes:-0}   failed: ${fails:-0}   warnings: ${warns:-0}"

# The runner writes its SUMMARY header BEFORE its own quiet preflight, so an
# aborted run leaves a SUMMARY with zero PASS/FAIL lines. Counting only
# failures would call that a clean sweep and exit 0 on a run that never
# happened — check for the abort and for nothing-ran explicitly.
if grep -q '^ABORT:' "$SUMMARY"; then
  echo "EXIT 3: the runner refused to start —"; grep '^ABORT:' "$SUMMARY"
  exit 3
fi
if (( ${passes:-0} == 0 )); then
  echo "EXIT 1: no leg reported PASS — the run did not execute (rc=$runner_rc)"
  exit 1
fi
# A failed leg is logged and the chain continues by design, so the runner's own
# rc does not reflect it — surface it here for the queue.
if (( ${fails:-0} > 0 )); then
  echo "EXIT 1: review the per-leg logs in ${LOGDIR%/}/"
  exit 1
fi
echo "all legs PASS"
exit 0
