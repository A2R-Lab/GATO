#!/usr/bin/env bash
# The W2a / H1 timing legs for a SOLO GPU slot — one line for a batch queue
# (another agent can launch it; it waits for the GPU to clear, then runs):
#
#   examples/benchmarks/run_timing_w2a.sh
#
# Legs run SEQUENTIALLY; a leg failure is logged and the chain continues; exit
# 0 only if every leg passed. The October 19 RTX 5090 run took about 3 min
# including fences; reserve 10 min for the same batch, excluding queue wait.
#
#   0. fence    : the built modules match HEAD's native sources + a GPU smoke
#                 (pytest test_receipt_contract.py + "gpu and not slow"; ~3 min).
#   1. matrix   : ADMM inner-loop time matrix, {indy7, iiwa14} x {N32, N64} x
#                 {quiet, kicked} x {free, fixed10-pcg, fixed10-fac, exit-pcg,
#                 exit-fac, exit-auto} (admm_time_matrix.py --leg matrix).
#   2. fit      : tools/autotune_linsys.py --admm per (plant, N) — fits and
#                 saves proposals in the isolated run directory, not the installed table.
#   3. deadline : H1's controller-call-boundary table, step() wall with and
#                 without a budget, indy7 + iiwa14 N64.
#
# Per-leg logs + SUMMARY.txt in examples/benchmarks/night_logs/<UTC stamp>-w2a/;
# result rows in that directory's results.jsonl
# (git SHA, GPU and quiet-box verdict in every row).
#
# ENV (all optional)
#   PY=path      python (default: the project .venv)
#   SETTLE=secs  wait up to this long for the GPU to clear first (default 7200:
#                queue this after another agent's timing and it waits its turn)
#   FORCE=1      skip the quiet-box preflight (plumbing smoke only)
#   QUICK=1      tiny subsets, numbers are GARBAGE (plumbing smoke)
#   DRYRUN=1     print the plan and exit without touching the GPU
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PY=${PY:-$REPO/.venv/bin/python}
SETTLE=${SETTLE:-7200}
# shellcheck source=night_lib.sh
source "$HERE/night_lib.sh"

fatal() { echo "FATAL: $*" >&2; exit 2; }
[[ -x $PY ]] || fatal "python not found: $PY (./tools/install.sh --test; or PY=...)"
command -v nvidia-smi >/dev/null || fatal "nvidia-smi not on PATH"
cd "$REPO" || fatal "cannot cd $REPO"
"$PY" -c "import gato, pinocchio" 2>/dev/null || fatal "$PY cannot import gato + pinocchio"
for m in bsqpN32_indy7 bsqpN64_indy7 bsqpN32_iiwa14 bsqpN64_iiwa14; do
  ls "$REPO"/python/gato/"$m".*.so >/dev/null 2>&1 || fatal "module $m not built (./tools/build.sh --profile receipt)"
done

HEAD_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
DIRTY=$(git status --porcelain --untracked-files=no --ignore-submodules=untracked 2>/dev/null | grep -c . || true)
cat <<EOT
=========================================================================
GATO W2a timing (ADMM inner loop + deadline step)
  repo      $REPO @ $HEAD_SHA$([[ ${DIRTY:-0} -gt 0 ]] && echo "  (tree has $DIRTY modified paths)")
  python    $PY
  settle    ${SETTLE}s      quick  ${QUICK:-0}      force  ${FORCE:-0}
  logs      $REPO/examples/benchmarks/night_logs/<UTC stamp>-w2a/
  results   per-run results.jsonl beside the logs (installed tuning is unchanged)
=========================================================================
EOT
if [[ "${DRYRUN:-0}" == "1" ]]; then echo "DRYRUN=1 — exiting before any GPU work."; exit 0; fi
if (( ${DIRTY:-0} > 0 )) && [[ "${QUICK:-0}" != "1" && "${FORCE:-0}" != "1" ]]; then
  fatal "commit tracked changes before timing"
fi

night_settle "$SETTLE" || exit 3

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOGDIR=$REPO/examples/benchmarks/night_logs/$STAMP-w2a
[[ ! -e $LOGDIR ]] || fatal "run directory already exists: $LOGDIR"
mkdir -p "$LOGDIR"
SUMMARY=$LOGDIR/SUMMARY.txt
# Copy calibration input for reproducibility, then fit only into the proposal.
BASE_TUNING=${GATO_LINSYS_TUNING:-$REPO/python/gato/linsys_tuning.json}
if [[ -f $BASE_TUNING ]]; then
  cp -- "$BASE_TUNING" "$LOGDIR/tuning-before.json"
  cp -- "$BASE_TUNING" "$LOGDIR/tuning-proposed.json"
fi
export GATO_LINSYS_TUNING=$LOGDIR/tuning-proposed.json
{ echo "W2a timing $STAMP (UTC)  HEAD=$HEAD_SHA  dirty_paths=${DIRTY:-0}"
  echo "preflight: $(nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader | head -1), load=$(cut -d' ' -f1 /proc/loadavg)"
} > "$SUMMARY"
night_preflight || exit 3

# Every run is isolated; appending to tracked results would dirty later rows.
QUICKQ=(--out "$LOGDIR/results.jsonl")
[[ "${QUICK:-0}" == "1" ]] && QUICKQ+=(--quick --allow-busy)
[[ "${FORCE:-0}" == "1" && "${QUICK:-0}" != "1" ]] && QUICKQ+=(--allow-busy)
DRIVER="$REPO/examples/benchmarks/admm_time_matrix.py"

leg fence-identity "$PY" -m pytest test/test_receipt_contract.py -q -p no:cacheprovider || exit 1
leg fence-gpu      "$PY" -m pytest test -m "gpu and not slow" -q -p no:cacheprovider || exit 1
leg matrix   "$PY" "$DRIVER" --leg matrix   "${QUICKQ[@]}"
leg fit      "$PY" "$DRIVER" --leg fit      "${QUICKQ[@]}"
leg deadline "$PY" "$DRIVER" --leg deadline "${QUICKQ[@]}"

echo "---- $(date -u +%H:%M:%S) done ----" | tee -a "$SUMMARY"
if grep -q "FAIL" "$SUMMARY"; then echo "SOME LEGS FAILED — see $SUMMARY"; exit 1; fi
echo "ALL LEGS PASSED — $SUMMARY"
