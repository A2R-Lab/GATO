#!/usr/bin/env bash
# ============================================================================
# GATO throughput-regression bisect — SINGLE-FILE ENTRY POINT for a quiet slot.
#
#   bash /home/plancher/Desktop/GATO/examples/benchmarks/run_gato_bisect_slot.sh
#
# Attributes the +45-70% batched-solve regression between the 08-01 and 08-04
# timing nights (docs/open-tasks/timing_night_2026-08-04_triage.md) by timing
# the SAME sweep cell across pre-built per-commit modules:
#
#   dbb5ec3 (08-01 base) -> ea2798f (merit two-pass) -> 3585fe2 (line-search
#   fix) -> d090a63 (per-knot f_ext + GLASS pcg barriers) -> 1c609c2 (GRiD
#   regen + posture anchor) -> 67f461a (Drake + per-joint cost vectors) -> HEAD
#
# Needs a SOLO slot (~20-30 min): every leg is timing. Modules were built ahead
# of time under /home/plancher/Desktop/GATO-bisect/<sha>/ (venv python, sm_120).
# Sweep = N=16, B in {1,64,512}, 400 solves, 2 interleaved cycles per sha; plus
# nsys kernel-share profiles at the two endpoints. Exit 0 = table produced.
#
# ENV: SETTLE=secs (default 600; 0 disables)  DRYRUN=1  SOLVES=n (default 400)
# ============================================================================
set -uo pipefail

REPO=/home/plancher/Desktop/GATO
BISECT=/home/plancher/Desktop/GATO-bisect
PY=/home/plancher/Desktop/GRiD/.venv/bin/python
SHAS="dbb5ec3 ea2798f 3585fe2 d090a63 1c609c2 67f461a"
SETTLE=${SETTLE:-600}
SOLVES=${SOLVES:-400}
SUF=cpython-312-x86_64-linux-gnu.so

fatal() { echo "FATAL: $*" >&2; exit 2; }
[[ -x $PY ]] || fatal "python not found: $PY"
[[ -f $REPO/python/gato/bsqpN16_iiwa14.$SUF ]] || fatal "HEAD module missing in $REPO/python/gato"
for sha in $SHAS; do
  [[ -f $BISECT/$sha/python/gato/bsqpN16_iiwa14.$SUF ]] \
    || fatal "bisect module missing for $sha (run scratchpad/build_bisect_worktrees.sh)"
done
command -v nvidia-smi >/dev/null || fatal "nvidia-smi not on PATH"

# The sweep driver hardcodes the MAIN repo on sys.path ahead of everything —
# a worktree run would silently import the main tree's module. Repoint each
# worktree's copy at itself (idempotent; worktrees are scratch).
for sha in $SHAS; do
  sed -i "s|/home/plancher/Desktop/GATO/python|$BISECT/$sha/python|g" \
    "$BISECT/$sha/examples/benchmarks/sweep_batch_iiwa_fig8.py"
done

STAMP=$(date -u +%Y%m%d_%H%M%S)
OUT=$REPO/examples/benchmarks/night_logs/bisect_$STAMP
mkdir -p "$OUT"
echo "bisect slot $STAMP  ->  $OUT"
[[ "${DRYRUN:-0}" == "1" ]] && { echo "DRYRUN=1 — exiting."; exit 0; }

# ---- settle: this slot must be SOLO ----
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
  (( apps > 0 )) && { echo "ABORT: GPU still busy after ${SETTLE}s — bisect timing needs a solo slot." >&2; exit 3; }
fi

# ---- sweeps: 2 interleaved cycles so box drift can't masquerade as a commit ----
run_sweep() {  # $1 = label, $2 = repo dir
  "$PY" "$2/examples/benchmarks/sweep_batch_iiwa_fig8.py" \
      --N 16 --batches 1,64,512 --solves "$SOLVES" --out "$OUT/sweep_$1.csv" \
      >> "$OUT/sweep_$1.log" 2>&1 \
    || echo "WARN: sweep $1 failed (see $OUT/sweep_$1.log)"
}
for cycle in 1 2; do
  echo "---- cycle $cycle $(date -u +%H:%M:%S)"
  for sha in $SHAS; do run_sweep "$sha" "$BISECT/$sha"; done
  run_sweep HEAD "$REPO"
done

# ---- nsys kernel shares at the endpoints (where did the time go) ----
NSYS=$(command -v nsys || echo /usr/local/cuda/bin/nsys)
if [[ -x $NSYS ]]; then
  for leg in "dbb5ec3:$BISECT/dbb5ec3" "HEAD:$REPO"; do
    name=${leg%%:*}; dir=${leg#*:}
    "$NSYS" profile --stats=true -o "$OUT/nsys_$name" \
        "$PY" "$dir/examples/benchmarks/sweep_batch_iiwa_fig8.py" \
        --N 16 --batches 512 --solves 60 --out "$OUT/nsys_sweep_$name.csv" \
        > "$OUT/nsys_$name.log" 2>&1 \
      || echo "WARN: nsys $name failed"
    grep -A 25 "cuda_gpu_kern_sum" "$OUT/nsys_$name.log" | head -30 > "$OUT/nsys_$name.kern.txt" || true
  done
else
  echo "WARN: nsys not found — skipping kernel-share profiles"
fi

# ---- verdict table ----
"$PY" - "$OUT" <<'PYEOF' | tee "$OUT/VERDICT.txt"
import csv, sys, os
out = sys.argv[1]
order = ["dbb5ec3","ea2798f","3585fe2","d090a63","1c609c2","67f461a","HEAD"]
label = {"dbb5ec3":"08-01 base","ea2798f":"merit two-pass","3585fe2":"line-search fix",
         "d090a63":"per-knot f_ext + GLASS pcg","1c609c2":"regen + posture anchor",
         "67f461a":"Drake + per-joint vectors","HEAD":"38e8260+"}
data = {}
for sha in order:
    p = os.path.join(out, f"sweep_{sha}.csv")
    if not os.path.exists(p): continue
    for r in csv.DictReader(open(p)):
        key = (sha, int(r["B"]))
        data.setdefault(key, []).append(float(r["median_ms"]))
Bs = [1, 64, 512]
print(f"{'sha':>9} {'step':<28}" + "".join(f"{'B='+str(b):>10}" for b in Bs) + "   (min of cycles, ms; delta vs prev)")
prev = {}
for sha in order:
    if (sha, Bs[0]) not in data: print(f"{sha:>9} {label[sha]:<28}  MISSING"); continue
    row = f"{sha:>9} {label[sha]:<28}"
    for b in Bs:
        m = min(data[(sha, b)])
        d = f" ({(m/prev[b]-1)*100:+.0f}%)" if b in prev else ""
        row += f"{m:>10.3f}" + d
        prev[b] = m
    print(row)
PYEOF

echo; echo "done -> $OUT (VERDICT.txt + per-sha CSVs/logs + nsys kernel shares)"
exit 0
