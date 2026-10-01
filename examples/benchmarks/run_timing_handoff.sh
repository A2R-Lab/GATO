#!/usr/bin/env bash
# Coordinator entry point. Preparation/dry-run never measures performance.
# All timings require an explicitly assigned, exclusive CPU+GPU slot.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PY="$REPO/.venv/bin/python"
SUITE=checkpoint
PREVIEW=0
while (( $# )); do
  case "$1" in
    --suite) [[ $# -ge 2 ]] || exit 2; SUITE=$2; shift 2 ;;
    --dry-run) PREVIEW=1; shift ;;
    *) echo 'usage: run_timing_handoff.sh [--suite checkpoint|seed-ab|compile|calibrate|all] [--dry-run]' >&2; exit 2 ;;
  esac
done
case "$SUITE" in checkpoint|seed-ab|compile|calibrate|all) ;; *) echo 'Unknown suite.' >&2; exit 2 ;; esac
if (( PREVIEW )); then
  echo "DRY RUN: suite=$SUITE; no GPU queries, builds, timing, locks, or result writes."
  echo 'checkpoint: three iiwa14 N64 B1/8/128 repeats, then ten B128 Fig-7 scenarios.'
  echo 'seed-ab: three repeats each of zero-tail and hold seeds; same frozen reference, alternating order; no Fig-7.'
  echo 'compile: isolated Release/sm120 indy7 N16 and iiwa14 N64; cold/no-op wall, RSS, sizes.'
  echo 'calibrate: indy7/iiwa14 N64 kicked fig8; PCG/BDSV and conditional auto validation.'
  echo 'all: checkpoint, compile, calibrate, sequentially. Default: checkpoint only.'
  echo 'Uses cooperative /tmp/a2rlab-timing.lock; requires a user-declared exclusive slot.'
  echo 'STOP file in announced output directory prevents starting the next leg; no hard timeout.'
  exit 0
fi
[[ "${GATO_QUIET_WINDOW:-}" == 1 ]] || {
  echo 'REFUSED: user-declared quiet window required (GATO_QUIET_WINDOW=1).' >&2; exit 2;
}
[[ "${FORCE:-0}" != 1 ]] || { echo 'REFUSED: FORCE bypass is forbidden.' >&2; exit 2; }
cd "$REPO"
[[ -z "$(git status --porcelain --ignore-submodules=untracked)" ]] || {
  echo 'REFUSED: commit source changes first.' >&2; exit 2;
}
[[ -x "$PY" && -f gpu-proof.json ]] || { echo 'Missing venv or receipt.' >&2; exit 2; }
for tool in flock nvidia-smi cmake nvcc; do command -v "$tool" >/dev/null || { echo "Missing tool: $tool" >&2; exit 2; }; done
if [[ "$SUITE" == compile || "$SUITE" == all ]]; then
  command -v systemd-run >/dev/null
  [[ -x /usr/bin/time ]] || { echo 'GNU time is required.' >&2; exit 2; }
fi
# Keep this inode in place. Removing it can allow two independent lock holders.
# Descendants inherit FD 9, so an unfinished child still holds the reservation.
exec 9>>/tmp/a2rlab-timing.lock
flock -n 9 || { echo 'REFUSED: another cooperating timing runner holds the box.' >&2; exit 75; }
source "$HERE/night_lib.sh"
nvidia-smi >/dev/null
night_preflight
mkdir -p "$HERE/night_logs"
LOGDIR=$(mktemp -d "$HERE/night_logs/handoff_XXXXXXXX")
SUMMARY="$LOGDIR/SUMMARY.txt"
export PYTHONUNBUFFERED=1
snapshot() {
  git rev-parse HEAD
  git submodule status --recursive
  sha256sum gpu-proof.json
  [[ ! -f build/CMakeCache.txt ]] || sha256sum build/CMakeCache.txt
  while IFS= read -r module; do sha256sum "$module"; done < <(find python/gato -type f -name '*.so' | sort)
}
snapshot > "$LOGDIR/inputs-before.txt"
{
  date -u; nvidia-smi; nvcc --version; cmake --version
  "$PY" --version; "$PY" -m pip freeze; free -g
} > "$LOGDIR/environment.txt"
echo "GATO suite=$SUITE output=$LOGDIR" | tee "$SUMMARY"
echo 'Status: RUNNING; do not infer success from partial CSVs.' >> "$SUMMARY"
finish() {
  local rc=$?
  trap - EXIT
  echo "EXIT rc=$rc at $(date -u +%FT%TZ); inspect SUMMARY and logs." >> "$SUMMARY"
  exit "$rc"
}
trap finish EXIT
check_inputs() {
  snapshot > "$LOGDIR/inputs-after.txt"
  cmp "$LOGDIR/inputs-before.txt" "$LOGDIR/inputs-after.txt"
  [[ -z "$(git status --porcelain --ignore-submodules=untracked)" ]]
}
run_leg() {
  if [[ -e "$LOGDIR/STOP" ]]; then
    echo 'DEFERRED: STOP requested; remaining legs were not run.' | tee -a "$SUMMARY"
    exit 3
  fi
  check_inputs
  nvidia-smi >/dev/null
  night_settle 60
  night_preflight
  leg "$@"
  night_settle 60
  nvidia-smi >/dev/null
  night_preflight
  check_inputs
}
if [[ "$SUITE" == checkpoint || "$SUITE" == all ]]; then
  run_leg checkpoint bash "$HERE/run_merge_checkpoint.sh"
  # The nested runner announces its unique output directory in checkpoint.log.
fi
if [[ "$SUITE" == seed-ab ]]; then
  # Isolate the benchmark migration's seed change on ONE source/binary. This
  # is not an old-vs-new CUDA implementation speedup comparison.
  goal_args=()
  for repeat in 1 2 3; do
    seeds=(zero-tail hold)
    [[ "$repeat" != 2 ]] || seeds=(hold zero-tail)
    for seed in "${seeds[@]}"; do
      csv="$LOGDIR/seed-$seed-repeat$repeat.csv"
      run_leg "seed-$seed-repeat$repeat" "$PY" "$HERE/sweep_batch_iiwa_fig8.py" \
        --N 64 --batches 1,8,128 --solves 400 --initial-guess "$seed" \
        "${goal_args[@]}" --out "$csv"
      if (( ${#goal_args[@]} == 0 )); then
        goal_file=$("$PY" -c 'import json,sys; print(json.loads(open(sys.argv[1]).readlines()[-1])["goal_file"])' "$csv.runs.jsonl")
        goal_args=(--goal-file "$goal_file")
      fi
    done
  done
fi
if [[ "$SUITE" == compile || "$SUITE" == all ]]; then
  BUILD="$LOGDIR/compile-build"
  OUT="$LOGDIR/compile-modules"
  pybind_dir=$("$PY" -m pybind11 --cmakedir)
  run_leg configure cmake -S "$REPO" -B "$BUILD" -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DPython3_EXECUTABLE="$PY" -Dpybind11_DIR="$pybind_dir" \
    -DGATO_RECEIPT_PROFILE=OFF -DGATO_BUILD_DEMO=OFF \
    -DGATO_CONTACT_FORCES=OFF -DGATO_EXACT_HESSIAN=OFF -DGATO_FAST_MATH=ON \
    -DMODULES='indy7:16;iiwa14:64' -DGATO_MODULE_OUTPUT_DIR="$OUT" \
    -DCMAKE_CUDA_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER=
  measure_build() {
    local target=$1 kind=$2 available_kib
    available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    (( available_kib >= 40 * 1024 * 1024 )) || {
      echo 'REFUSED: less than 40 GiB available for a 36 GiB capped build.' >&2; return 1;
    }
    systemd-run --user --scope -p MemoryMax=36G -p MemorySwapMax=0 --same-dir \
      /usr/bin/time -v -o "$LOGDIR/$target-$kind.time.txt" \
      cmake --build "$BUILD" --target "$target" --parallel 2
  }
  for target in bsqpN16_indy7 bsqpN64_iiwa14; do
    run_leg "$target-cold" measure_build "$target" cold
    run_leg "$target-noop" measure_build "$target" noop
  done
  find "$OUT" -maxdepth 1 -type f -name '*.so' -printf '%f %s bytes\n' > "$LOGDIR/binary-sizes.txt"
fi
if [[ "$SUITE" == calibrate || "$SUITE" == all ]]; then
  for plant in indy7 iiwa14; do
    run_leg "calibrate-$plant" "$PY" tools/autotune_linsys.py \
      --plant "$plant" --N 64 --task-tag fig8 --sim-time 6 --kick-every 25 --seed 7 \
      --tuning-path "$LOGDIR/linsys-tuning.json"
  done
fi
echo 'DONE: all selected legs completed; validate quiet continuity and numerical quality before accepting results.' | tee -a "$SUMMARY"
