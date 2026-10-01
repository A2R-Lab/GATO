#!/usr/bin/env bash
# Focused, GATO-only checkpoint. No builds, tuning, baseline-repo mutations or
# full figure refresh. Preview is always safe; execution needs a declared slot.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PY="$REPO/.venv/bin/python"

case "${1:-}" in
  --dry-run)
    [[ $# == 1 ]] || exit 2
    echo 'DRY RUN: no GPU queries, builds, timing, or result writes.'
    echo '1. Check clean source, explicit quiet-window declaration, and idle GPU/CPU.'
    echo '2. Record source/submodule SHAs, binary hashes, receipt, hardware/toolchain.'
    echo '3. Fig-3: iiwa14 N64; B=1,8,128; 400 solves; three process repeats, new CSVs.'
    echo '4. Fig-7: B128; 10 scenarios; seed 0; unique output tag; regression sample only.'
    echo '5. Record ending source/binary hashes; reject changed inputs or contamination.'
    echo 'Not included: full figures, CPU/MPCGPU baselines, autotuning, compile timing.'
    exit 0 ;;
  '') ;;
  *) echo 'usage: run_merge_checkpoint.sh [--dry-run]' >&2; exit 2 ;;
esac

[[ "${GATO_QUIET_WINDOW:-}" == 1 ]] || {
  echo 'REFUSED: a user-declared quiet window is required (GATO_QUIET_WINDOW=1).' >&2
  exit 2
}
[[ "${FORCE:-0}" != 1 ]] || { echo 'REFUSED: FORCE bypass is not permitted.' >&2; exit 2; }
[[ -x "$PY" ]] || { echo 'Missing project .venv.' >&2; exit 2; }
cd "$REPO"
[[ -z "$(git status --porcelain --ignore-submodules=untracked)" ]] || {
  echo 'REFUSED: commit source changes before collecting results.' >&2; exit 2;
}
[[ -f gpu-proof.json ]] || { echo 'Missing correctness receipt.' >&2; exit 2; }
"$PY" - <<'PY'
import importlib.util
import gato
missing = [name for name in ('numpy', 'pinocchio', 'matplotlib')
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit('Missing checkpoint dependencies: ' + ', '.join(missing)
                     + '; install the examples extra before reserving the GPU.')
required = {('iiwa14', 16), ('iiwa14', 64)}
if not required <= set(gato.available()):
    raise SystemExit('Build iiwa14 N16 and N64 before reserving the GPU.')
PY
command -v nvidia-smi >/dev/null
nvidia-smi >/dev/null  # fail closed if the device/driver query fails
# shellcheck source=night_lib.sh
source "$HERE/night_lib.sh"
night_preflight

mkdir -p "$HERE/night_logs"
LOGDIR=$(mktemp -d "$HERE/night_logs/merge_checkpoint_XXXXXXXX")
SUMMARY="$LOGDIR/SUMMARY.txt"
TAG=$(basename "$LOGDIR")
snapshot() {
  git rev-parse HEAD
  git submodule status --recursive
  sha256sum gpu-proof.json
  while IFS= read -r module; do sha256sum "$module"; done < <(find python/gato -type f -name '*.so' | sort)
}
snapshot > "$LOGDIR/inputs-before.txt"
{
  date -u
  nvidia-smi
  nvcc --version
  cmake --version
  "$PY" --version
  "$PY" -m pip freeze
  [[ ! -f build/CMakeCache.txt ]] || grep -E 'CMAKE_(BUILD_TYPE|CUDA_ARCHITECTURES)|GATO_|MODULES:|PLANT:|KNOTS:' build/CMakeCache.txt
} > "$LOGDIR/environment.txt"
printf 'GATO merge checkpoint; output=%s; tag=%s\n' "$LOGDIR" "$TAG" | tee "$SUMMARY"

checkpoint_leg() {
  nvidia-smi >/dev/null
  night_settle 60  # let the previous leg's utilization sample settle
  night_preflight
  leg "$@"
  # Do not stop other agents. If another process appears, stop OUR chain.
  night_settle 60
  night_preflight
}
for repeat in 1 2 3; do
  checkpoint_leg "fig3-repeat$repeat" "$PY" examples/benchmarks/sweep_batch_iiwa_fig8.py \
    --N 64 --batches 1,8,128 --solves 400 --out "$LOGDIR/fig3-repeat$repeat.csv"
done
checkpoint_leg fig7-sample "$PY" examples/paper-figures/reproduce_fig7_pickplace.py \
  --batch-sizes 128 --n-scenarios 10 --seed 0 --tag "$TAG"
snapshot > "$LOGDIR/inputs-after.txt"
cmp "$LOGDIR/inputs-before.txt" "$LOGDIR/inputs-after.txt"
[[ -z "$(git status --porcelain --ignore-submodules=untracked)" ]] || {
  echo 'INVALID: tracked source changed during the batch.' | tee -a "$SUMMARY"; exit 1;
}
echo 'DONE: review provenance, quiet-window continuity, and numerical quality before accepting timings.' | tee -a "$SUMMARY"
