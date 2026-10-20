#!/usr/bin/env bash
# Assigned-window launcher; correctness preparation uses controller_replay.py directly.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PY=$REPO/.venv/bin/python
if [[ ${1:-} == --dry-run && $# == 1 ]]; then
  echo 'DRY RUN: no GPU queries, writes, builds or timing.'
  echo 'Five arms, N64, B1/8, off/default ADMM, NumPy/CUDA torch/CUDA JAX.'
  echo 'Three rotated-order repeats, 20 warmup + 200 sampled calls: 180 result rows.'
  echo 'Reserve 20 minutes initially (estimate, not measured); output night_logs/replay_XXXXXXXX.'
  exit 0
fi
[[ $# == 0 ]] || { echo 'usage: bash run_controller_replay.sh [--dry-run]' >&2; exit 2; }
[[ ${GATO_QUIET_WINDOW:-} == 1 ]] || { echo 'REFUSED: assigned quiet window required.' >&2; exit 2; }
cd "$REPO"
[[ -z $(git status --porcelain --ignore-submodules=untracked) ]] || { echo 'REFUSED: clean tree required.' >&2; exit 2; }
exec 9>>/tmp/a2rlab-timing.lock
flock -n 9 || { echo 'REFUSED: another timing runner owns the cooperative lock.' >&2; exit 75; }
"$REPO/.venv/bin/gpu-proof" verify --receipt gpu-proof.json --repo . \
  --policy test/gpu-proof-policy.yaml --expected-skips test/expected_skips.txt --require-gpu
"$PY" tools/native_identity.py --verify
source "$HERE/night_lib.sh"
nvidia-smi >/dev/null
[[ ${FORCE:-0} != 1 ]] || { echo 'REFUSED: no FORCE bypass.' >&2; exit 2; }
night_preflight
mkdir -p "$HERE/night_logs"
LOGDIR=$(mktemp -d "$HERE/night_logs/replay_XXXXXXXX")
SUMMARY=$LOGDIR/SUMMARY.txt
echo "GATO controller replay output=$LOGDIR" | tee "$SUMMARY"
echo "launcher_pid=$$" | tee -a "$SUMMARY"
git rev-parse HEAD >> "$SUMMARY"
trap 'rc=$?; echo "EXIT rc=$rc" >> "$SUMMARY"' EXIT
leg correctness "$PY" -m pytest test/test_controller_replay.py test/test_arrays.py -q -p no:cacheprovider
night_quiesce 60
night_preflight
# No build or calibration changes; all results stay in this unique directory.
leg replay "$PY" "$HERE/controller_replay.py" --mode timing --out "$LOGDIR/results.jsonl"
echo 'COMPLETE: inspect parity, inventory, provenance and raw samples before publication.' | tee -a "$SUMMARY"
