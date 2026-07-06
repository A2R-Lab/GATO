#!/usr/bin/env bash
# Run the FULL GATO test suite and emit a SIGNED gpu-proof receipt (gpu-proof.json).
#
# This is the local half of GATO's GPU CI: the suite runs on the lab GPU box,
# signs a receipt binding {git SHA, source fingerprint, per-test outcomes, GPU
# info}, and the CPU-only GitHub Action (.github/workflows/verify-gpu-proof.yml)
# verifies the signature against github.com/plancherb1.keys on every push.
# Commit the receipt together with (or right after) the change it attests.
#
# Prerequisites:
#   - built solver modules for BOTH vendored robots (the smoke tests iterate
#     every built (plant, N) combo; test_build additionally dogfoods gato.build)
#   - a python WITH pinocchio (the [examples] extra, or GRiD's .venv) — without
#     it the solver-construction tests SKIP, and a receipt with unexpected
#     skips fails CI verification (skips prove nothing).
#   - the pytest-gpu-proof submodule (git submodule update --init) and an SSH
#     signing key (~/.ssh/id_*) whose public half is on the keyholder's GitHub.
#
# Usage:
#   ./test/run_gpu_proof.sh                      # full receipt -> gpu-proof.json
#   PYTHON=path/to/python ./test/run_gpu_proof.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Refuse to sign a dirty tree: the fingerprint cannot descend into the
# external/GRiD + external/GLASS submodules; a clean tree is what pins them via
# the receipt's commit SHA (mirrors test/gpu-proof-policy.yaml allow_dirty:false).
# Untracked content inside a submodule (sqpcpu build deps) is fine — the pin is
# what matters.
if [[ -n "$(git status --porcelain --ignore-submodules=untracked)" ]]; then
    echo "ERROR: working tree is dirty. Commit or stash before signing a receipt." >&2
    exit 1
fi

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" -m pip install -q -e test/pytest-gpu-proof

# --gpu-proof-github-user: the signer must be the human KEYHOLDER — the
# plugin's remote-derived default would guess the org (A2R-Lab), and orgs have
# no SSH keys. The rest of the config lives in pyproject [tool.gpu_proof].
"$PYTHON" -m pytest test/ -q "$@" \
    --gpu-proof-enable \
    --gpu-proof-out gpu-proof.json \
    --gpu-proof-github-user plancherb1

echo
echo "Signed receipt: gpu-proof.json — 'git add gpu-proof.json' to attest this run."
