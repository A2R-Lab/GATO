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
#   - the receipt module set built into python/gato/ (test/receipt_modules.txt;
#     `./tools/build.sh --profile receipt`) — the script REFUSES to sign when
#     one is missing (its tests would skip; skips prove nothing)
#   - a python WITH pinocchio + mujoco + scipy (the [test] extra:
#     `./tools/install.sh --test`, or `pip install -e ".[test,dev]"`) — the
#     script REFUSES to run without them: missing deps turn tests into skips,
#     and a receipt with unexpected skips fails CI verification (skips prove
#     nothing). The project .venv is the canonical signer python.
#   - the pytest-gpu-proof plugin (PyPI, pinned 0.4.0 — schema 3) and an SSH
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

# Refuse a python that would sign a skip-laden receipt (see Prerequisites).
if ! "$PYTHON" - <<'PY'
import importlib.util, sys
missing = [m for m in ("pinocchio", "mujoco", "scipy", "gymnasium") if importlib.util.find_spec(m) is None]
if missing:
    print("ERROR: %s lacks %s — install the [test] extra: pip install -e '.[test,dev]'"
          % (sys.executable, ", ".join(missing)), file=sys.stderr)
    sys.exit(1)
PY
then exit 1; fi

"$PYTHON" -m pip install -q "pytest-gpu-proof==0.4.0" pyyaml

# Refuse to sign with a partial module set (test/receipt_modules.txt = D12 profile).
missing=()
while read -r plant knot variant; do
    [[ -z "${plant}" || "${plant}" == \#* ]] && continue
    suffix=""; [[ -n "${variant}" && "${variant}" != "default" ]] && suffix="_${variant}"
    compgen -G "python/gato/bsqpN${knot}_${plant}${suffix}.*.so" > /dev/null || missing+=("bsqpN${knot}_${plant}${suffix}")
done < test/receipt_modules.txt
if (( ${#missing[@]} )); then
    echo "ERROR: receipt module set incomplete — missing: ${missing[*]}" >&2
    echo "       build it with: ./tools/build.sh --profile receipt" >&2
    exit 1
fi

# --gpu-proof-github-user: the signer must be the human KEYHOLDER — the
# plugin's remote-derived default would guess the org (A2R-Lab), and orgs have
# no SSH keys. The rest of the config lives in pyproject [tool.gpu_proof].
"$PYTHON" -m pytest test/ -q "$@" \
    --gpu-proof-enable \
    --gpu-proof-out gpu-proof.json \
    --gpu-proof-github-user plancherb1

echo
echo "Signed receipt: gpu-proof.json — 'git add gpu-proof.json' to attest this run."
