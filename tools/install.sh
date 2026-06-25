#!/usr/bin/env bash
#
# GATO host-native install (no Docker, no uv — assumes CUDA is installed on the
# host, same model as GRiD's base_install.sh). Creates a project-local .venv and
# installs into it.
#
#   ./tools/install.sh              # lean: codegen + build deps + submodules + regen grid.cuh
#   ./tools/install.sh --examples   #   + heavy runtime to run MPC/benchmark examples (torch, pin, viz)
#   ./tools/install.sh --dev        #   + test tooling (pytest)
#   ./tools/install.sh --all        #   + examples + dev
#   ./tools/install.sh --no-regen   # skip the regen_grid.py codegen step
#
# After install:  source .venv/bin/activate  &&  ./tools/build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

WANT_EXAMPLES=0
WANT_DEV=0
DO_REGEN=1

for arg in "$@"; do
  case "${arg}" in
    --examples) WANT_EXAMPLES=1 ;;
    --dev)      WANT_DEV=1 ;;
    --all)      WANT_EXAMPLES=1; WANT_DEV=1 ;;
    --no-regen) DO_REGEN=0 ;;
    -h|--help)
      sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^#\{0,1\} \{0,1\}//'
      exit 0 ;;
    *) echo "unknown option: ${arg}" >&2; exit 2 ;;
  esac
done

# Build the pip extras suffix, e.g. "" or "[examples,dev]".
EXTRAS=""
if (( WANT_EXAMPLES || WANT_DEV )); then
  parts=()
  (( WANT_EXAMPLES )) && parts+=("examples")
  (( WANT_DEV ))      && parts+=("dev")
  EXTRAS="[$(IFS=,; echo "${parts[*]}")]"
fi

echo "----------------------------------------"
echo "Initializing submodules (GRiD + nested GRiDCodeGenerator/URDFParser/RBDReference/GLASS)..."
git -C "${REPO_ROOT}" submodule update --init --recursive

echo "----------------------------------------"
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating venv at ${VENV_DIR} ..."
  python3 -m venv "${VENV_DIR}"
else
  echo "Reusing existing venv at ${VENV_DIR}"
fi

echo "----------------------------------------"
echo "Installing GATO into the venv  (pip install -e .${EXTRAS}) ..."
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -e "${REPO_ROOT}${EXTRAS}"

if (( DO_REGEN )); then
  echo "----------------------------------------"
  echo "Generating grid.cuh for each robot (tools/regen_grid.py) ..."
  "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/regen_grid.py"
fi

echo "----------------------------------------"
echo "Setup complete."
echo " - activate:  source .venv/bin/activate"
echo " - build:     ./tools/build.sh   (or: cmake -DPLANT=... -DKNOTS=... && cmake --build)"
(( WANT_EXAMPLES )) || echo " - to run the examples you also need the runtime stack: ./tools/install.sh --examples"
echo " - paper Fig-3 CPU baseline (optional): ./baselines/build_cpu_baseline.sh builds the threaded"
echo "     BatchThneed (pysqpcpu) — osqp+osqp-eigen into a LOCAL prefix, reusing the venv's cmeel"
echo "     pinocchio (NO ROS, NO source pinocchio). Then: source baselines/sqpcpu_env.sh"
