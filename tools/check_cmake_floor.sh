#!/usr/bin/env bash
# Configure the tree with the oldest CMake the README promises (3.22), from the
# cmake wheel in a scratch venv, so a command newer than the floor fails here
# and not on a user's Ubuntu 22.04 or JetPack 6 box (issue #23). Configure only;
# nothing is compiled. test/test_cmake_floor.py is the static form of this check.
#
#   ./tools/check_cmake_floor.sh [ARCH]      # ARCH default 120
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="${1:-120}"
FLOOR="${CMAKE_FLOOR:-3.22.6}"
WORK="${TMPDIR:-/tmp}/gato-cmake-floor-$$"
PY="${REPO_ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || PY="$(command -v python3)"
mkdir -p "${WORK}"
trap 'rm -rf "${WORK}"' EXIT
python3 -m venv "${WORK}/venv"
"${WORK}/venv/bin/pip" install -q "cmake==${FLOOR}"
"${WORK}/venv/bin/cmake" --version | head -1
"${WORK}/venv/bin/cmake" -S "${REPO_ROOT}" -B "${WORK}/build" -DCMAKE_BUILD_TYPE=Release \
  -DGATO_BUILD_DEMO=OFF -DGATO_RECEIPT_PROFILE=OFF "-DMODULES=indy7:32:default;iiwa14:16:rk4" \
  "-DCMAKE_CUDA_ARCHITECTURES=${ARCH}" "-DPython3_EXECUTABLE=${PY}" \
  "-Dpybind11_DIR=$("${PY}" -m pybind11 --cmakedir)" > "${WORK}/configure.log" 2>&1 \
  || { tail -20 "${WORK}/configure.log"; echo "configure FAILED on CMake ${FLOOR}"; exit 1; }
echo "configure OK on CMake ${FLOOR} (arch ${ARCH})"
