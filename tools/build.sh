#!/usr/bin/env bash
#
# Incremental build of the GATO solver modules.
#
#   ./tools/build.sh                         # incremental (reuses build/)
#   ./tools/build.sh --clean                 # wipe build/ and reconfigure
#   PLANT=indy7 KNOTS=32 ./tools/build.sh    # subset of the module matrix
#   ARCH=86 ./tools/build.sh                 # override CUDA arch (default: native)
#   JOBS=6  ./tools/build.sh                 # parallel jobs (default 4 — each TU
#                                            #  pulls the large grid.cuh, ~RAM-bound)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
VENV_PY="${REPO_ROOT}/.venv/bin/python"
JOBS="${JOBS:-4}"

if [[ "${1:-}" == "--clean" ]]; then
  echo "Cleaning build directory: ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

PY="$(command -v python || true)"
[[ -x "${VENV_PY}" ]] && PY="${VENV_PY}"

CMAKE_ARGS=(-S "${REPO_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
            -DPython3_EXECUTABLE="${PY}")
# use the venv's pybind11 if available
if PB11="$("${PY}" -m pybind11 --cmakedir 2>/dev/null)"; then
  CMAKE_ARGS+=(-Dpybind11_DIR="${PB11}")
fi
[[ -n "${PLANT:-}" ]] && CMAKE_ARGS+=(-DPLANT="${PLANT}")
[[ -n "${KNOTS:-}" ]] && CMAKE_ARGS+=(-DKNOTS="${KNOTS}")
[[ -n "${ARCH:-}" ]]  && CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${ARCH}")

echo "Configuring: cmake ${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}"

echo "Building (parallel ${JOBS})..."
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

echo "Build complete — modules in python/gato/"
