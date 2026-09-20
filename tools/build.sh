#!/usr/bin/env bash
#
# Incremental build of the GATO solver modules.
#
#   ./tools/build.sh                         # incremental (reuses build/)
#   ./tools/build.sh --clean                 # wipe build/ and reconfigure
#   PLANT=indy7 KNOTS=32 ./tools/build.sh    # subset of the module matrix
#   MODULES="indy7:8,16;go2:16" ./tools/build.sh   # explicit per-plant horizons
#   ./tools/build.sh --profile receipt       # exactly test/receipt_modules.txt (what the receipt attests)
#   ./tools/build.sh --variant fc            # fc (contact-force) or eh (exact-Hessian) variant modules
#                                            #  (bsqpN{N}_{plant}_fc.so, side by side with the defaults)
#   ARCH=86 ./tools/build.sh                 # override CUDA arch (default: native)
#   JOBS=6  ./tools/build.sh                 # parallel jobs (default 4 — each TU
#                                            #  pulls the large grid.cuh, ~RAM-bound)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
VENV_PY="${REPO_ROOT}/.venv/bin/python"
JOBS="${JOBS:-4}"

PROFILE=""
VARIANT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) echo "Cleaning build directory: ${BUILD_DIR}"; rm -rf "${BUILD_DIR}"; shift ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

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
[[ -n "${MODULES:-}" ]] && CMAKE_ARGS+=(-DMODULES="${MODULES}")
case "${VARIANT}" in
  "")  CMAKE_ARGS+=(-DGATO_CONTACT_FORCES=OFF -DGATO_EXACT_HESSIAN=OFF) ;;
  fc)  CMAKE_ARGS+=(-DGATO_CONTACT_FORCES=ON -DGATO_EXACT_HESSIAN=OFF) ;;
  eh)  CMAKE_ARGS+=(-DGATO_CONTACT_FORCES=OFF -DGATO_EXACT_HESSIAN=ON) ;;
  *) echo "unknown --variant '${VARIANT}' (fc|eh)" >&2; exit 2 ;;
esac
case "${PROFILE}" in
  "")       CMAKE_ARGS+=(-DGATO_RECEIPT_PROFILE=OFF) ;;
  receipt)  CMAKE_ARGS+=(-DGATO_RECEIPT_PROFILE=ON) ;;
  *) echo "unknown --profile '${PROFILE}' (only: receipt)" >&2; exit 2 ;;
esac

echo "Configuring: cmake ${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}"

echo "Building (parallel ${JOBS})..."
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

echo "Build complete — modules in python/gato/"
