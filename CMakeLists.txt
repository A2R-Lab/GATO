#
# cmake -S . -B build [-DPLANT="indy7;iiwa14"] [-DKNOTS="8;16;32;64;128"]
# cmake --build build --parallel 4
#

cmake_minimum_required(VERSION 3.22)
project(GATO LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(GATO_FAST_MATH "Compile CUDA with -use_fast_math (disable for tight-tolerance validation)" ON)
option(GATO_BUILD_DEMO "Build the bsqp C++ demo executable" ON)
option(GATO_EXACT_HESSIAN "Compile the exact-Hessian (SO-SQP) stage-projection path; OFF keeps the device tree preprocessor-identical to the GN-only solver" OFF)
option(GATO_CONTACT_FORCES "CL-3a: append a per-knot contact wrench to the control vector (f_c extends u; module-ABI change — build into build_fc/); OFF keeps the baseline solver bit-identical" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
# NOTE: -G is an nvcc flag and lives in the CUDA debug flags only; ASan on a CUDA
# pybind module is fraught, so host debug flags stay plain.
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fPIC -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
if(GATO_FAST_MATH)
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -use_fast_math")
endif()

# "native" (CMake 3.24+) detects the local GPU at configure time and compiles
# only for that architecture. On older CMake (e.g. 3.22 on Ubuntu 22.04), fall
# back to a broad list of recent arches. Override at configure time with: cmake
# -DCMAKE_CUDA_ARCHITECTURES="75;86;87;89;100;120" ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    # No silent fallback to a stale arch list (the old 75;86;89 default could
    # not even target the sm_120 boxes this repo is developed on): older CMake
    # cannot detect the GPU, so the caller must say which one to build for.
    message(FATAL_ERROR "CMake < 3.24 cannot detect the CUDA arch ('native'); pass "
                        "-DCMAKE_CUDA_ARCHITECTURES=<sm> explicitly (e.g. 120 for RTX 50xx, 89 for RTX 40xx)")
  endif()
endif()
message(STATUS "Targeting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/gato)

# --- Executable Targets ---
if(GATO_BUILD_DEMO)
  add_executable(bsqp examples/bsqp.cu)
  target_compile_definitions(bsqp PRIVATE
    KNOT_POINTS=16
    GATO_PLANT_NAME=indy7
    GATO_PLANT_HEADER="dynamics/plant.cuh")
  if(GATO_EXACT_HESSIAN)
    target_compile_definitions(bsqp PRIVATE USE_EXACT_HESSIAN=1)
  endif()
  if(GATO_CONTACT_FORCES)
    target_compile_definitions(bsqp PRIVATE GATO_CONTACT_FORCES=1)
  endif()
  # Top-level GLASS (global `glass::`, distinct from grid.cuh's vendored `grid::glass`).
  # The per-robot dir supplies grid.cuh + limits.cuh to the generic plant header.
  target_include_directories(bsqp PRIVATE ${CMAKE_SOURCE_DIR}/gato/dynamics/indy7
                                          ${CMAKE_SOURCE_DIR}/external/GLASS
                                          ${CMAKE_SOURCE_DIR}/external/GRiD/grid_codegen/collision)
  target_link_libraries(bsqp PRIVATE CUDA::cudart)
endif()

# --- Python Bindings ---

# horizon length and plant options (configurable via -D at configure time).
# Plants are discovered by convention: gato/dynamics/<name>/{grid.cuh,limits.cuh}
# (generated) + the shared gato/dynamics/plant.cuh adapter — adding a robot
# requires NO CMake edits.
set(KNOTS_DEFAULT "8;16;32;64;128")
set(PLANT_DEFAULT "indy7;iiwa14")
# Where the built modules land. Default = the python package (import gato picks
# them up). Point it elsewhere for side-by-side variant builds (e.g. an
# exact-Hessian A/B) so a variant configure can't clobber the default modules.
set(GATO_MODULE_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/python/gato" CACHE PATH "Output directory for the bsqp python modules")
set(KNOTS "${KNOTS_DEFAULT}" CACHE STRING "Semicolon-separated horizon lengths (e.g., 32;64;128)")
set(PLANT "${PLANT_DEFAULT}" CACHE STRING "Semicolon-separated plant targets (e.g., indy7;iiwa14)")
# Explicit per-plant horizons: 'plant:N,N;plant:N' (overrides the PLANT x KNOTS
# cross-product — needed whenever plants have DIFFERENT supported horizons,
# e.g. go2 is N16-only while the arms carry the full ladder).
set(MODULES "" CACHE STRING "Explicit module list 'indy7:8,16;go2:16' (overrides PLANT x KNOTS)")
# The receipt profile = exactly test/receipt_modules.txt (the set the signed
# gpu-proof receipt attests); run_gpu_proof.sh preflights the same file.
option(GATO_RECEIPT_PROFILE "Configure exactly the receipt module set (test/receipt_modules.txt)" OFF)
# Normalize to semicolon lists (accept commas/spaces)
string(REPLACE "," ";" KNOTS "${KNOTS}")
string(REPLACE " " ";" KNOTS "${KNOTS}")
string(REPLACE "," ";" PLANT "${PLANT}")
string(REPLACE " " ";" PLANT "${PLANT}")

# Resolve the module set into a list of "plant:knot" pairs.
set(GATO_MODULE_PAIRS "")
if(GATO_RECEIPT_PROFILE)
  file(STRINGS "${CMAKE_SOURCE_DIR}/test/receipt_modules.txt" _receipt_lines REGEX "^[a-z0-9_]+[ \t]+[0-9]+")
  foreach(_line ${_receipt_lines})
    string(REGEX MATCH "^([a-z0-9_]+)[ \t]+([0-9]+)" _m "${_line}")
    list(APPEND GATO_MODULE_PAIRS "${CMAKE_MATCH_1}:${CMAKE_MATCH_2}")
  endforeach()
  message(STATUS "Receipt profile: ${GATO_MODULE_PAIRS}")
elseif(MODULES)
  string(REPLACE " " ";" _entries "${MODULES}")
  foreach(_entry ${_entries})
    string(REGEX MATCH "^([a-z0-9_]+):([0-9,]+)$" _m "${_entry}")
    if(NOT _m)
      message(FATAL_ERROR "MODULES entry '${_entry}' is not 'plant:N,N'")
    endif()
    string(REPLACE "," ";" _ks "${CMAKE_MATCH_2}")
    foreach(_k ${_ks})
      list(APPEND GATO_MODULE_PAIRS "${CMAKE_MATCH_1}:${_k}")
    endforeach()
  endforeach()
  message(STATUS "Explicit modules: ${GATO_MODULE_PAIRS}")
else()
  foreach(plant ${PLANT})
    foreach(knot ${KNOTS})
      list(APPEND GATO_MODULE_PAIRS "${plant}:${knot}")
    endforeach()
  endforeach()
  message(STATUS "Configuring Python modules for PLANT='${PLANT}' KNOTS='${KNOTS}'")
endif()

foreach(pair ${GATO_MODULE_PAIRS})
  string(REPLACE ":" ";" _pk "${pair}")
  list(GET _pk 0 plant)
  list(GET _pk 1 knot)
  foreach(gen_header grid.cuh limits.cuh)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/gato/dynamics/${plant}/${gen_header}")
      message(FATAL_ERROR "Plant '${plant}': missing gato/dynamics/${plant}/${gen_header} "
                          "(generate it with gato.build / tools/regen_grid.py)")
    endif()
  endforeach()
  block()
    set(MODULE_NAME "bsqpN${knot}_${plant}")
    message(STATUS "Configuring Python module: ${MODULE_NAME}")
    pybind11_add_module(${MODULE_NAME} SHARED python/bindings.cu)

    # The per-robot dir supplies grid.cuh + limits.cuh to the generic plant
    # header; the GRiD collision dir supplies grid_collision_geometry.cuh
    # (SDF primitives grid.cuh includes when built with collision_res).
    target_include_directories(${MODULE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/gato
                                                      ${CMAKE_SOURCE_DIR}/gato/dynamics/${plant}
                                                      ${CMAKE_SOURCE_DIR}/external/GLASS
                                                      ${CMAKE_SOURCE_DIR}/external/GRiD/grid_codegen/collision)

    target_compile_definitions(${MODULE_NAME} PRIVATE
      KNOT_POINTS=${knot}
      GATO_PLANT_NAME=${plant}
      GATO_PLANT_HEADER="dynamics/plant.cuh")
    if(GATO_EXACT_HESSIAN)
      target_compile_definitions(${MODULE_NAME} PRIVATE USE_EXACT_HESSIAN=1)
    endif()
    if(GATO_CONTACT_FORCES)
      target_compile_definitions(${MODULE_NAME} PRIVATE GATO_CONTACT_FORCES=1)
    endif()

    set_target_properties(
      ${MODULE_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                ${GATO_MODULE_OUTPUT_DIR})
    target_link_libraries(${MODULE_NAME} PRIVATE CUDA::cudart)
  endblock()
endforeach()
