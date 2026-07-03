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
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
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
    GATO_PLANT_HEADER="dynamics/indy7/indy7_plant.cuh")
  # Top-level GLASS (global `glass::`, distinct from grid.cuh's vendored `grid::glass`).
  target_include_directories(bsqp PRIVATE ${CMAKE_SOURCE_DIR}/external/GLASS)
  target_link_libraries(bsqp PRIVATE CUDA::cudart)
endif()

# --- Python Bindings ---

# horizon length and plant options (configurable via -D at configure time).
# Plants are discovered by convention: gato/dynamics/<name>/<name>_plant.cuh —
# adding a robot requires NO CMake edits.
set(KNOTS_DEFAULT "8;16;32;64;128")
set(PLANT_DEFAULT "indy7;iiwa14")
set(KNOTS "${KNOTS_DEFAULT}" CACHE STRING "Semicolon-separated horizon lengths (e.g., 32;64;128)")
set(PLANT "${PLANT_DEFAULT}" CACHE STRING "Semicolon-separated plant targets (e.g., indy7;iiwa14)")
# Normalize to semicolon lists (accept commas/spaces)
string(REPLACE "," ";" KNOTS "${KNOTS}")
string(REPLACE " " ";" KNOTS "${KNOTS}")
string(REPLACE "," ";" PLANT "${PLANT}")
string(REPLACE " " ";" PLANT "${PLANT}")
message(STATUS "Configuring Python modules for PLANT='${PLANT}' KNOTS='${KNOTS}'")

foreach(plant ${PLANT})
  set(PLANT_HEADER_REL "dynamics/${plant}/${plant}_plant.cuh")
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/gato/${PLANT_HEADER_REL}")
    message(FATAL_ERROR "Plant '${plant}': missing ${CMAKE_SOURCE_DIR}/gato/${PLANT_HEADER_REL} "
                        "(generate it with gato.build / tools/regen_grid.py)")
  endif()
  foreach(knot ${KNOTS})
    set(MODULE_NAME "bsqpN${knot}_${plant}")
    message(STATUS "Configuring Python module: ${MODULE_NAME}")
    pybind11_add_module(${MODULE_NAME} SHARED python/bindings.cu)

    target_include_directories(${MODULE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/gato
                                                      ${CMAKE_SOURCE_DIR}/external/GLASS)

    target_compile_definitions(${MODULE_NAME} PRIVATE
      KNOT_POINTS=${knot}
      GATO_PLANT_NAME=${plant}
      GATO_PLANT_HEADER="${PLANT_HEADER_REL}")

    set_target_properties(
      ${MODULE_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                ${CMAKE_SOURCE_DIR}/python/gato)
    target_link_libraries(${MODULE_NAME} PRIVATE CUDA::cudart)

  endforeach()
endforeach()
