cmake_minimum_required(VERSION 3.18)
project(GATO LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG
    "-O0 -g -fPIC -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -O3 -std=c++17")

set(CMAKE_CUDA_ARCHITECTURES 89 86 75)
message(STATUS "Targeting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(Eigen3 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDA REQUIRED)

message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")

include_directories(${CMAKE_SOURCE_DIR}/gato ${EIGEN3_INCLUDE_DIR})

# output directory for executables (relative to build directory)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Executable Targets ---
# add_executable(single_sqp examples/single_sqp.cu) add_executable(batch_sqp
# examples/batch_sqp.cu) add_executable(benchmark_batch_sqp
# examples/benchmark_batch_sqp.cu)

# Link Eigen3 if necessary target_link_libraries(single_sqp PRIVATE
# Eigen3::Eigen) target_link_libraries(batch_sqp PRIVATE Eigen3::Eigen)
# target_link_libraries(benchmark_batch_sqp PRIVATE Eigen3::Eigen)

# --- Python Bindings ---
set(KNOTS 8 16 32 64 128 256)

foreach(knot ${KNOTS})
  set(MODULE_NAME "bsqpN${knot}")
  message(STATUS "Configuring Python module: ${MODULE_NAME}")

  pybind11_add_module(${MODULE_NAME} SHARED python/bindings.cu)
  install(
    TARGETS ${MODULE_NAME}
    DESTINATION .
    COMPONENT python)

  target_include_directories(
    ${MODULE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/gato ${EIGEN3_INCLUDE_DIR}
                           ${CUDA_INCLUDE_DIRS})

  target_compile_definitions(${MODULE_NAME} PRIVATE KNOT_POINTS=${knot})
  target_compile_options(
    ${MODULE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe
                           --diag_suppress=20012>) # suppress CUDA warnings
                                                   # about Eigen
  # --- Precompiled Headers ---
  # List headers that are included frequently and change infrequently. Add the
  # main header(s) from your custom header-only library here.
  target_precompile_headers(
    ${MODULE_NAME}
    PRIVATE
    <Eigen/Core>
    <pybind11/pybind11.h>
    <pybind11/eigen.h> # Assuming you bind Eigen types
    <pybind11/stl.h> # Assuming you bind STL containers
    "bsqp/bsqp.cuh"
    "dynamics/indy7/indy7_plant.cuh"
    "dynamics/indy7/indy7_grid.cuh")
  # --- End Precompiled Headers ---
  set_target_properties(
    ${MODULE_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                              ${CMAKE_SOURCE_DIR}/python/bsqp)
  target_link_libraries(${MODULE_NAME} PRIVATE Eigen3::Eigen ${CUDA_LIBRARIES})

endforeach()

# mkdir build && cd build cmake .. cmake --build . --parallel
