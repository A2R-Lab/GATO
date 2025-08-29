#
# mkdir build && cd build cmake .. cmake --build . --parallel
#

cmake_minimum_required(VERSION 3.18)
project(GATO LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG
    "-O0 -g -G -fPIC -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC -DNDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -use_fast_math -O3")

set(CMAKE_CUDA_ARCHITECTURES 89 86 75)
message(STATUS "Targeting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Ensure CMake uses the correct CUDA toolkit
set(CUDAToolkit_ROOT
    "/usr/local/cuda-12.6"
    CACHE PATH "CUDA Toolkit root directory")

# Check CUDA version and warn if not 12.6
execute_process(
  COMMAND ${CUDAToolkit_ROOT}/bin/nvcc --version
  OUTPUT_VARIABLE NVCC_VERSION_OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE)
# string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" _ ${NVCC_VERSION_OUTPUT})
# set(CUDA_VERSION_DETECTED ${CMAKE_MATCH_1}) if(NOT CUDA_VERSION_DETECTED
# STREQUAL "12.6") message(WARNING "Detected CUDA version is
# ${CUDA_VERSION_DETECTED}, but CUDA 12.6 is required for sm_89 support. Please
# check your environment.") endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
# --- Automatically detect pybind11 in the active Python environment ---
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import pybind11, os; print(os.path.join(pybind11.get_cmake_dir()))"
  OUTPUT_VARIABLE pybind11_cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(EXISTS "${pybind11_cmake_dir}/pybind11Config.cmake")
  list(APPEND CMAKE_PREFIX_PATH "${pybind11_cmake_dir}")
endif()
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDA REQUIRED)

message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")

include_directories(${CMAKE_SOURCE_DIR}/gato)

# output directory for executables (relative to build directory)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Executable Targets ---
add_executable(bsqp examples/bsqp.cu)
target_compile_definitions(bsqp PRIVATE KNOT_POINTS=16)

# Add integrator test executable
add_executable(integratortest examples/integratortest.cu)
# Use the same include directories as bsqp implicitly has
target_include_directories(integratortest PRIVATE ${CMAKE_SOURCE_DIR})

# --- Python Bindings ---

# horizon length options
set(KNOTS
    8
    16
    32
    64
    128
    256
    512)

foreach(knot ${KNOTS})
  set(MODULE_NAME "bsqpN${knot}")
  message(STATUS "Configuring Python module: ${MODULE_NAME}")
  pybind11_add_module(${MODULE_NAME} SHARED python/bindings.cu)

  # Use build directory for output, not source
  set_target_properties(
    ${MODULE_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                              ${CMAKE_BINARY_DIR}/python/bsqp)

  # Install to a relative path under the install prefix
  install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION python/bsqp
                                         COMPONENT python)

  target_include_directories(${MODULE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/gato
                                                    ${CUDA_INCLUDE_DIRS})

  target_compile_definitions(${MODULE_NAME} PRIVATE KNOT_POINTS=${knot})
  # target_precompile_headers( ${MODULE_NAME} PRIVATE <pybind11/pybind11.h>
  # <pybind11/stl.h> "bsqp/bsqp.cuh" "dynamics/indy7/indy7_plant.cuh"
  # "dynamics/indy7/indy7_grid.cuh")

  target_link_libraries(${MODULE_NAME} PRIVATE ${CUDA_LIBRARIES})

endforeach()
