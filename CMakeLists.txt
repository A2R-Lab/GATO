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

set(CMAKE_CXX_FLAGS_DEBUG
    "-O0 -g -fPIC -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -O3 -std=c++17")

set(CMAKE_CUDA_ARCHITECTURES 89 86 75)
message(STATUS "Targeting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
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

# --- Python Bindings ---

# horizon length options
set(KNOTS 8 16 32 64 128)

foreach(knot ${KNOTS})
  set(MODULE_NAME "bsqpN${knot}")
  message(STATUS "Configuring Python module: ${MODULE_NAME}")
  pybind11_add_module(${MODULE_NAME} SHARED python/bindings.cu)
  install(
    TARGETS ${MODULE_NAME}
    DESTINATION .
    COMPONENT python)

  target_include_directories(${MODULE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/gato
                                                    ${CUDA_INCLUDE_DIRS})

  target_compile_definitions(${MODULE_NAME} PRIVATE KNOT_POINTS=${knot})
  # target_precompile_headers( ${MODULE_NAME} PRIVATE <pybind11/pybind11.h>
  # <pybind11/stl.h> "bsqp/bsqp.cuh" "dynamics/indy7/indy7_plant.cuh"
  # "dynamics/indy7/indy7_grid.cuh")

  set_target_properties(
    ${MODULE_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                              ${CMAKE_SOURCE_DIR}/python/bsqp)
  target_link_libraries(${MODULE_NAME} PRIVATE ${CUDA_LIBRARIES})

endforeach()
