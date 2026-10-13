# Standalone build of a C++/CUDA program against an installed gato-mpc (a wheel
# or an editable checkout): `python -m gato.paths` supplies the include
# directories, defines and flags for the chosen (robot, horizon, variant).
#
#   cmake -S examples/cpp -B build-cpp -DGATO_ROBOT=indy7 -DGATO_N=32 [-DPython3_EXECUTABLE=.venv/bin/python]
#   cmake --build build-cpp && ./build-cpp/batched_solve
#
# Robots registered with gato.register_robot resolve the same way (their
# generated headers live in the per-host cache). Each translation unit pulls
# the whole generated grid.cuh: expect a few minutes and ~6 GB of RAM per TU.
cmake_minimum_required(VERSION 3.24)
project(gato_cpp_example LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
set(GATO_ROBOT "indy7" CACHE STRING "plant name (vendored or registered with gato.register_robot)")
set(GATO_N "32" CACHE STRING "horizon length (knot points)")
set(GATO_VARIANT "default" CACHE STRING "default | fc | eh")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND ${Python3_EXECUTABLE} -m gato.paths --robot ${GATO_ROBOT} --N ${GATO_N}
                        --variant ${GATO_VARIANT} --format cmake
                OUTPUT_VARIABLE _gato OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "|" ";" _parts "${_gato}")       # include dirs | defines | flags
list(GET _parts 0 _inc)
list(GET _parts 1 _def)
list(GET _parts 2 _flags)
string(REPLACE "," ";" GATO_INCLUDE_DIRS "${_inc}")
string(REPLACE "," ";" GATO_DEFINES "${_def}")
string(REPLACE "," ";" GATO_CUDA_FLAGS "${_flags}")
message(STATUS "gato include dirs: ${GATO_INCLUDE_DIRS}")
message(STATUS "gato defines: ${GATO_DEFINES}")

find_package(CUDAToolkit REQUIRED)
add_executable(batched_solve batched_solve.cu)
target_include_directories(batched_solve PRIVATE ${GATO_INCLUDE_DIRS})
target_compile_definitions(batched_solve PRIVATE ${GATO_DEFINES} GATO_PLANT_NAME_STR="${GATO_ROBOT}")
target_compile_options(batched_solve PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GATO_CUDA_FLAGS}>)
target_link_libraries(batched_solve PRIVATE CUDA::cudart)
