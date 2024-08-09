cmake_minimum_required(VERSION 3.14)
project(examples CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.2/bin/nvcc" CACHE PATH "CUDA compiler")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -G")

find_package(CUDA REQUIRED)


set(SRC_DIR ${CMAKE_SOURCE_DIR}/../src)
set(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/../examples)
set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/../external)
set(CONFIG_DIR ${CMAKE_SOURCE_DIR}/../config)


include_directories(
    ${CMAKE_SOURCE_DIR}/..
    ${SRC_DIR}
    ${CONFIG_DIR}
    ${EXTERNAL_DIR}/GLASS
    ${EXTERNAL_DIR}/qdldl/include
)

link_directories(${EXTERNAL_DIR}/qdldl/build/out)

add_executable(pcg ${EXAMPLES_DIR}/single_mpc_simulation_pcg.cu)
set_target_properties(pcg PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(pcg cublas)

add_custom_target(deps DEPENDS 
    ${SRC_DIR}/sim/mpcsim.cuh
    ${SRC_DIR}/solvers/sqp/sqp_pcg.cuh
    ${SRC_DIR}/gato.cuh
    ${CONFIG_DIR}/sim_settings.h
    ${CONFIG_DIR}/solver_settings.h
    ${CONFIG_DIR}/cost_settings.h
)
add_dependencies(pcg deps)