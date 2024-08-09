cmake_minimum_required(VERSION 3.14)
project(MPCGPU CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.2/bin/nvcc" CACHE PATH "CUDA compiler")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -G")

find_package(CUDA REQUIRED)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/../../src)
set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/../../external)
set(CONFIG_DIR ${CMAKE_SOURCE_DIR}/../../config)


include_directories(
    ${CMAKE_SOURCE_DIR}/../..
    ${SRC_DIR}
    ${CONFIG_DIR}
    ${EXTERNAL_DIR}
    ${EXTERNAL_DIR}/GLASS
    ${EXTERNAL_DIR}/qdldl/include
    ${EXTERNAL_DIR}/qdldl/build/include
)

link_directories(${EXTERNAL_DIR}/qdldl/build/out)


add_executable(sqp_pcg ./track_iiwa_pcg.cu)
set_target_properties(sqp_pcg PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(sqp_pcg cublas)

add_custom_target(pcg_deps DEPENDS 
    ${SRC_DIR}/sim/mpcsim.cuh
    ${SRC_DIR}/solvers/sqp/sqp_pcg.cuh
    ${SRC_DIR}/gato.cuh
    ${CONFIG_DIR}/sim_settings.h
    ${CONFIG_DIR}/solver_settings.h
    ${CONFIG_DIR}/cost_settings.h
)
add_dependencies(sqp_pcg pcg_deps)


add_executable(sqp_qdldl ./track_iiwa_qdldl.cu)
set_target_properties(sqp_qdldl PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_definitions(sqp_qdldl PRIVATE LINSYS_SOLVE=0)
target_link_libraries(sqp_qdldl cublas qdldl)

add_custom_target(qdldl_deps DEPENDS 
    ${SRC_DIR}/sim/mpcsim.cuh
    ${SRC_DIR}/solvers/sqp/sqp_pcg.cuh
    ${SRC_DIR}/gato.cuh
    ${CONFIG_DIR}/sim_settings.h
    ${CONFIG_DIR}/solver_settings.h
    ${CONFIG_DIR}/cost_settings.h
)
add_dependencies(sqp_qdldl qdldl_deps)


add_custom_target(build_qdldl
    COMMAND mkdir -p build && cd build && cmake -DQDLDL_FLOAT=true -DQDLDL_LONG=false .. && cmake --build .
    WORKING_DIRECTORY ${EXTERNAL_DIR}/qdldl
)
add_dependencies(sqp_qdldl build_qdldl)