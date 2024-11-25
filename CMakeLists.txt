cmake_minimum_required(VERSION 3.14)
project(GATO CUDA CXX)

# Common settings
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.2/bin/nvcc" CACHE PATH "CUDA compiler")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -G")

find_package(CUDA REQUIRED)

# Directory settings
set(SRC_DIR ${CMAKE_SOURCE_DIR}/gato)
set(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/examples)
set(DEPENDENCIES_DIR ${CMAKE_SOURCE_DIR}/dependencies)
set(CONFIG_DIR ${CMAKE_SOURCE_DIR}/config)
set(DATA_DIR ${CMAKE_SOURCE_DIR}/data)
set(MPCGPU_DIR ${CMAKE_SOURCE_DIR}/experiments/MPCGPU)
set(MULTISOLVE_DIR ${CMAKE_SOURCE_DIR}/experiments/multisolve)



# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${SRC_DIR}
    ${CONFIG_DIR}
    ${DATA_DIR}
    ${DEPENDENCIES_DIR}
    ${DEPENDENCIES_DIR}/GLASS
    ${DEPENDENCIES_DIR}/qdldl/include
    ${DEPENDENCIES_DIR}/qdldl/build/include
)

link_directories(${DEPENDENCIES_DIR}/qdldl/build/out)


set(CUDA_COMPILE_OPTIONS
    --compiler-options -Wall
    -O3
    -G
    -g
)


# Function to add a CUDA executable with common properties
function(add_cuda_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    set_target_properties(${target_name} PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS}>)
    target_link_libraries(${target_name} cublas)
endfunction()



# Targets from examples/CMakeLists.txt
add_cuda_executable(mpc ${EXAMPLES_DIR}/mpc.cu)
add_cuda_executable(multi-sqp ${EXAMPLES_DIR}/sqp_n.cu)
add_cuda_executable(multi-mpc ${EXAMPLES_DIR}/mpc_n.cu)

# Targets from experiments/MPCGPU/CMakeLists.txt
add_cuda_executable(MPCGPU-pcg ${MPCGPU_DIR}/track_iiwa_pcg.cu)
add_cuda_executable(MPCGPU-qdldl ${MPCGPU_DIR}/track_iiwa_qdldl.cu)
target_compile_definitions(MPCGPU-qdldl PRIVATE LINSYS_SOLVE=0)
target_link_libraries(MPCGPU-qdldl qdldl)

# Targets from experiments/multisolve/CMakeLists.txt
add_cuda_executable(multisolve-sqp ${MULTISOLVE_DIR}/test_sqp_n.cu)  
add_cuda_executable(multisolve-sqp-single ${MULTISOLVE_DIR}/test_sqp_single.cu)  

# Custom targets for dependencies
add_custom_target(deps DEPENDS 
    ${SRC_DIR}/sim/mpcsim.cuh
    ${SRC_DIR}/sim/mpcsim_n.cuh
    ${SRC_DIR}/solvers/sqp/sqp_pcg.cuh
    ${SRC_DIR}/solvers/sqp/sqp_pcg_n.cuh
    ${SRC_DIR}/solvers/sqp/sqp_qdldl.cuh
    ${SRC_DIR}/gato.cuh
    ${CONFIG_DIR}/sim_settings.h
    ${CONFIG_DIR}/solver_settings.h
    ${CONFIG_DIR}/cost_settings.h
)

# Add dependencies
add_dependencies(mpc deps)
add_dependencies(multi-sqp deps)
add_dependencies(multi-mpc deps)
add_dependencies(MPCGPU-pcg deps)
add_dependencies(MPCGPU-qdldl deps)
add_dependencies(multisolve-sqp deps)
add_dependencies(multisolve-sqp-single deps)

# Build QDLDL library
add_custom_target(build_qdldl
    COMMAND mkdir -p build && cd build && cmake -DQDLDL_FLOAT=true -DQDLDL_LONG=false .. && cmake --build .
    WORKING_DIRECTORY ${DEPENDENCIES_DIR}/qdldl
)
add_dependencies(MPCGPU-qdldl build_qdldl)
