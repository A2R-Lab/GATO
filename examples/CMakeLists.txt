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
set(DATA_DIR ${CMAKE_SOURCE_DIR}/../data)


include_directories(
    ${CMAKE_SOURCE_DIR}/..
    ${SRC_DIR}
    ${CONFIG_DIR}
    ${DATA_DIR}
    ${EXTERNAL_DIR}/GLASS
    ${EXTERNAL_DIR}/qdldl/include
    ${EXTERNAL_DIR}/GBD-PCG/include
)

link_directories(${EXTERNAL_DIR}/qdldl/build/out)

#pcg
add_executable(pcg ${EXAMPLES_DIR}/single_mpc_simulation_pcg.cu)
set_target_properties(pcg PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(pcg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --compiler-options -Wall
    -O3
    -G
    -g
>)
target_link_libraries(pcg cublas)

add_custom_target(pcg_deps DEPENDS 
    ${SRC_DIR}/sim/mpcsim.cuh
    ${SRC_DIR}/solvers/sqp/sqp_pcg.cuh
    ${SRC_DIR}/gato.cuh
    ${CONFIG_DIR}/sim_settings.h
    ${CONFIG_DIR}/solver_settings.h
    ${CONFIG_DIR}/cost_settings.h
)
add_dependencies(pcg pcg_deps)




#multi
add_executable(multi ${EXAMPLES_DIR}/sqp_n.cu)
set_target_properties(multi PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(multi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --compiler-options -Wall
    -O3
    -G
    -g
>)
target_link_libraries(multi cublas)

add_custom_target(multi_deps DEPENDS 
    #${SRC_DIR}/sim/mpcsim.cuh
    ${SRC_DIR}/solvers/sqp/sqp_pcg_n.cuh
    ${SRC_DIR}/gato.cuh
    ${CONFIG_DIR}/sim_settings.h
    ${CONFIG_DIR}/solver_settings.h
    ${CONFIG_DIR}/cost_settings.h
)
add_dependencies(multi multi_deps)


#multi-mpc
add_executable(multi-mpc ${EXAMPLES_DIR}/mpc_n.cu)
set_target_properties(multi-mpc PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(multi-mpc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --compiler-options -Wall
    -O3
    -G
    -g
>)
target_link_libraries(multi-mpc cublas)

add_custom_target(multi_mpc_deps DEPENDS 
    ${SRC_DIR}/sim/mpcsim_n.cuh
    ${SRC_DIR}/solvers/sqp/sqp_pcg_n.cuh
    ${SRC_DIR}/gato.cuh
    ${CONFIG_DIR}/sim_settings.h
    ${CONFIG_DIR}/solver_settings.h
    ${CONFIG_DIR}/cost_settings.h
)
add_dependencies(multi multi_mpc_deps)