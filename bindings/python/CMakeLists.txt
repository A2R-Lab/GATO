cmake_minimum_required(VERSION 3.14)
project(GATO CUDA CXX)

# Common settings
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.2/bin/nvcc" CACHE PATH "CUDA compiler")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -G")

find_package(CUDA REQUIRED)

# make sure to find python from conda, if a conda virtual env is activated
set(Python3_FIND_VIRTUALENV "STANDARD")
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/../../gato)
set(DEPENDENCIES_DIR ${CMAKE_SOURCE_DIR}/../../dependencies)
set(CONFIG_DIR ${CMAKE_SOURCE_DIR}/../../config)
set(DATA_DIR ${CMAKE_SOURCE_DIR}/../../data)


include_directories(
    ${SRC_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/../..
    ${CONFIG_DIR}
    ${DATA_DIR}
    ${DEPENDENCIES_DIR}
    ${DEPENDENCIES_DIR}/pybind11/include
    ${DEPENDENCIES_DIR}/GLASS
)

cuda_add_library(gato SHARED
    gato_wrapper.cu)
    

#target_include_directories(gpu_library PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(gato PUBLIC)
target_link_libraries(gato 
    cublas
    Python3::Python
    cudart)

# set what comes before the library name, here no "lib"
set_target_properties(gato PROPERTIES PREFIX "")

#configure_file(test.py test.py COPYONLY)

#-------------------------------------------------------------------#
message("//===================================================")
message("// Generating Python bindings for gato")
message("//===================================================")
message("  CUDA compiler ID      : ${CMAKE_CUDA_COMPILER_ID}")
message("  CUDA compiler Version : ${CMAKE_CUDA_COMPILER_VERSION}")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
    "${CMAKE_CXX_COMPILER_VERSION} "
    "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
message("  CUDA Compile flags : ${CMAKE_CUDA_FLAGS}")
message("  CUDA toolkit inc   : ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message("")
message("  Python3_found       : ${Python3_FOUND}")
message("  Python3 version     : ${Python3_VERSION}")
message("  Python3 interpreter : ${Python3_EXECUTABLE}")
message("  Python3 interp. Id  : ${Python3_INTERPRETER_ID}")
message("  Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message("  Python3_LIBRARY_DIRS: ${Python3_LIBRARY_DIRS}")
message("//===================================================")