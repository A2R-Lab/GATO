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
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -use_fast_math -O3")

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
target_compile_definitions(bsqp PRIVATE KNOT_POINTS=16 PLANT_INDY7=1)

# --- Python Bindings ---

# horizon length and plant options (configurable via -D at configure time)
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
  foreach(knot ${KNOTS})
    set(MODULE_NAME "bsqpN${knot}_${plant}")
    message(STATUS "Configuring Python module: ${MODULE_NAME}")
    pybind11_add_module(${MODULE_NAME} SHARED python/bindings.cu)
    install(
      TARGETS ${MODULE_NAME}
      DESTINATION .
      COMPONENT python)

    target_include_directories(${MODULE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/gato
                                                      ${CUDA_INCLUDE_DIRS})

    # Define plant type at compile time
    if(plant STREQUAL "indy7")
      target_compile_definitions(${MODULE_NAME} PRIVATE KNOT_POINTS=${knot} PLANT_INDY7=1)
    elseif(plant STREQUAL "iiwa14")
      target_compile_definitions(${MODULE_NAME} PRIVATE KNOT_POINTS=${knot} PLANT_IIWA14=1)
    endif()

    set_target_properties(
      ${MODULE_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                ${CMAKE_SOURCE_DIR}/python/bsqp)
    target_link_libraries(${MODULE_NAME} PRIVATE ${CUDA_LIBRARIES})

  endforeach()
endforeach()
