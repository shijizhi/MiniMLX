cmake_minimum_required(VERSION 3.27)
project(minimlx_q4_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(MLX CONFIG REQUIRED)
find_package(nanobind CONFIG REQUIRED)

add_library(minimlx_q4_kernel_ops SHARED minimlx_q4_qmm.cpp minimlx_fa256.cpp)
target_include_directories(minimlx_q4_kernel_ops PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(
  minimlx_q4_kernel_ops
  PUBLIC mlx
  PRIVATE "-framework Metal" "-framework Foundation")

set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/minimlx_q4_qmm.air)
set(NAX_AIR ${CMAKE_CURRENT_BINARY_DIR}/minimlx_q4_nax.air)
set(FA256_AIR ${CMAKE_CURRENT_BINARY_DIR}/minimlx_fa256.air)
set(METALLIB ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/minimlx_q4_kernels.metallib)
set(NAX_METALLIB ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/minimlx_q4_nax_kernels.metallib)
set(METAL_INCLUDE_ARGS -I${CMAKE_CURRENT_LIST_DIR})
foreach(INCLUDE_DIR IN LISTS MLX_INCLUDE_DIRS)
  list(APPEND METAL_INCLUDE_ARGS -I${INCLUDE_DIR})
endforeach()
add_custom_command(
  OUTPUT ${METAL_AIR}
  COMMAND xcrun -sdk macosx metal -x metal -Wall -Wextra -fno-fast-math
          ${METAL_INCLUDE_ARGS}
          -c ${CMAKE_CURRENT_LIST_DIR}/minimlx_q4_qmm.metal -o ${METAL_AIR}
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/minimlx_q4_qmm.metal
  VERBATIM)
add_custom_command(
  OUTPUT ${NAX_AIR}
  COMMAND xcrun -sdk macosx metal -x metal -Wall -Wextra -fno-fast-math
          -mmacosx-version-min=26.2
          ${METAL_INCLUDE_ARGS}
          -c ${CMAKE_CURRENT_LIST_DIR}/minimlx_q4_nax.metal -o ${NAX_AIR}
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/minimlx_q4_nax.metal
  VERBATIM)
add_custom_command(
  OUTPUT ${FA256_AIR}
  COMMAND xcrun -sdk macosx metal -x metal -Wall -Wextra -fno-fast-math
          ${METAL_INCLUDE_ARGS}
          -c ${CMAKE_CURRENT_LIST_DIR}/minimlx_fa256.metal -o ${FA256_AIR}
  DEPENDS
          ${CMAKE_CURRENT_LIST_DIR}/minimlx_fa256.metal
          ${CMAKE_CURRENT_LIST_DIR}/minimlx_fa256_block_token.metal.h
  VERBATIM)
add_custom_command(
  OUTPUT ${METALLIB}
  COMMAND xcrun -sdk macosx metal ${METAL_AIR} ${FA256_AIR} -o ${METALLIB}
  DEPENDS ${METAL_AIR} ${FA256_AIR}
  VERBATIM)
add_custom_command(
  OUTPUT ${NAX_METALLIB}
  # The NAX AIR requires the 26.2 deployment floor at link time as well.
  # Without it, air-lld can silently drop the tensor-unit object.
  COMMAND xcrun -sdk macosx metal -mmacosx-version-min=26.2
          ${NAX_AIR} -o ${NAX_METALLIB}
  DEPENDS ${NAX_AIR}
  VERBATIM)
add_custom_target(minimlx_q4_kernels_metallib ALL DEPENDS ${METALLIB})
add_custom_target(minimlx_q4_nax_kernels_metallib ALL DEPENDS ${NAX_METALLIB})
add_dependencies(minimlx_q4_kernel_ops minimlx_q4_kernels_metallib)
add_dependencies(minimlx_q4_kernel_ops minimlx_q4_nax_kernels_metallib)

nanobind_add_module(_ext NB_STATIC STABLE_ABI LTO NOMINSIZE NB_DOMAIN mlx bindings.cpp)
target_link_libraries(_ext PRIVATE minimlx_q4_kernel_ops)
if(APPLE)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path -Wl,-rpath,@loader_path/../../../mlx/lib)
endif()
