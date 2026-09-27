cmake_minimum_required(VERSION 3.27)
project(minimlx_q4_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(MLX CONFIG REQUIRED)
find_package(nanobind CONFIG REQUIRED)

add_library(minimlx_q4_probe_core q4_reference.cpp)
target_include_directories(minimlx_q4_probe_core PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(minimlx_q4_probe_core PUBLIC mlx)

nanobind_add_module(_ext NB_STATIC STABLE_ABI LTO NOMINSIZE NB_DOMAIN mlx bindings.cpp)
target_link_libraries(_ext PRIVATE minimlx_q4_probe_core)
if(APPLE)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
