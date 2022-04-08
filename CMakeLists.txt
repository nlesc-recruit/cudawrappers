cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(
  cudawrappers
  DESCRIPTION "C++ Wrappers for the CUDA Driver API"
  VERSION 0.3.0
  HOMEPAGE_URL https://github.com/nlesc-recruit/cudawrappers
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")

find_package(CUDAToolkit REQUIRED)

# --- auto-ignore build directory
if(NOT EXISTS ${PROJECT_BINARY_DIR}/.gitignore)
  file(WRITE ${PROJECT_BINARY_DIR}/.gitignore "*")
endif()

set(SOURCE_FILES src/cu.cpp src/nvrtc.cpp)

add_library(cudawrappers SHARED ${SOURCE_FILES})

target_include_directories(
  cudawrappers
  PUBLIC $<INSTALL_INTERFACE:include/cudawrappers>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(cudawrappers PUBLIC CUDA::cuda_driver CUDA::nvrtc)

# Including linters rules
include(cmake/linter-tools.cmake)

# Tests
add_subdirectory(tests)

# Installation
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/cu.hpp
              ${CMAKE_CURRENT_SOURCE_DIR}/include/nvrtc.hpp
        DESTINATION "include/cudawrappers")

install(
  TARGETS cudawrappers
  EXPORT cudawrappers-export
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(
  EXPORT cudawrappers-export
  FILE cudawrappers-targets.cmake
  DESTINATION lib/cmake/cudawrappers)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cudawrappers-config.cmake
        DESTINATION "lib/cmake/cudawrappers")
