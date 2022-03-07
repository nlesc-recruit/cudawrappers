cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(
  cudawrappers
  DESCRIPTION "C++ Wrappers for the CUDA Driver API"
  VERSION 0.2.0
  HOMEPAGE_URL https://github.com/nlesc-recruit/cudawrappers
  LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")

find_package(CUDAToolkit REQUIRED)

# --- auto-ignore build directory
if(NOT EXISTS ${PROJECT_BINARY_DIR}/.gitignore)
  file(WRITE ${PROJECT_BINARY_DIR}/.gitignore "*")
endif()

set(SOURCE_FILES src/cu.cpp src/nvrtc.cpp)

include_directories(./include/)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

add_library(cudawrappers SHARED ${SOURCE_FILES})

# Including linters rules
include(cmake/linter-tools.cmake)

# Tests
add_subdirectory(tests)
