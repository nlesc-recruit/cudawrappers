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

set(SOURCE_FILES src/cu.cpp src/nvrtc.cpp)
include_directories(./include/)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

add_library(cudawrappers SHARED ${SOURCE_FILES})
