cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

#these two lines work miracles to make CMake use the compilers you also use
find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)


PROJECT(CUDA-Wrappers
    DESCRIPTION "C++ Wrappers for the CUDA Driver API and related tools"
    VERSION 0.0.1
    HOMEPAGE_URL https://github.com/nlesc-recruit/CUDA-wrappers
    LANGUAGES CUDA CXX)


set(CMAKE_BUILD_TYPE "Release")

find_package(CUDAToolkit REQUIRED)


set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES src/cu.cc src/nvrtc.cc)
include_directories(include)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

