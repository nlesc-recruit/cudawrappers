cmake_minimum_required(VERSION 3.17)
project(main)

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

include(FetchContent)
FetchContent_Declare(cudawrappers SOURCE_DIR /home/veenboer/src/cudawrappers)
FetchContent_MakeAvailable(cudawrappers)

add_executable(${PROJECT_NAME} main.cu)
target_link_libraries(main cudawrappers::cu)
