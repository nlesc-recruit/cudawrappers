include(CMakeFindDependencyMacro)

find_package(CUDAToolkit REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/cudawrappers-targets.cmake")
