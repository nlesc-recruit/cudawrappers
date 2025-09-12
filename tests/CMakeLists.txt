include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.6.0
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(Catch)

set(COMPONENTS ${CUDAWRAPPERS_MANDATORY_COMPONENTS} vector_add graph)
foreach(component ${CUDAWRAPPERS_COMPONENTS})
  list(FIND COMPONENTS ${component} _found)
  if(_found EQUAL -1)
    list(APPEND COMPONENTS "${component}")
  endif()
endforeach()

set(LINK_LIBRARIES Catch2::Catch2WithMain cudawrappers::cu cudawrappers::nvrtc)

foreach(component ${COMPONENTS})
  if(${CUDAWRAPPERS_BACKEND_HIP})
    set_source_files_properties(test_${component}.cpp PROPERTIES LANGUAGE HIP)
  endif()
  add_executable(test_${component} test_${component}.cpp)
  target_include_directories(
    test_${component} PRIVATE "${CMAKE_SOURCE_DIR}/include"
  )
  catch_discover_tests(test_${component})
  target_link_libraries(test_${component} PUBLIC ${LINK_LIBRARIES})
endforeach()

if(CUDAWRAPPERS_BUILD_NVML AND CUDAWRAPPERS_BACKEND_CUDA)
  target_link_libraries(test_nvml PUBLIC cudawrappers::nvml)
endif()

target_include_directories(
  test_nvrtc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
)

target_embed_source(test_nvrtc kernels/vector_add_kernel.cu)
target_embed_source(test_nvrtc kernels/single_include_kernel.cu)
target_embed_source(test_nvrtc kernels/recursive_include_kernel.cu)

if(CUDAWRAPPERS_BUILD_CUFFT)
  target_link_libraries(test_cufft PUBLIC cudawrappers::cufft)
endif()

if(CUDAWRAPPERS_BUILD_NVTX)
  target_link_libraries(test_nvtx PUBLIC cudawrappers::nvtx)
endif()
