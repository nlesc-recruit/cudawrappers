include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.6.0
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(Catch)

set(COMPONENTS ${CUDAWRAPPERS_MANDATORY_COMPONENTS} vector_add graph
               comprehensive)
foreach(component ${CUDAWRAPPERS_COMPONENTS})
  list(FIND COMPONENTS ${component} _found)
  if(_found EQUAL -1)
    list(APPEND COMPONENTS "${component}")
  endif()
endforeach()

if(CUDAWRAPPERS_BACKEND_ALL)
  # --- Dual-backend mode: unified cu library, single test per component ---

  foreach(component ${COMPONENTS})
    set(test_name test_${component})
    add_executable(${test_name} test_${component}.cpp)
    target_include_directories(
      ${test_name} PRIVATE "${CMAKE_SOURCE_DIR}/include"
    )
    target_link_libraries(
      ${test_name} PUBLIC Catch2::Catch2WithMain cudawrappers::cu gcov
    )
    catch_discover_tests(${test_name})
  endforeach()

  # NVML (CUDA-only, skip for now in unified mode)
  if(CUDAWRAPPERS_BUILD_NVML)
    target_link_libraries(test_nvml PUBLIC cudawrappers::nvml_cuda)
  endif()

  # Tests that include nvrtc.hpp need the nvrtc_cuda library
  if(CUDAWRAPPERS_BUILD_NVRTC)
    target_link_libraries(test_nvrtc PUBLIC cudawrappers::nvrtc_cuda)
    target_link_libraries(test_vector_add PUBLIC cudawrappers::nvrtc_cuda)
    target_link_libraries(test_graph PUBLIC cudawrappers::nvrtc_cuda)
  endif()

  # NVRTC kernels
  target_include_directories(
    test_nvrtc PRIVATE "${CMAKE_SOURCE_DIR}/include"
                       "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
  )
  target_embed_source(test_nvrtc kernels/vector_add_kernel.cu)
  target_embed_source(test_nvrtc kernels/single_include_kernel.cu)
  target_embed_source(test_nvrtc kernels/recursive_include_kernel.cu)

  # cuFFT
  if(CUDAWRAPPERS_BUILD_CUFFT)
    target_link_libraries(test_cufft PUBLIC cudawrappers::cufft_cuda)
  endif()

  # NVTX
  if(CUDAWRAPPERS_BUILD_NVTX)
    target_link_libraries(test_nvtx PUBLIC cudawrappers::nvtx_cuda)
  endif()

  # Multi-backend test: single binary that lists GPUs from both CUDA and HIP
  set(cuda_helpers_obj test_multi_backend_cuda_helpers)
  add_library(${cuda_helpers_obj} OBJECT cuda_helpers.cpp)
  target_link_libraries(${cuda_helpers_obj} PRIVATE CUDA::cuda_driver)

  set(hip_helpers_obj test_multi_backend_hip_helpers)
  set(hip_helpers_wrapper
      "${CMAKE_CURRENT_BINARY_DIR}/hip_helpers_wrapper.cpp"
  )
  file(WRITE "${hip_helpers_wrapper}"
       "#include \"${CMAKE_CURRENT_SOURCE_DIR}/hip_helpers.cpp\"\n"
  )
  set_source_files_properties("${hip_helpers_wrapper}" PROPERTIES LANGUAGE HIP)
  add_library(${hip_helpers_obj} OBJECT "${hip_helpers_wrapper}")
  target_link_libraries(${hip_helpers_obj} PRIVATE hip::host)

  add_executable(test_multi_backend test_multi_backend.cpp)
  target_include_directories(
    test_multi_backend PRIVATE "${CMAKE_SOURCE_DIR}/include"
  )
  target_link_libraries(
    test_multi_backend
    PUBLIC Catch2::Catch2WithMain ${cuda_helpers_obj} ${hip_helpers_obj} gcov
  )
  catch_discover_tests(test_multi_backend)

else()
  # --- Single-backend mode: existing behavior ---

  set(LINK_LIBRARIES Catch2::Catch2WithMain cudawrappers::cu
                      cudawrappers::nvrtc
  )

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
    test_nvrtc PRIVATE "${CMAKE_SOURCE_DIR}/include"
                       "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
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
endif()
