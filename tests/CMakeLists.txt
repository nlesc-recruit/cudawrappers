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

if(CUDAWRAPPERS_BACKEND_ALL)
  # --- Dual-backend mode: unified cu library, single test per component ---

  foreach(component ${COMPONENTS})
    set(TEST_NAME test_${component})
    add_executable(${TEST_NAME} test_${component}.cpp)
    target_include_directories(
      ${TEST_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/include"
    )
    set(LINK_LIBS Catch2::Catch2 cudawrappers::cu)
    if(NOT CUDAWRAPPERS_BACKEND_HIP)
      list(APPEND LINK_LIBS gcov)
    endif()
    target_link_libraries(
      ${TEST_NAME} PUBLIC Catch2::Catch2WithMain ${LINK_LIBS}
    )
    catch_discover_tests(${TEST_NAME})
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
  set(CUDA_HELPERS_OBJ test_multi_backend_cuda_helpers)
  add_library(${CUDA_HELPERS_OBJ} OBJECT cuda_helpers.cpp)
  target_link_libraries(${CUDA_HELPERS_OBJ} PRIVATE CUDA::cuda_driver)

  set(HIP_HELPERS_OBJ test_multi_backend_hip_helpers)
  set(HIP_HELPERS_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/hip_helpers_wrapper.cpp")
  file(WRITE "${HIP_HELPERS_WRAPPER}"
       "#include \"${CMAKE_CURRENT_SOURCE_DIR}/hip_helpers.cpp\"\n"
  )
  set_source_files_properties("${HIP_HELPERS_WRAPPER}" PROPERTIES LANGUAGE HIP)
  add_library(${HIP_HELPERS_OBJ} OBJECT "${HIP_HELPERS_WRAPPER}")
  target_link_libraries(${HIP_HELPERS_OBJ} PRIVATE hip::host)

  add_executable(test_multi_backend test_multi_backend.cpp)
  target_include_directories(
    test_multi_backend PRIVATE "${CMAKE_SOURCE_DIR}/include"
  )
  set(LINK_LIBS Catch2::Catch2WithMain ${CUDA_HELPERS_OBJ} ${HIP_HELPERS_OBJ})
  if(NOT CUDAWRAPPERS_BACKEND_HIP)
    list(APPEND LINK_LIBS gcov)
  endif()
  target_link_libraries(test_multi_backend PUBLIC ${LINK_LIBS})
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

  if(CUDAWRAPPERS_BUILD_NVRTC)
    target_include_directories(
      test_nvrtc PRIVATE "${CMAKE_SOURCE_DIR}/include"
                         "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
    )

    target_embed_source(test_nvrtc kernels/vector_add_kernel.cu)
    target_embed_source(test_nvrtc kernels/single_include_kernel.cu)
    target_embed_source(test_nvrtc kernels/recursive_include_kernel.cu)
  endif()

  if(CUDAWRAPPERS_BUILD_CUFFT)
    target_link_libraries(test_cufft PUBLIC cudawrappers::cufft)
  endif()

  if(CUDAWRAPPERS_BUILD_NVTX)
    target_link_libraries(test_nvtx PUBLIC cudawrappers::nvtx)
  endif()
endif()
