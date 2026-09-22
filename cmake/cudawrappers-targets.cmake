include(FetchContent)

# Get the include directory in the source tree
get_filename_component(
  CUDAWRAPPERS_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../include ABSOLUTE
)

# Define all the individual components that cudawrappers provides
if(CUDAWRAPPERS_BACKEND_ALL)
  # --- Dual-backend mode: header-only cu library with runtime dispatch ---

  list(APPEND CUDAWRAPPERS_COMPONENTS macros)
  set(LINK_macros_hip hip::host)

  # cu: header-only, needs both CUDA and HIP headers for compilation
  add_library(cu INTERFACE)
  add_library(${PROJECT_NAME}::cu ALIAS cu)
  target_include_directories(
    cu INTERFACE $<BUILD_INTERFACE:${CUDAWRAPPERS_INCLUDE_DIR}>
                 $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  target_link_libraries(
    cu INTERFACE CUDA::cuda_driver hip::host ${CMAKE_DL_LIBS}
  )
  set_target_properties(
    cu PROPERTIES PUBLIC_HEADER ${CUDAWRAPPERS_INCLUDE_DIR}/cudawrappers/cu.hpp
  )

  # HIP-only: macros target
  add_library(macros INTERFACE)
  add_library(${PROJECT_NAME}::macros ALIAS macros)
  target_link_libraries(macros INTERFACE hip::host)
  target_include_directories(
    macros INTERFACE $<BUILD_INTERFACE:${CUDAWRAPPERS_INCLUDE_DIR}>
                     $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  # CUDA-only components (nvml)
  if(CUDAWRAPPERS_BUILD_NVML)
    set(LINK_nvml_cuda CUDA::cuda_driver CUDA::nvml)
    add_library(nvml_cuda INTERFACE)
    add_library(${PROJECT_NAME}::nvml_cuda ALIAS nvml_cuda)
    target_link_libraries(nvml_cuda INTERFACE ${LINK_nvml_cuda})
    target_include_directories(
      nvml_cuda INTERFACE $<BUILD_INTERFACE:${CUDAWRAPPERS_INCLUDE_DIR}>
                          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
  endif()

  # NVRTC targets (header-only, still needed)
  if(CUDAWRAPPERS_BUILD_NVRTC)
    set(LINK_nvrtc_cuda CUDA::cuda_driver CUDA::nvrtc ${CMAKE_DL_LIBS})
    add_library(nvrtc_cuda INTERFACE)
    add_library(${PROJECT_NAME}::nvrtc_cuda ALIAS nvrtc_cuda)
    target_link_libraries(nvrtc_cuda INTERFACE ${LINK_nvrtc_cuda})
    target_include_directories(
      nvrtc_cuda INTERFACE $<BUILD_INTERFACE:${CUDAWRAPPERS_INCLUDE_DIR}>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    set(LINK_nvrtc_hip hip::host)
    if(CUDAWRAPPERS_LINK_HIPRTC)
      list(APPEND LINK_nvrtc_hip hiprtc::hiprtc)
    endif()
    add_library(nvrtc_hip INTERFACE)
    add_library(${PROJECT_NAME}::nvrtc_hip ALIAS nvrtc_hip)
    target_link_libraries(nvrtc_hip INTERFACE ${LINK_nvrtc_hip})
    target_include_directories(
      nvrtc_hip INTERFACE $<BUILD_INTERFACE:${CUDAWRAPPERS_INCLUDE_DIR}>
                          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    # Combined convenience target matching the single-backend component name
    add_library(nvrtc INTERFACE)
    add_library(${PROJECT_NAME}::nvrtc ALIAS nvrtc)
    target_link_libraries(nvrtc INTERFACE nvrtc_cuda nvrtc_hip)
  endif()

  # cuFFT targets (header-only)
  if(CUDAWRAPPERS_BUILD_CUFFT)
    add_library(cufft_cuda INTERFACE)
    add_library(${PROJECT_NAME}::cufft_cuda ALIAS cufft_cuda)
    target_link_libraries(cufft_cuda INTERFACE CUDA::cuda_driver CUDA::cufft)
    target_include_directories(
      cufft_cuda INTERFACE $<BUILD_INTERFACE:${CUDAWRAPPERS_INCLUDE_DIR}>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    add_library(cufft_hip INTERFACE)
    add_library(${PROJECT_NAME}::cufft_hip ALIAS cufft_hip)
    target_link_libraries(cufft_hip INTERFACE hip::host hip::hipfft)
    target_include_directories(
      cufft_hip INTERFACE $<BUILD_INTERFACE:${CUDAWRAPPERS_INCLUDE_DIR}>
                          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
  endif()

  # NVTX targets (header-only)
  if(CUDAWRAPPERS_BUILD_NVTX)
    add_library(nvtx_cuda INTERFACE)
    add_library(${PROJECT_NAME}::nvtx_cuda ALIAS nvtx_cuda)
    if(NOT CUDAWRAPPERS_USE_NVTX3)
      target_link_libraries(nvtx_cuda INTERFACE CUDA::nvToolsExt)
    endif()
    target_include_directories(
      nvtx_cuda INTERFACE $<BUILD_INTERFACE:${CUDAWRAPPERS_INCLUDE_DIR}>
                          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    add_library(nvtx_hip INTERFACE)
    add_library(${PROJECT_NAME}::nvtx_hip ALIAS nvtx_hip)
    target_link_libraries(nvtx_hip INTERFACE hip::host)
    target_include_directories(
      nvtx_hip INTERFACE $<BUILD_INTERFACE:${CUDAWRAPPERS_INCLUDE_DIR}>
                         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
  endif()

else()
  # --- Single-backend mode: header-only ---

  if(${CUDAWRAPPERS_BACKEND_HIP})
    list(APPEND CUDAWRAPPERS_COMPONENTS macros)
    set(LINK_macros hip::host)
    set(LINK_cu hip::host ${CMAKE_DL_LIBS})
    if(CUDAWRAPPERS_BUILD_CUFFT)
      set(LINK_cufft hip::host hip::hipfft)
    endif()
    if(CUDAWRAPPERS_BUILD_NVML)
      set(LINK_nvml hip::host)
    endif()
    if(CUDAWRAPPERS_BUILD_NVRTC)
      set(LINK_nvrtc hip::host)
      if(CUDAWRAPPERS_LINK_HIPRTC)
        list(APPEND LINK_nvrtc hiprtc::hiprtc)
      endif()
    endif()
    if(CUDAWRAPPERS_BUILD_NVTX)
      set(LINK_nvtx hip::host)
    endif()

    foreach(component ${CUDAWRAPPERS_COMPONENTS})
      add_library(${component} INTERFACE)
      add_library(${PROJECT_NAME}::${component} ALIAS ${component})
      target_link_libraries(${component} INTERFACE ${LINK_${component}})
      target_include_directories(
        ${component} INTERFACE $<BUILD_INTERFACE:${CUDAWRAPPERS_INCLUDE_DIR}>
                               $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
      )
      set_target_properties(
        ${component}
        PROPERTIES PUBLIC_HEADER
                   ${CUDAWRAPPERS_INCLUDE_DIR}/cudawrappers/${component}.hpp
      )
    endforeach()
  else()
    # CUDA-only single-backend: header-only
    set(LINK_cu CUDA::cuda_driver ${CMAKE_DL_LIBS})

    if(CUDAWRAPPERS_BUILD_CUFFT)
      set(LINK_cufft CUDA::cuda_driver CUDA::cufft)
    endif()
    if(CUDAWRAPPERS_BUILD_NVML)
      set(LINK_nvml CUDA::cuda_driver CUDA::nvml)
    endif()
    if(CUDAWRAPPERS_BUILD_NVRTC)
      set(LINK_nvrtc CUDA::cuda_driver CUDA::nvrtc ${CMAKE_DL_LIBS})
    endif()
    if(CUDAWRAPPERS_BUILD_NVTX AND NOT CUDAWRAPPERS_USE_NVTX3)
      set(LINK_nvtx CUDA::nvToolsExt)
    endif()

    foreach(component ${CUDAWRAPPERS_COMPONENTS})
      add_library(${component} INTERFACE)
      add_library(${PROJECT_NAME}::${component} ALIAS ${component})
      target_link_libraries(${component} INTERFACE ${LINK_${component}})
      target_include_directories(
        ${component} INTERFACE $<BUILD_INTERFACE:${CUDAWRAPPERS_INCLUDE_DIR}>
                               $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
      )
      set_target_properties(
        ${component}
        PROPERTIES PUBLIC_HEADER
                   ${CUDAWRAPPERS_INCLUDE_DIR}/cudawrappers/${component}.hpp
      )
    endforeach()
  endif()
endif()

if(CUDAWRAPPERS_BUILD_CUFFT)
  set(MAGIC_ENUM_OPT_INSTALL TRUE)
  FetchContent_Declare(
    magic_enum
    GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
    GIT_TAG v0.9.7
  )

  FetchContent_MakeAvailable(magic_enum)
  if(CUDAWRAPPERS_BACKEND_ALL)
    target_link_libraries(cufft_cuda INTERFACE magic_enum)
    target_link_libraries(cufft_hip INTERFACE magic_enum)
  else()
    target_link_libraries(cufft INTERFACE magic_enum)
  endif()
endif()

# Collect all created targets for installation
set(CUDAWRAPPERS_INSTALL_TARGETS cu)
if(CUDAWRAPPERS_BACKEND_ALL)
  list(APPEND CUDAWRAPPERS_INSTALL_TARGETS macros)
  if(CUDAWRAPPERS_BUILD_NVML)
    list(APPEND CUDAWRAPPERS_INSTALL_TARGETS nvml_cuda)
  endif()
  if(CUDAWRAPPERS_BUILD_NVRTC)
    list(APPEND CUDAWRAPPERS_INSTALL_TARGETS nvrtc_cuda nvrtc_hip)
  endif()
  if(CUDAWRAPPERS_BUILD_CUFFT)
    list(APPEND CUDAWRAPPERS_INSTALL_TARGETS cufft_cuda cufft_hip)
  endif()
  if(CUDAWRAPPERS_BUILD_NVTX)
    list(APPEND CUDAWRAPPERS_INSTALL_TARGETS nvtx_cuda nvtx_hip)
  endif()
else()
  foreach(component ${CUDAWRAPPERS_COMPONENTS})
    if(NOT "${component}" STREQUAL "cu")
      list(APPEND CUDAWRAPPERS_INSTALL_TARGETS ${component})
    endif()
  endforeach()
endif()

# Install the header files and export the configuration
install(
  TARGETS ${CUDAWRAPPERS_INSTALL_TARGETS}
  EXPORT ${PROJECT_NAME}-targets
  COMPONENT ${PROJECT_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)
