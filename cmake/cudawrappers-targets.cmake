# Get the include directory in the source tree
get_filename_component(
  CUDAWRAPPERS_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../include ABSOLUTE
)

# Define all the individual components that cudawrappers provides
if(${CUDAWRAPPERS_BACKEND_HIP})
  list(APPEND CUDAWRAPPERS_COMPONENTS macros)
  set(LINK_macros hip::host)
  set(LINK_cu hip::host)
  if(CUDAWRAPPERS_BUILD_CUFFT)
    set(LINK_cufft hip::host hip::hipfft)
  endif()
  if(CUDAWRAPPERS_BUILD_NVML)
    set(LINK_nvml hip::host)
  endif()
  if(CUDAWRAPPERS_BUILD_NVRTC)
    set(LINK_nvrtc hip::host)
    if(CUDAWRAPPERS_LINK_HIPRTC)
      list(APPEND LINK_nvrtc hiprtc)
    endif()
  endif()
  if(CUDAWRAPPERS_BUILD_NVTX)
    set(LINK_nvtx hip::host)
  endif()
else()
  set(LINK_cu CUDA::cuda_driver)
  if(CUDAWRAPPERS_BUILD_CUFFT)
    set(LINK_cufft CUDA::cuda_driver CUDA::cufft)
  endif()
  if(CUDAWRAPPERS_BUILD_NVML)
    set(LINK_nvml CUDA::cuda_driver CUDA::nvml)
  endif()
  if(CUDAWRAPPERS_BUILD_NVRTC)
    set(LINK_nvrtc CUDA::cuda_driver CUDA::nvrtc ${CMAKE_DL_LIBS})
  endif()
  # NVTX 3 is header only, so don't link nvToolsExt
  if(CUDAWRAPPERS_BUILD_NVTX AND NOT CUDAWRAPPERS_USE_NVTX3)
    set(LINK_nvtx CUDA::nvToolsExt)
  endif()
endif()

foreach(component ${CUDAWRAPPERS_COMPONENTS})
  add_library(${component} INTERFACE)
  # cudawrappers exposes targets like, cudawrappers::cu, so an alias is created
  add_library(${PROJECT_NAME}::${component} ALIAS ${component})
  target_link_libraries(${component} INTERFACE ${LINK_${component}})
  if(${CUDAWRAPPERS_INSTALLED})
    target_include_directories(
      ${component} INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    set_target_properties(
      ${component}
      PROPERTIES PUBLIC_HEADER
                 ${CUDAWRAPPERS_INCLUDE_DIR}/cudawrappers/${component}.hpp
    )
  else()
    target_include_directories(
      ${component} INTERFACE ${CUDAWRAPPERS_INCLUDE_DIR}
    )
  endif()
endforeach()

if(CUDAWRAPPERS_BUILD_NVTX AND CUDAWRAPPERS_USE_NVTX3)
  target_compile_definitions(nvtx INTERFACE USE_NVTX3)
endif()

# Install the header files and export the configuration
install(
  TARGETS ${CUDAWRAPPERS_COMPONENTS}
  EXPORT ${PROJECT_NAME}-targets
  COMPONENT ${PROJECT_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)
