# You can use cudawrappers by installing it, or by including it in your project
set(CUDAWRAPPERS_INSTALLED ${CMAKE_PROJECT_NAME} STREQUAL cudawrappers)

# cudawrappers requires the CUDA Toolkit.If you include cudawrappers in your
# project, you need to include the toolkit yourself
set(CUDA_MIN_VERSION 10.0)
if(${CUDAWRAPPERS_INSTALLED})
  find_package(CUDAToolkit ${CUDA_MIN_VERSION} REQUIRED)
else()
  if($CUDAToolkit_FOUND)
    if(${CUDAToolkit_VERSION_MAJOR} LESS ${CUDA_MIN_VERSION})
      message(FATAL_ERROR "Insufficient CUDA version: " ${CUDAToolkit_VERSION}
                          " < " ${CUDA_MIN_VERSION}
      )
    endif()
  else()
    message(
      FATAL_ERROR
        "CUDAToolkit not found... use find_package(CUDAToolkit REQUIRED)"
    )
  endif()
endif()

# Get the include directory in the source tree
get_filename_component(
  CUDAWRAPPERS_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../include ABSOLUTE
)

# Define all the individual components that cudawrappers provides
set(CUDAWRAPPERS_COMPONENTS cu cufft nvrtc nvtx)
set(LINK_cu CUDA::cuda_driver)
set(LINK_cufft CUDA::cuda_driver CUDA::cufft)
set(LINK_nvrtc CUDA::cuda_driver CUDA::nvrtc)
set(LINK_nvtx CUDA::nvToolsExt)

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

# Install the header files and export the configuration
# if(${CUDAWRAPPERS_INSTALLED})
install(
  TARGETS ${CUDAWRAPPERS_COMPONENTS}
  EXPORT ${PROJECT_NAME}-config
  COMPONENT ${PROJECT_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)
# endif()
