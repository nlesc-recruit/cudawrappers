cmake_minimum_required(VERSION 3.26)
project(PrintNvrtcProps LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

# Helper function to print all properties of a target
function(print_target_properties tgt)
  if(NOT TARGET ${tgt})
    message(WARNING "Target ${tgt} not found")
    return()
  endif()
  message(STATUS "---- Properties for ${tgt} ----")
  get_all_properties(_props)
  foreach(prop ${_props})
    get_target_property(_val ${tgt} ${prop})
    if(NOT _val)
      continue()
    endif()
    message(STATUS "${prop} = ${_val}")
  endforeach()
endfunction()

# Helper: get full property list (CMake doesn't expose this directly)
function(get_all_properties out_var)
  set(props
      TYPE
      IMPORTED
      IMPORTED_LOCATION
      IMPORTED_LOCATION_RELEASE
      IMPORTED_LOCATION_DEBUG
      INTERFACE_INCLUDE_DIRECTORIES
      INTERFACE_LINK_LIBRARIES
      INTERFACE_COMPILE_DEFINITIONS
      INTERFACE_COMPILE_OPTIONS
      IMPORTED_IMPLIB
      IMPORTED_NO_SONAME
      IMPORTED_SONAME
      IMPORTED_LINK_INTERFACE_LANGUAGES
      IMPORTED_LINK_DEPENDENT_LIBRARIES
      IMPORTED_LINK_INTERFACE_LIBRARIES
  )
  set(${out_var}
      "${props}"
      PARENT_SCOPE
  )
endfunction()

print_target_properties(CUDA::nvrtc)
print_target_properties(CUDA::nvrtc_builtins)

add_executable(minimal minimal.cpp)
target_link_libraries(minimal PRIVATE CUDA::nvrtc) # CUDA::nvrtc_builtins)

# embed path to nvrtc-builtins.so
get_target_property(_nvrtc CUDA::nvrtc IMPORTED_LOCATION)
get_filename_component(_nvrtc_dir "${_nvrtc}" DIRECTORY)
set_target_properties(
  minimal PROPERTIES BUILD_RPATH "${_nvrtc_dir}" INSTALL_RPATH "${_nvrtc_dir}"
)
