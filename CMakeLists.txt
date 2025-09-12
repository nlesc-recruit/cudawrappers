cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

# Find out if cudawrappers if used as a dependency. Build tests by default only
# if not used as a dependency.
if(NOT DEFINED PROJECT_NAME AND BUILD_TESTING)
  set(CUDAWRAPPERS_TESTING_DEFAULT True)
else()
  set(CUDAWRAPPERS_TESTING_DEFAULT False)
endif()

project(
  cudawrappers
  DESCRIPTION "C++ Wrappers for the CUDA Driver API"
  VERSION 0.9.0
  HOMEPAGE_URL https://github.com/nlesc-recruit/cudawrappers
  LANGUAGES CXX
)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)
set(CMAKE_BUILD_TYPE Release)
option(BUILD_SHARED_LIBS "Create shared libraries" True)
option(CUDAWRAPPERS_BUILD_TESTING "Build cudawrappers tests"
       ${CUDAWRAPPERS_TESTING_DEFAULT}
)

if(NOT DEFINED CUDAWRAPPERS_BACKEND)
  set(CUDAWRAPPERS_BACKEND "CUDA")
endif()
set(CUDAWRAPPERS_BACKEND
    ${CUDAWRAPPERS_BACKEND}
    CACHE STRING "Backend API to use in cudawrappers"
)
set_property(CACHE CUDAWRAPPERS_BACKEND PROPERTY STRINGS "CUDA" "HIP")

if(${CUDAWRAPPERS_BACKEND} STREQUAL "CUDA")
  set(CUDAWRAPPERS_BACKEND_CUDA True)
elseif(${CUDAWRAPPERS_BACKEND} STREQUAL "HIP")
  set(CUDAWRAPPERS_BACKEND_HIP True)
else()
  message(
    FATAL_ERROR
      "Invalid value for CUDAWRAPPERS_BACKEND: ${CUDAWRAPPERS_BACKEND}"
  )
endif()

# Set the components to build. cu should always be built, the rest are optional.
set(CUDAWRAPPERS_ALL_COMPONENTS cu cufft nvml nvrtc nvtx)
set(CUDAWRAPPERS_MANDATORY_COMPONENTS cu)
set(CUDAWRAPPERS_DEFAULT_COMPONENTS cu nvrtc)

if(NOT DEFINED CUDAWRAPPERS_COMPONENTS)
  set(CUDAWRAPPERS_COMPONENTS
      "${CUDAWRAPPERS_DEFAULT_COMPONENTS}"
      CACHE STRING "List of components to build"
  )
endif()

# If 'all' is specified, set to all known components
list(FIND CUDAWRAPPERS_COMPONENTS "all" _found)
if(_found GREATER -1)
  set(CUDAWRAPPERS_COMPONENTS ${CUDAWRAPPERS_ALL_COMPONENTS})
endif()

# If HIP backend is used, remove NVML
if(CUDAWRAPPERS_BACKEND_HIP)
  list(FIND CUDAWRAPPERS_COMPONENTS "nvml" _found)
  if(_found GREATER -1)
    message(WARNING "HIP backend is used, removing nvml component.")
    list(REMOVE_ITEM CUDAWRAPPERS_COMPONENTS nvml)
  endif()
endif()

# Make sure the mandatory components are always built
foreach(mandatory ${CUDAWRAPPERS_MANDATORY_COMPONENTS})
  list(FIND CUDAWRAPPERS_COMPONENTS "${mandatory}" _found)
  if(_found EQUAL -1)
    list(APPEND CUDAWRAPPERS_COMPONENTS "${mandatory}")
  endif()
endforeach()

# Add helper variable for each component to indicate if it should be built
foreach(component ${CUDAWRAPPERS_ALL_COMPONENTS})
  string(TOUPPER "${component}" COMPONENT_UPPER)
  list(FIND CUDAWRAPPERS_COMPONENTS "${component}" _found_index)
  if(_found_index GREATER -1)
    set(CUDAWRAPPERS_BUILD_${COMPONENT_UPPER} ON)
  else()
    set(CUDAWRAPPERS_BUILD_${COMPONENT_UPPER} OFF)
  endif()
endforeach()

if(${CUDAWRAPPERS_BACKEND_HIP})
  enable_language(HIP)
  set(CMAKE_HIP_COMPILER_ENV_VAR "HIPCXX")
endif()

# Enable Clang compiler warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(
    -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded
  )
endif()

# Export compile commands for use by clang-tidy and cppcheck
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

# Include CMake helper functions
include(cmake/cudawrappers-helper.cmake)

# You can use cudawrappers by installing it, or by including it in your project
set(CUDAWRAPPERS_INSTALLED ${CMAKE_PROJECT_NAME} STREQUAL cudawrappers)

# Check if the dependencies are satisfied This may update the
# CUDAWRAPPERS_COMPONENTS variable if some components are not available (e.g.
# hipfft).
include(cmake/cudawrappers-dependencies.cmake)

# NVTX 2 is deprecated since CUDA 12.8, so we use NVTX 3 instead
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
  set(CUDAWRAPPERS_USE_NVTX3 True)
endif()

# Update the final list of components in the cache
set(CUDAWRAPPERS_COMPONENTS
    ${CUDAWRAPPERS_COMPONENTS}
    CACHE STRING "List of components to build" FORCE
)

# Configure tests
if(CUDAWRAPPERS_BUILD_TESTING)
  include(CTest)
  enable_testing()

  # Set up coverage collection
  add_compile_options(--coverage -g -O0)
  add_link_options("--coverage")
  add_subdirectory(tests)
endif()

# Create the component libraries
include(cmake/cudawrappers-targets.cmake)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  # Install project cmake targets
  include(CMakePackageConfigHelpers)

  # Install cmake config file
  install(
    EXPORT ${PROJECT_NAME}-targets
    NAMESPACE ${PROJECT_NAME}::
    FILE ${PROJECT_NAME}-exported.cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

  # Install version file
  write_basic_package_version_file(
    ${PROJECT_NAME}-config-version.cmake
    VERSION ${cudawrappers_VERSION}
    COMPATIBILITY AnyNewerVersion
  )

  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    NO_SET_AND_CHECK_MACRO NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )

  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/cudawrappers-config.cmake
          ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cudawrappers-helper.cmake
          ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cudawrappers-dependencies.cmake
          ${CMAKE_CURRENT_SOURCE_DIR}/cmake/inline-common.cmake
          ${CMAKE_CURRENT_SOURCE_DIR}/cmake/inline-local-includes.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

  # --- auto-ignore build directory
  if(NOT EXISTS ${PROJECT_BINARY_DIR}/.gitignore)
    file(WRITE ${PROJECT_BINARY_DIR}/.gitignore "*")
  endif()

endif()
