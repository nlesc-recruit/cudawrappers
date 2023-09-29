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
  VERSION 0.5.0
  HOMEPAGE_URL https://github.com/nlesc-recruit/cudawrappers
  LANGUAGES CXX
)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)
set(CMAKE_BUILD_TYPE Release)
option(BUILD_SHARED_LIBS "Create shared libraries" True)
option(CUDAWRAPPERS_BUILD_TESTING "Build cudawrappers tests"
       ${CUDAWRAPPERS_TESTING_DEFAULT}
)

# Enable Clang compiler warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(
    -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded
  )
endif()

# Export compile commands for use by clang-tidy and cppcheck
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

# Include CMake helper functions
include(cmake/helper.cmake)

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
include(cmake/cudawrappers-config.cmake)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  # Install project cmake targets
  include(CMakePackageConfigHelpers)

  # Install cmake config file
  install(
    EXPORT ${PROJECT_NAME}-config
    NAMESPACE ${PROJECT_NAME}::
    EXPORT_LINK_INTERFACE_LIBRARIES
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

  # Install version file
  write_basic_package_version_file(
    ${PROJECT_NAME}-config-version.cmake
    VERSION ${cudawrappers_VERSION}
    COMPATIBILITY AnyNewerVersion
  )

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

  # --- auto-ignore build directory
  if(NOT EXISTS ${PROJECT_BINARY_DIR}/.gitignore)
    file(WRITE ${PROJECT_BINARY_DIR}/.gitignore "*")
  endif()

endif()
