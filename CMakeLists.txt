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
  VERSION 0.3.0
  HOMEPAGE_URL https://github.com/nlesc-recruit/cudawrappers
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)
set(CMAKE_BUILD_TYPE Release)
option(BUILD_SHARED_LIBS "Create shared libraries" True)
option(CUDAWRAPPERS_BUILD_TESTING "Build cudawrappers tests"
       ${CUDAWRAPPERS_TESTING_DEFAULT}
)
option(CUDAWRAPPERS_BUILD_EXAMPLES "Build cudawrappers examples" False)

# Enable Clang compiler warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(
    -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded
  )
endif()

# Export compile commands for use by clang-tidy and cppcheck
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

set(CUDA_MIN_VERSION 10.0)
find_package(CUDAToolkit ${CUDA_MIN_VERSION} REQUIRED)

# Define components
set(COMPONENTS cu nvrtc)
set(LINK_cu CUDA::cuda_driver)
set(LINK_nvrtc CUDA::cuda_driver CUDA::nvrtc)

include(GNUInstallDirs)
foreach(component ${COMPONENTS})
  # Create library
  add_library(${component})
  # Add a pretend namespace for use cases where the project in not installed
  add_library(${PROJECT_NAME}::${component} ALIAS ${component})
  # Add sources
  target_sources(${component} PRIVATE src/${component}.cpp)
  # Add headers
  set_target_properties(
    ${component} PROPERTIES PUBLIC_HEADER include/${component}.hpp
  )
  # Add the project name as a prefix to the output libraries
  set_target_properties(${component} PROPERTIES PREFIX ${PROJECT_NAME}-)
  # Add includes
  target_include_directories(
    ${component}
    PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
  )
  # Add links
  target_link_libraries(${component} PUBLIC ${LINK_${component}})
  # Install libraries and headers
  install(
    TARGETS ${component}
    EXPORT ${component}-config # export component cmake targets
    COMPONENT ${component}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
  )
  # Install component cmake targets
  install(
    EXPORT ${component}-config
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
endforeach()

# Install project cmake targets
include(CMakePackageConfigHelpers)

configure_file(
  cmake/${PROJECT_NAME}-config.cmake.in ${PROJECT_NAME}-config.cmake @ONLY
)

write_basic_package_version_file(
  ${PROJECT_NAME}-config-version.cmake
  VERSION ${cudawrappers_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# Including linters rules
include(cmake/linter-tools.cmake)

# Configure examples
if(CUDAWRAPPERS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
# Configure tests
if(CUDAWRAPPERS_BUILD_TESTING)
  include(CTest)

  # Set up coverage collection
  add_compile_options(--coverage -g -O0)
  add_link_options("--coverage")
  add_subdirectory(tests)
endif()

# --- auto-ignore build directory
if(NOT EXISTS ${PROJECT_BINARY_DIR}/.gitignore)
  file(WRITE ${PROJECT_BINARY_DIR}/.gitignore "*")
endif()
