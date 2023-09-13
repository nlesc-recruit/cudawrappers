include(FetchContent)

# We use Catch2 v2.x because it is the last version that supports C++11, see
# https://github.com/catchorg/Catch2.
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.9
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(Catch)

add_executable(test_cu test_cu.cpp)
target_link_libraries(test_cu PUBLIC Catch2::Catch2 cudawrappers::cu)
catch_discover_tests(test_cu)

add_executable(test_nvrtc test_nvrtc.cpp)
target_link_libraries(
  test_nvrtc PUBLIC Catch2::Catch2 cudawrappers::cu cudawrappers::nvrtc
)
catch_discover_tests(test_nvrtc)

add_executable(test_cufft test_cufft.cpp)
target_link_libraries(
  test_cufft PUBLIC Catch2::Catch2 cudawrappers::cu cudawrappers::cufft
)
catch_discover_tests(test_cufft)

set(COMPONENTS cu nvrtc cufft)
foreach(component ${COMPONENTS})
  target_include_directories(
    test_${component} PRIVATE "${CMAKE_SOURCE_DIR}/include"
  )
endforeach()
