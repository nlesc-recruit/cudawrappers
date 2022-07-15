include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.9
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(Catch)

# Compile Catch2
add_library(tests)
target_sources(tests PRIVATE main.cpp)
target_link_libraries(tests PUBLIC Catch2::Catch2)

# Add library tests
add_executable(test_cu)
target_sources(test_cu PRIVATE test_cu.cpp)
target_link_libraries(test_cu PUBLIC cudawrappers::cu tests)
catch_discover_tests(test_cu)
foreach(test_name ${test_cu_TESTS})
  add_test(NAME test_cu::${test_name} COMMAND test_cu ${test_name})
endforeach()
