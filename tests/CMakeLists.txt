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

# Compile Catch2
add_library(tests)
target_sources(tests PRIVATE main.cpp)
target_link_libraries(tests PUBLIC Catch2::Catch2)
set(LINK_cu cudawrappers::cu)
set(LINK_cufft cudawrappers::cu cudawrappers::cufft)
set(LINK_nvrtc cudawrappers::nvrtc)

# Add library tests
set(COMPONENTS cu nvrtc cufft)
foreach(component ${COMPONENTS})
  add_executable(test_${component} test_${component}.cpp)
  target_sources(test_${component} PRIVATE test_${component}.cpp)
  target_link_libraries(test_${component} PUBLIC ${LINK_${component}} tests)
  catch_discover_tests(test_${component})
  foreach(test_name ${test_${component}_TESTS})
    add_test(NAME test_${component}::${test_name} COMMAND test_${component}
                                                          ${test_name}
    )
  endforeach()
endforeach()
