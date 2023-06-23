add_executable(convolution convolution.cpp)
target_link_libraries(
  convolution PRIVATE cudawrappers::cu cudawrappers::nvrtc cudawrappers::cufft
)

add_test(NAME convolution COMMAND convolution)
