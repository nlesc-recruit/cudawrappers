# Copies the kernel function to the tests folder
file(COPY vector_add_kernel.cu DESTINATION ..)
add_executable(vector_add vector_add.cpp)
target_link_libraries(vector_add PRIVATE cudawrappers CUDA::cuda_driver CUDA::nvrtc)