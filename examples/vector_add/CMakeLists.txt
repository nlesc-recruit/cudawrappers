# Make it possible to #include cuda source code
function(include_cuda_code target input_file)
  # Save file containing cuda code as a C++ raw string literal
  file(READ ${input_file} content)
  set(delim "for_c++_include")
  set(content "R\"${delim}(\n${content})${delim}\"")
  set(output_file "${CMAKE_CURRENT_BINARY_DIR}/${input_file}")
  file(WRITE ${output_file} "${content}")
  # Add save path to the include directories
  get_filename_component(output_subdir ${input_file} DIRECTORY)
  target_include_directories(
    ${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/${output_subdir}"
  )
endfunction(include_cuda_code)

add_executable(vector_add vector_add.cpp)
target_link_libraries(vector_add PRIVATE cudawrappers::cu CUDA::cudart cudawrappers::nvrtc)
include_cuda_code(vector_add kernels/vector_add_kernel.cu)

add_test(NAME vector_add COMMAND vector_add)
