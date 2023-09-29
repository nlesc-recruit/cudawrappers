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

# Make it possible to embed a source file in a library, and link it to a target.
# E.g. to link <kernel.cu> into target <example_program>, use
# target_embed_source(example_program, kernel.cu). This will expose symbols
# _binary_kernel_cu_start and _binary_ls_kernel_cu_end.
function(target_embed_source target input_file)
  # Strip the path and extension from input_file
  get_filename_component(NAME ${input_file} NAME_WLE)
  # Link the input_file into an object file
  add_custom_command(
    OUTPUT ${NAME}.o
    COMMAND ld ARGS -r -b binary -o ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.o
            ${input_file}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${input_file}
  )
  if(NOT TARGET ${NAME})
    # Create a proper static library for the .o file
    add_library(${NAME} STATIC ${NAME}.o)
    set_target_properties(${NAME} PROPERTIES LINKER_LANGUAGE CXX)
  endif()
  # Link the static library to the target
  target_link_libraries(${target} PRIVATE ${NAME})
endfunction()
