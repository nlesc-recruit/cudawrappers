# Return a list of absolute file names for all the local includes of the
# input_file.  Only files in the root directory will be considered.
function(get_local_includes input_file root_dir)
  file(READ ${input_file} input_file_contents)
  set(include_regex "(^|\r?\n)(#include[ \t]*\"([^\"]+)\")")
  string(REGEX MATCHALL ${include_regex} includes ${input_file_contents})
  set(include_files "")
  foreach(include ${includes})
    # Get the name of the file to include, e.g. 'helper.h'
    string(REGEX REPLACE ${include_regex} "\\3" include_name ${include})
    # Get the complete line of the include, e.g.  '#include <helper.h>'
    file(GLOB_RECURSE INCLUDE_PATHS "${root_dir}/*/${include_name}")
    if(NOT INCLUDE_PATHS STREQUAL "")
      list(SORT INCLUDE_PATHS ORDER DESCENDING)
      list(GET INCLUDE_PATHS 0 include_PATH)
      get_local_includes(${include_PATH} ${root_dir} include_files)
      list(APPEND include_files ${include_PATH})
    endif()
  endforeach()
endfunction()

# Make it possible to embed a source file in a library, and link it to a target.
# E.g. to link <kernel.cu> into target <example_program>, use
# target_embed_source(example_program, kernel.cu). This will expose symbols
# _binary_kernel_cu_start and _binary_kernel_cu_end.
function(target_embed_source target input_file)
  include(CMakeDetermineSystem)
  # Strip the path and extension from input_file
  get_filename_component(NAME ${input_file} NAME_WLE)
  # Get absolute path for input file
  get_filename_component(input_file_absolute ${input_file} ABSOLUTE)
  # Make a copy of the input file in the binary dir with inlined header files
  string(REPLACE "${PROJECT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}"
                 input_file_inlined ${input_file_absolute}
  )
  # Get a list of all local includes so that they can be added as dependencies
  set(include_files "")
  get_local_includes(${input_file} ${PROJECT_SOURCE_DIR} include_files)
  # Create a copy of the input file with all local headers inlined
  add_custom_command(
    OUTPUT ${input_file_inlined}
    COMMAND
      ${CMAKE_COMMAND} -Dinput_file=${input_file_absolute}
      -Doutput_file=${input_file_inlined} -Droot_dir=${PROJECT_SOURCE_DIR} -P
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cudawrappers-inline-local-includes.cmake"
    DEPENDS "${input_file_absolute};${include_files}"
    COMMENT "Inlining all includes of ${input_file}"
  )
  # Link the input_file into an object file
  string(REPLACE "${PROJECT_SOURCE_DIR}/" "" input_file_inlined_relative
                 ${input_file_absolute}
  )
  add_custom_command(
    OUTPUT ${NAME}.o
    COMMAND
      ld ARGS -r -b binary -A ${CMAKE_SYSTEM_PROCESSOR} -o
      "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.o" ${input_file_inlined_relative}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS ${input_file_inlined}
    COMMENT "Creating object file for ${input_file}"
  )
  if(NOT TARGET ${NAME})
    # Create a proper static library for the .o file
    add_library(${NAME} STATIC ${NAME}.o)
    set_target_properties(${NAME} PROPERTIES LINKER_LANGUAGE CXX)
  endif()
  # Link the static library to the target
  target_link_libraries(${target} PRIVATE ${NAME})
endfunction()
