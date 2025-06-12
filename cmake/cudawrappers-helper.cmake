# =============================================================================
# cmake-format: off
# This module enables embedding kernel or source files (e.g. .cu) into targets
# by linking their binary representation as static libraries. It also inlines
# any local includes (#include "...") in the source file.
#
# Functions:
#   - get_local_includes: Recursively collect local includes.
#   - inline_local_includes: Prepend headers and remove #include "..."
#   - target_embed_source: Embed a source file as binary and link to a target.
#
# cmake-format: on
# =============================================================================

# cmake-format: off
# Return a list of absolute file names for all the local includes of the
# input_file. Only files in the root_dir will be considered.
#
# Parameters:
#   input_file: The file to scan for includes
#   root_dir:   The root directory in which to search for include files
#   out_var:    Output variable name to receive the list of include files (absolute paths)
# cmake-format: on
# get_local_includes
function(get_local_includes input_file root_dir out_var)
  file(READ "${input_file}" input_file_contents)
  set(include_regex "(^|\r?\n)(#include[ \t]*\"([^\"]+)\")")
  string(REGEX MATCHALL "${include_regex}" includes "${input_file_contents}")

  set(include_files_local "")

  foreach(include ${includes})
    # Extract the filename from the include directive
    string(REGEX REPLACE "${include_regex}" "\\3" include_name "${include}")
    file(GLOB_RECURSE include_paths "${root_dir}/*/${include_name}")
    if(include_paths)
      list(SORT include_paths ORDER DESCENDING)
      list(GET include_paths 0 include_path)
      get_local_includes("${include_path}" "${root_dir}" recursive_includes)
      list(APPEND include_files_local ${recursive_includes} "${include_path}")
    else()
      message(
        WARNING "Could not find include: ${include_name} in ${input_file}"
      )
    endif()
  endforeach()

  list(REMOVE_DUPLICATES include_files_local)
  set(${out_var}
      "${include_files_local}"
      PARENT_SCOPE
  )
endfunction()

# cmake-format: off
# Create a new file with inlined (prepended) headers and local #include "..."
# lines removed from the input file.
#
# Parameters:
#   input_file:     Absolute path to main source file
#   output_file:    File to write the resulting inlined source
#   include_files:  List of header files (absolute paths) to inline
# cmake-format: on
# inline_local_includes
function(inline_local_includes input_file output_file include_files)
  file(READ "${input_file}" input_contents)
  set(include_regex "(^|\r?\n)(#include[ \t]*\"([^\"]+)\")")
  string(REGEX REPLACE "${include_regex}" "" input_contents "${input_contents}")

  set(output_content "")
  foreach(include_file ${include_files})
    file(READ "${include_file}" header_contents)
    set(output_content "${output_content}\n${header_contents}")
  endforeach()

  set(output_content "${output_content}\n${input_contents}")
  file(WRITE "${output_file}" "${output_content}")
endfunction()

# cmake-format: off
# Embed a source file in a static library and link it to a target. It inlines
# any local includes (#include "...")
#
# Parameters:
#   target:     CMake target to link to
#   input_file: Path to the file to embed
# cmake-format: on
# target_embed_source
function(target_embed_source target input_file)
  include(CMakeDetermineSystem)

  get_filename_component(name "${input_file}" NAME_WLE)
  get_filename_component(input_file_absolute "${input_file}" REALPATH)

  string(REPLACE "${PROJECT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}"
                 input_file_inlined "${input_file_absolute}"
  )

  get_filename_component(
    input_file_inlined_dir "${input_file_inlined}" DIRECTORY
  )
  file(MAKE_DIRECTORY "${input_file_inlined_dir}")

  get_local_includes(
    "${input_file_absolute}" "${PROJECT_SOURCE_DIR}" include_files
  )

  if("${include_files}" STREQUAL "")
    configure_file("${input_file_absolute}" "${input_file_inlined}" COPYONLY)
  else()
    inline_local_includes(
      "${input_file_absolute}" "${input_file_inlined}" "${include_files}"
    )
  endif()

  file(RELATIVE_PATH input_file_inlined_relative "${PROJECT_SOURCE_DIR}"
       "${input_file_absolute}"
  )

  set(embed_object_file "${CMAKE_CURRENT_BINARY_DIR}/${name}.o")
  set(embed_tool ld)
  set(embed_tool_args
      -r
      -b
      binary
      -A
      ${CMAKE_SYSTEM_PROCESSOR}
      -o
      "${embed_object_file}"
      "${input_file_inlined_relative}"
  )

  add_custom_command(
    OUTPUT "${embed_object_file}"
    COMMAND ${embed_tool} ARGS ${embed_tool_args}
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    DEPENDS "${input_file_absolute}" ${include_files}
    COMMENT "Embedding binary source: ${input_file}"
  )

  if(NOT TARGET ${name})
    add_library(${name} STATIC "${embed_object_file}")
    set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX)
  endif()

  target_link_libraries(${target} PRIVATE ${name})
endfunction()
