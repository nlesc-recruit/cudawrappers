include("${CMAKE_CURRENT_LIST_DIR}/inline-common.cmake")

# cmake-format: off
#
# Inline a C/C++ source file with local includes as a binary object using `ld -b binary`.
#
# Usage:
#   target_embed_source(<target_name> <input_file>)
#
# Produces:
#   - A generated inlined source file with all local includes
#   - A binary object (.o) compiled from it
#   - A header file exposing _start, _end, _size symbols
#   - A static library target named after the input file's basename
#   - Links this static library to the provided target
#
# cmake-format: on
#
# target_embed_source(<target_name> <input_file>)
function(target_embed_source target_name input_file)
  get_filename_component(input_name "${input_file}" NAME)
  get_filename_component(input_basename "${input_file}" NAME_WE)
  get_filename_component(input_path "${input_file}" REALPATH)

  file(RELATIVE_PATH output_source_file "${PROJECT_SOURCE_DIR}" "${input_path}")
  set(output_object_file "${output_source_file}.o")

  set(all_deps "")
  set(processed_files "")
  get_dependencies("${input_path}" all_deps processed_files)

  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/${output_source_file}"
    COMMAND
      ${CMAKE_COMMAND} -Dinput_file="${input_path}"
      -Doutput_file="${output_source_file}" -Droot_dir="${PROJECT_SOURCE_DIR}"
      -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/inline-local-includes.cmake"
    DEPENDS "${input_path}" ${all_deps}
    COMMENT "Inlining all includes of ${input_file}"
  )

  string(REPLACE "." "_" symbol_base "${output_source_file}")
  string(REPLACE "/" "_" symbol_base "${symbol_base}")

  add_custom_command(
    OUTPUT "${output_object_file}"
    COMMAND ${CMAKE_LINKER} -r -b binary -A ${CMAKE_SYSTEM_PROCESSOR} -o
            "${output_object_file}" "${output_source_file}"
    DEPENDS "${output_source_file}"
    COMMENT "Creating object file for ${input_file}"
    VERBATIM
  )

  set(header_file "${CMAKE_BINARY_DIR}/${output_source_file}.h")
  set(header_content
      "
extern const unsigned char _binary_${symbol_base}_start[];
extern const unsigned char _binary_${symbol_base}_end[];
extern const unsigned int  _binary_${symbol_base}_size;
"
  )

  file(WRITE "${header_file}.in" "${header_content}")
  configure_file("${header_file}.in" "${header_file}" @ONLY)

  add_library(${input_basename} STATIC "${output_object_file}")
  set_target_properties(${input_basename} PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(${input_basename} PUBLIC "${CMAKE_BINARY_DIR}")
  target_link_libraries(${target_name} PRIVATE ${input_basename})
endfunction()
