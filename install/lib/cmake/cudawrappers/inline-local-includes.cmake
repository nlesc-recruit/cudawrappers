include("${CMAKE_CURRENT_LIST_DIR}/inline-common.cmake")

# cmake-format: off
#
# Copy the contents of the input file to the output file with all the local
# includes inlined. Local includes are assumed to use quotes (""), e.g.
# '#include "helper.h"' will cause `helper.h` to be inlined recursively. Only
# files within the root directory are considered.
# Inline all local includes in a C/C++ source file by recursively expanding
# quoted includes (e.g., #include "header.h") into a single flat source file.
#
# Usage:
#   Called via -P in CMake script mode with the following variables defined:
#     - input_file: Path to the source file with includes
#     - output_file: Path to write the inlined result
#     - root_dir: Project root directory (used to search for includes)
#
# Behavior:
#   - Recursively resolves and inlines only quote-style includes ("...").
#   - Only files within the root directory are considered.
#   - Skips duplicate includes (once per file).
#
# cmake-format: on
# inline_local_includes(<input_file> <output_string> <root_dir>)
function(inline_local_includes input_file output_string root_dir)
  file(READ "${input_file}" input_contents)

  string(REGEX MATCHALL "(^|\r?\n)#include[ \t]*\"([^\"]+)\"" includes
               "${input_contents}"
  )

  set(processed_files "")
  set(already_included "")

  foreach(match IN LISTS includes)
    string(REGEX REPLACE "(^|\r?\n)#include[ \t]*\"([^\"]+)\"" "\\2"
                         include_name "${match}"
    )
    file(GLOB_RECURSE found_paths "${root_dir}/*/${include_name}")
    if(NOT found_paths STREQUAL "")
      list(SORT found_paths ORDER DESCENDING)
      list(GET found_paths 0 include_path)

      list(FIND already_included "${include_path}" found_idx)
      if(found_idx EQUAL -1)
        list(APPEND already_included "${include_path}")
        set(include_contents "")
        inline_local_includes("${include_path}" include_contents "${root_dir}")
        string(REPLACE "#include \"${include_name}\"" "${include_contents}"
                       input_contents "${input_contents}"
        )
      else()
        string(REPLACE "#include \"${include_name}\"" "" input_contents
                       "${input_contents}"
        )
      endif()
    endif()
  endforeach()

  set(${output_string}
      "${input_contents}"
      PARENT_SCOPE
  )
endfunction()

# Entry point
set(OUTPUT_STRING "")
inline_local_includes("${input_file}" OUTPUT_STRING "${root_dir}")
file(WRITE "${output_file}" "${OUTPUT_STRING}")
