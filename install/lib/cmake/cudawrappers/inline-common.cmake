# cmake-format: off
#
# Common CMake utilities for source file inlining.
#
# Provides:
#   - get_dependencies(input deps_var processed_var)
#
# Description:
#   Recursively collects local dependencies from a C/C++ source file by scanning
#   for quote-style includes (e.g., #include "header.h").
#
#   The result is stored in a list variable <deps_var>, with duplicate prevention
#   using a <processed_var> list to track visited files.
#
#   Intended for use in scripts that inline or bundle source files.

# cmake-format: on
#
# get_dependencies(<input> <deps_var> <processed_var>)
function(get_dependencies input deps_var processed_var)
  list(FIND ${processed_var} "${input}" idx)
  if(NOT idx EQUAL -1)
    return()
  endif()

  list(APPEND ${processed_var} "${input}")

  file(READ "${input}" content)
  string(REGEX MATCHALL "#[ \t]*include[ \t]+\"([^\"]+)\"" includes
               "${content}"
  )

  get_filename_component(input_dir "${input}" DIRECTORY)

  foreach(match IN LISTS includes)
    string(REGEX REPLACE "#[ \t]*include[ \t]+\"([^\"]+)\"" "\\1" included_file
                         "${match}"
    )
    set(full_path "${input_dir}/${included_file}")

    if(EXISTS "${full_path}")
      get_dependencies("${full_path}" ${deps_var} ${processed_var})
      list(APPEND ${deps_var} "${full_path}")
    endif()
  endforeach()

  set(${deps_var}
      "${${deps_var}}"
      PARENT_SCOPE
  )
  set(${processed_var}
      "${${processed_var}}"
      PARENT_SCOPE
  )
endfunction()
