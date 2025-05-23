# Copy the contents of the input file to the output file with all the local
# includes inlined. Local includes are assumed to have ""'s, e.g. having a line
# '#include "helper.h"` will lead to `helper.h` being inlined. Only files in the
# root directory will be considered.
function(inline_local_includes input_file output_string root_dir)

  # If the list of already included files does not exist, initialize it
  if(NOT DEFINED ALREADY_INCLUDED)
    set(ALREADY_INCLUDED
        ""
        PARENT_SCOPE
    )
  endif()

  file(READ ${input_file} input_file_contents)
  set(include_regex "(^|\r?\n)(#include[ \t]*\"([^\"]+)\")")
  string(REGEX MATCHALL ${include_regex} includes ${input_file_contents})
  foreach(include ${includes})
    # Get the name of the file to include, e.g. 'helper.h'
    string(REGEX REPLACE ${include_regex} "\\3" include_name ${include})
    # Get the complete line of the include, e.g.  '#include <helper.h>'
    string(REGEX REPLACE ${include_regex} "\\2" include_line ${include})
    file(GLOB_RECURSE INCLUDE_PATHS "${root_dir}/*/${include_name}")
    if(NOT INCLUDE_PATHS STREQUAL "")
      list(SORT INCLUDE_PATHS ORDER DESCENDING)
      list(GET INCLUDE_PATHS 0 include_PATH)
      list(FIND ALREADY_INCLUDED ${INCLUDE_PATHS} found_index)
      set(include_contents "")
      if(found_index EQUAL -1)
        list(APPEND ALREADY_INCLUDED ${include_PATH})
        inline_local_includes(${include_PATH} include_contents ${root_dir})
      endif()
      # Replace the include line with the include file contents or
      # remove it if the file was already included
      string(REPLACE "${include_line}" "${include_contents}"
                     input_file_contents "${input_file_contents}"
      )
    endif()
  endforeach()

  set(${output_string}
      "${input_file_contents}"
      PARENT_SCOPE
  )

  set(ALREADY_INCLUDED
      ${ALREADY_INCLUDED}
      PARENT_SCOPE
  )
endfunction()

set(OUTPUT_STRING "")
inline_local_includes(${input_file} OUTPUT_STRING ${root_dir})
file(WRITE ${output_file} "${OUTPUT_STRING}")
