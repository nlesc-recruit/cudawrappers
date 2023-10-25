# Make it possible to embed a source file in a library, and link it to a target.
# E.g. to link <kernel.cu> into target <example_program>, use
# target_embed_source(example_program, kernel.cu). This will expose symbols
# _binary_kernel_cu_start and _binary_kernel_cu_end.
function(target_embed_source target input_file)
  include(CMakeDetermineSystem)
  # Strip the path and extension from input_file
  get_filename_component(NAME ${input_file} NAME_WLE)

  set(work_dir ${ARGN})

  if(NOT DEFINED work_dir)
    set(work_dir ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
  # Link the input_file into an object file
  add_custom_command(
    OUTPUT ${NAME}.o
    COMMAND ld ARGS -r -b binary -A ${CMAKE_SYSTEM_PROCESSOR} -o
            ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.o ${input_file}
    WORKING_DIRECTORY ${work_dir}
    DEPENDS ${input_file}
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

set(MATHDX_VERSION "22.11.0")
set(MATHDX_ARCH "linux-x86_64")
set(MATHDX_BASEURL "https://developer.download.nvidia.com/\
compute/mathdx/redist/mathdx/${MATHDX_ARCH}"
)
set(MATHDX_URL ${MATHDX_BASEURL}/nvidia-mathdx-${MATHDX_VERSION}-Linux.tar.gz ")

# Make it possible to download and embed a NVIDIA mathdx library into a target.
# E.g. to link the nvidia library to target <example_program>, use
# target_add_mathdx(example_program). This will expose symbols
# _binary_nvidia_mathdx_linux_tar_gz_start and
# _binary__nvidia_mathdx_linux_tar_gz_end.
function(target_add_mathdx target)
  file(DOWNLOAD ${MATHDX_URL}
       ${CMAKE_CURRENT_BINARY_DIR}/nvidia-mathdx-linux.tar.gz
  )
  target_embed_source(
    target nvidia-mathdx-linux.tar.gz ${CMAKE_CURRENT_BINARY_DIR}
  )
endfunction()
