if(${CUDAWRAPPERS_BACKEND_HIP})
  # cmake-format: off
  # This following code attempts to locate the HIP runtime library's root
  # directory.
  #
  # 1. Checks if the ROCM_ROOT and ROCM_PATH variables are defined.
  #    - Assign them from corresponding environment variables if not defined.
  # 2. Searches for the HIP runtime header file in the paths specified
  #    by ROCM_ROOT and ROCM_PATH.
  #    - If the path is set but HIP is not found, an error message is generated.
  # 3. If HIP is still not found, the script searches in the default path.
  # 4. Adding HIP_ROOT_DIR to CMAKE_PREFIX_PATH and Finding Packages
  #    - The HIP runtime directory is appended to CMAKE_PREFIX_PATH
  #      to ensure that CMake can find the HIP-related packages.
  #
  # Usage:
  # Set the ROCM_ROOT or ROCM_PATH environment variables, or
  # pass them as CMake options, e.g.:
  #   cmake -DROCM_ROOT=/path/to/rocm ..
  #   or
  #   ROCM_ROOT=/path/to/rocm cmake ..
  # cmake-format: on

  foreach(var IN ITEMS ROCM_ROOT ROCM_PATH)
    # Step 1.
    if(NOT DEFINED ${var})
      set(${var} $ENV{${var}})
    endif()

    # Step 2.
    find_path(
      HIP_ROOT_DIR
      NAMES include/hip/hip_runtime.h
      PATHS ${${var}}
      NO_DEFAULT_PATH
    )

    if(NOT HIP_ROOT_DIR AND ${var})
      message(FATAL_ERROR "HIP not found in " ${${var}})
    endif()
  endforeach()

  if(NOT HIP_ROOT_DIR)
    # Step 3.
    find_path(
      HIP_ROOT_DIR
      NAMES include/hip/hip_runtime.h
      PATHS /opt/rocm
    )
  endif()

  # Step 4.
  list(APPEND CMAKE_PREFIX_PATH ${HIP_ROOT_DIR})
  set(HIP_MIN_VERSION 6.1)
  find_package(hip ${HIP_MIN_VERSION} REQUIRED)
  if(CUDAWRAPPERS_BUILD_CUFFT)
    find_package(hipfft QUIET)
    if(NOT hipfft_FOUND)
      message(WARNING "hipfft was not found, cufft component is disabled.")
      list(REMOVE_ITEM CUDAWRAPPERS_COMPONENTS cufft)
      set(CUDAWRAPPERS_BUILD_CUFFT OFF)
    endif()
  endif()
else()
  # cudawrappers requires the CUDA Toolkit.If you include cudawrappers in your
  # project, you need to include the toolkit yourself
  set(CUDA_MIN_VERSION 10.0)
  if(${CUDAWRAPPERS_INSTALLED})
    find_package(CUDAToolkit ${CUDA_MIN_VERSION} REQUIRED)
  else()
    if(${CUDAToolkit_FOUND})
      if(${CUDAToolkit_VERSION_MAJOR} LESS ${CUDA_MIN_VERSION})
        message(FATAL_ERROR "Insufficient CUDA version: "
                            ${CUDAToolkit_VERSION} " < " ${CUDA_MIN_VERSION}
        )
      endif()
    else()
      message(
        FATAL_ERROR
          "CUDAToolkit not found, use find_package(CUDAToolkit REQUIRED)."
      )
    endif()
  endif()
endif()
