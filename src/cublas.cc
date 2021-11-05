#include "cublas.h"
#include <sstream>

namespace cublas {

const char *Error::what() const noexcept
{
  switch (_result) {
    case CUBLAS_STATUS_SUCCESS:
      return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "alloc failed";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "license error";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "arch mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "not supported";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "license error";

    default:
       std::stringstream str;
       str << "unknown error " << _result;
       return str.str().c_str();
   }
}

}

