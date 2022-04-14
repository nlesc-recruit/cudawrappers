#include "cufft.hpp"

namespace cufft {

const char *Error::what() const noexcept {
  switch (_result) {
    case CUFFT_SUCCESS:
      return "success";
    case CUFFT_INVALID_PLAN:
      return "invalid plan";
    case CUFFT_ALLOC_FAILED:
      return "alloc failed";
    case CUFFT_INVALID_TYPE:
      return "invalid type";
    case CUFFT_INVALID_VALUE:
      return "invalid value";
    case CUFFT_INTERNAL_ERROR:
      return "internal error";
    case CUFFT_EXEC_FAILED:
      return "exec failed";
    case CUFFT_SETUP_FAILED:
      return "setup failed";
    case CUFFT_INVALID_SIZE:
      return "invalid size";
    case CUFFT_UNALIGNED_DATA:
      return "unaligned data";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "incomplete parameter list";
    case CUFFT_INVALID_DEVICE:
      return "invalid device";
    case CUFFT_PARSE_ERROR:
      return "parse error";
    case CUFFT_NO_WORKSPACE:
      return "no workspace";
    case CUFFT_NOT_IMPLEMENTED:
      return "not implemented";
    case CUFFT_LICENSE_ERROR:
      return "license error";
    case CUFFT_NOT_SUPPORTED:
      return "not supported";

    default:
      return "unknown";
  }
}

}  // namespace cufft
