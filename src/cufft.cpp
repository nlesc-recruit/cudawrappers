#include "cufft.hpp"

#include <array>

namespace cufft {

const std::array<const char *const, MAX_CUFFT_ERROR> CUFFT_STATUS_MSG = {
    "success",
    "invalid plan",
    "alloc failed",
    "invalid type",
    "invalid value",
    "internal error",
    "exec failed",
    "setup failed",
    "invalid size",
    "unaligned data",
    "incomplete parameter list",
    "invalid device",
    "parse error",
    "no workspace",
    "not implemented",
    "license error",
    "not supported"};

const char *Error::what() const noexcept {
  return CUFFT_STATUS_MSG.at(_result);
}

}  // namespace cufft
