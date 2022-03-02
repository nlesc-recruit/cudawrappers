#include "nvrtc.hpp"

namespace nvrtc {

const char *Error::what() const noexcept {
  return nvrtcGetErrorString(_result);
}

}  // namespace nvrtc
