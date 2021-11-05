#include "nvrtc.h"


namespace nvrtc {

const char *Error::what() const noexcept
{
  return nvrtcGetErrorString(_result);
}

}
