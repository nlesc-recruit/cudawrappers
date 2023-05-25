#include "cudawrappers/cu.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>

namespace cu {

const char *Error::what() const noexcept {
  const char *str{};
  return cuGetErrorString(_result, &str) != CUDA_ERROR_INVALID_VALUE
             ? str
             : "unknown error";
}

Context Device::primaryCtxRetain() {
  CUcontext context{};
  checkCudaCall(cuDevicePrimaryCtxRetain(&context, _obj));
  return {context, *this};
}

void DeviceMemory::zero(size_t size) {
  checkCudaCall(cuMemsetD8(_obj, 0, size));
}

void DeviceMemory::zero(size_t size, Stream &stream) {
  checkCudaCall(cuMemsetD8Async(_obj, 0, size, stream));
}

}  // namespace cu
