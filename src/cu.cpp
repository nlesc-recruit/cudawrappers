#include "cu.hpp"

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

void Source::compile(const char *output_file_name,
                     const char *compiler_options) {
  std::stringstream command_line;
  command_line << "nvcc -cubin " << compiler_options << " -o "
               << output_file_name << ' ' << _input_file_name;
  //#pragma omp critical (clog)
  // std::clog << command_line.str() << std::endl;

  int retval = system(command_line.str().c_str());

  if (WEXITSTATUS(retval) != 0) throw Error(CUDA_ERROR_INVALID_SOURCE);
}

}  // namespace cu
