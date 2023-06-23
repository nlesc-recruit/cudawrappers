// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

void vector_add() {
  const int N = 1024;
  size_t bytesize = N * sizeof(float);

  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);

  cu::HostMemory h_a(bytesize);
  cu::HostMemory h_b(bytesize);
  cu::HostMemory h_c(bytesize);

  auto *a = static_cast<float *>(h_a);
  auto *b = static_cast<float *>(h_b);
  auto *c = static_cast<float *>(h_c);
  for (int i = 0; i < N; i++) {
    a[i] = 1.0;
    b[i] = 1.0;
    c[i] = 0.0;
  }

  cu::DeviceMemory d_a(bytesize);
  cu::DeviceMemory d_b(bytesize);
  cu::DeviceMemory d_c(bytesize);

  cu::Stream my_stream;
  my_stream.memcpyHtoDAsync(d_a, a, bytesize);
  my_stream.memcpyHtoDAsync(d_b, b, bytesize);

  // compile kernel
  std::vector<std::string> options = {};
  const std::string vector_add_kernel =
#include "vector_add_kernel.cu"
      ;
  nvrtc::Program program(vector_add_kernel, "vector_add_kernel.cu");
  try {
    program.compile(options);
  } catch (nvrtc::Error &error) {
    std::cerr << program.getLog();
    throw;
  }

  cu::Module module =
      cu::Module(static_cast<const void *>(program.getPTX().data()));

  cu::Function function(module, "vector_add");

  // call kernel
  std::vector<const void *> parameters = {d_c.parameter(), d_a.parameter(),
                                          d_b.parameter(), &N};

  my_stream.launchKernel(function, 1, 1, 1, N, 1, 1, 0, parameters);

  my_stream.memcpyDtoHAsync(c, d_c, bytesize);

  my_stream.synchronize();

  std::cout << "hurray! " << c[0] << " \n";
}

int main(int argc, char *argv[]) {
  int err{0};
  try {
    cu::init();
    vector_add();
  } catch (cu::Error &error) {
    std::cerr << "cu::Error: " << error.what() << std::endl;
    err = 1;
  }
  return err;
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
