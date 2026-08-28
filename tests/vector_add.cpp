#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

void initialize_arrays(float *a, float *b, float *c, float *r, int N) {
  for (int i = 0; i < N; i++) {
    a[i] = 1.0 + i;
    b[i] = 2.0 - (N - i);
    c[i] = 0.0;
    r[i] = a[i] + b[i];
  }
}

int main(int argc, char* argv[]) {
  const std::string kernel = R"(
    extern "C" __global__ void vector_add(float *c, float *a, float *b, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) {
        c[i] = a[i] + b[i];
      }
    }
  )";

  cu::init();
  const int N = 1024;
  const size_t bytesize = N * sizeof(float);

  cu::Device device(0);

  cu::Stream stream;

  // std::vector<std::string> options = {};
  const std::vector<std::string> options = {"-ffast-math"};
  nvrtc::Program program(kernel, "vector_add_kernel.cu");
  try {
    program.compile(options);
    std::cout << "Compilation successfull!" << std::endl;
  } catch (nvrtc::Error &error) {
    std::cerr << program.getLog();
    throw;
  }
    // std::cout << "log: " << program.getLog() << std::endl;;

  std::string ptx = program.getPTX();
  // std::cout << "ptx: " << std::endl << ptx << std::endl;
  std::cout << "ptx: " << std::endl << ptx.size() << std::endl;
  return 0;

  cu::Module module(static_cast<const void *>(program.getPTX().data()));
  cu::Function function(module, "vector_add");


  cu::HostMemory h_a(bytesize);
  cu::HostMemory h_b(bytesize);
  cu::HostMemory h_c(bytesize);
  std::vector<float> reference_c(N);

  initialize_arrays(static_cast<float *>(h_a), static_cast<float *>(h_b),
                    static_cast<float *>(h_c), reference_c.data(), N);

  cu::DeviceMemory d_a(bytesize);
  cu::DeviceMemory d_b(bytesize);
  cu::DeviceMemory d_c(bytesize);

  stream.memcpyHtoDAsync(d_a, h_a, bytesize);
  stream.memcpyHtoDAsync(d_b, h_b, bytesize);
  std::vector<const void *> parameters = {d_c.parameter(), d_a.parameter(),
                                          d_b.parameter(), &N};
  stream.launchKernel(function, 1, 1, 1, N, 1, 1, 0, parameters);
  stream.memcpyDtoHAsync(h_c, d_c, bytesize);
  stream.synchronize();
}
