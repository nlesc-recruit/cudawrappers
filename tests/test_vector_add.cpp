#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <cuda.h>
#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

void check_arrays_equal(const float *a, const float *b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    CHECK(a[i] == Approx(b[i]).epsilon(1e-6));
  }
}

TEST_CASE("Vector add") {
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
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);

  cu::HostMemory h_a(bytesize);
  cu::HostMemory h_b(bytesize);
  cu::HostMemory h_c(bytesize);
  std::vector<float> reference_c(N);

  float *a = static_cast<float *>(h_a);
  float *b = static_cast<float *>(h_b);
  float *c = static_cast<float *>(h_c);
  for (int i = 0; i < N; i++) {
    a[i] = 1.0 + i;
    b[i] = 2.0 - (N - i);
    c[i] = 0.0;
    reference_c[i] = a[i] + b[i];
  }

  cu::DeviceMemory d_a(bytesize);
  cu::DeviceMemory d_b(bytesize);
  cu::DeviceMemory d_c(bytesize);

  cu::Stream stream;
  stream.memcpyHtoDAsync(d_a, a, bytesize);
  stream.memcpyHtoDAsync(d_b, b, bytesize);

  std::vector<std::string> options = {};
  nvrtc::Program program(kernel, "vector_add_kernel.cu");
  try {
    program.compile(options);
  } catch (nvrtc::Error &error) {
    std::cerr << program.getLog();
    throw;
  }

  cu::Module module(static_cast<const void *>(program.getPTX().data()));
  cu::Function function(module, "vector_add");

  SECTION("Run kernel") {
    std::vector<const void *> parameters = {d_c.parameter(), d_a.parameter(),
                                            d_b.parameter(), &N};
    stream.launchKernel(function, 1, 1, 1, N, 1, 1, 0, parameters);
    stream.memcpyDtoHAsync(c, d_c, bytesize);
    stream.synchronize();
    check_arrays_equal(c, reference_c.data(), N);
  }
}
