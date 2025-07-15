#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

#include <cudawrappers/nvrtc.hpp>

TEST_CASE("Test nvrtc::Program", "[program]") {
  const std::string kernel = R"(
    __global__ void vector_add(float *c, float *a, float *b, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) {
        c[i] = a[i] + b[i];
      }
    }
  )";

  nvrtc::Program program(kernel, "kernel.cu");

#if defined(__HIP__)
  const std::vector<std::string> options = {"-ffast-math"};
#else
  const std::vector<std::string> options = {"-use_fast_math"};
#endif

  SECTION("Test Program.compile") { CHECK_NOTHROW(program.compile(options)); }

  SECTION("Test Program.getPTX") {
    program.compile(options);
    const std::string ptx{program.getPTX()};
    CHECK(ptx.size() > 0);
  }
}

#include "tests/kernels/vector_add_kernel.cu.o.h"

TEST_CASE("Test nvrtc::Program embedded source", "[program]") {
  nvrtc::Program program(vector_add_kernel_source, "vector_add_kernel.cu");

#if defined(__HIP__)
  const std::vector<std::string> options = {"-ffast-math"};
#else
  const std::vector<std::string> options = {"-use_fast_math"};
#endif

  SECTION("Test Program.compile") { CHECK_NOTHROW(program.compile(options)); }

  SECTION("Test Program.getPTX") {
    program.compile(options);
    const std::string ptx{program.getPTX()};
    CHECK(ptx.size() > 0);
  }
}

extern const char _binary_tests_kernels_single_include_kernel_cu_start,
    _binary_tests_kernels_single_include_kernel_cu_end;

TEST_CASE("Test nvrtc::Program inlined header", "[program]") {
  const std::string kernel(
      &_binary_tests_kernels_single_include_kernel_cu_start,
      &_binary_tests_kernels_single_include_kernel_cu_end);
  nvrtc::Program program(kernel, "single_include_kernel.cu");

  const std::vector<std::string> options = {};

  SECTION("Test Program.compile") { CHECK_NOTHROW(program.compile(options)); }
}

extern const char _binary_tests_kernels_recursive_include_kernel_cu_start,
    _binary_tests_kernels_recursive_include_kernel_cu_end;

TEST_CASE("Test nvrtc::Program recursively inlined header", "[program]") {
  const std::string kernel(
      &_binary_tests_kernels_recursive_include_kernel_cu_start,
      &_binary_tests_kernels_recursive_include_kernel_cu_end);
  nvrtc::Program program(kernel, "recursive_include_kernel.cu");

  const std::vector<std::string> options = {};

  SECTION("Test Program.compile") { CHECK_NOTHROW(program.compile(options)); }
}

TEST_CASE("Test nvrtc::findIncludePath", "[helper]") {
  const std::string path = nvrtc::findIncludePath();
  CHECK(path.find("include") != std::string::npos);
}
