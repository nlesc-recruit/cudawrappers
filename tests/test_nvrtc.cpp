#include <catch2/catch_test_macros.hpp>
#include <filesystem>
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
#if defined(__HIP__)
  CHECK(path.size() > 0);
#else
  CHECK(path.find("include") != std::string::npos);
#endif
}

TEST_CASE("Test nvrtc::findIncludePaths", "[helper]") {
  const std::vector<std::string> paths = nvrtc::findIncludePaths();
  CHECK(paths.size() > 0);

  size_t non_empty_paths = 0;

  for (const std::filesystem::path &path : paths) {
    if (path.empty()) {
      continue;
    }

    ++non_empty_paths;
    CHECK(std::filesystem::exists(path));
    CHECK(std::filesystem::is_directory(path));

#if !defined(__HIP__)
    CHECK(path.string().find("include") != std::string::npos);
#endif
  }

  CHECK(non_empty_paths > 0);
}

TEST_CASE("Test nvrtc::version", "[version]") {
  auto [major, minor] = nvrtc::version();
  CHECK(major >= 0);
  CHECK(minor >= 0);
}

TEST_CASE("Test nvrtc::getSupportedArchs", "[archs]") {
  auto archs = nvrtc::getSupportedArchs();
  CHECK(archs.size() > 0);
}

TEST_CASE("Test nvrtc::util compiler options", "[util]") {
  // For every device (across all backends) the compiler options produced by
  // nvrtc::util must match the device's backend and be accepted by the
  // runtime compiler.  Backends whose devices cannot be enumerated in this
  // build are reported rather than failed.
  const std::string kernel = R"(
    extern "C" __global__ void nothing() {}
  )";
  int globalOffset = 0;
  int devicesTested = 0;
  for (size_t bi = 0; bi < getBackendCount(); ++bi) {
    int count = 0;
    try {
      // The apps always initialize the backend through context creation;
      // do the same here so device enumeration can succeed.
      if (getBackend(bi).init) getBackend(bi).init(0);
      count = cu::Device::getCount(static_cast<int>(bi));
    } catch (const std::exception &e) {
      INFO("backend " << bi << " not available: " << e.what());
      continue;
    }
    for (int local = 0; local < count; ++local) {
      const int ordinal = globalOffset + local;
      try {
        cu::Device device(ordinal);
        INFO("device " << device.getName());

        const int cap = nvrtc::util::capability(device);
        CHECK(cap >= 0);

        const std::string arch = nvrtc::util::archOption(device);
        if (device.isCuda()) {
          CHECK(arch.rfind("-arch=sm_", 0) == 0);
          if (cap >= 900) CHECK(arch.back() == 'a');
        } else {
          CHECK(arch.rfind("--offload-arch=", 0) == 0);
          CHECK(arch.size() > std::string("--offload-arch=").size());
        }

        CHECK(nvrtc::util::archDefine(device) ==
              "-D__HIP_ARCH__=" + std::to_string(cap));

        // The emitted option set must compile a trivial kernel on this device.
        nvrtc::Program program(kernel, "util_test.cu", {}, {},
                               device.getBackendIdx());
        CHECK_NOTHROW(program.compile(nvrtc::util::compileOptions(device)));
        ++devicesTested;
      } catch (const std::exception &e) {
        INFO("device " << ordinal << " failed: " << e.what());
      }
    }
    globalOffset += count;
  }
  CHECK(devicesTested > 0);
}
