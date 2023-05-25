#include <string>
#include <vector>

#include <catch2/catch.hpp>
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

  SECTION("Test Program.compile") {
    const std::vector<std::string> options = {"--generate-line-info"};
    REQUIRE_NOTHROW(program.compile(options));
  }

  SECTION("Test Program.getPTX") {
    const std::string ptx{program.getPTX()};
    REQUIRE(ptx.size() > 0);
  }
}
