#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <iostream>

extern int cudaInit();
extern int cudaDeviceCount();
extern void cudaListDevices();

extern int hipInitWrapper();
extern int hipDeviceCount();
extern void hipListDevices();

TEST_CASE("Test multi-backend GPU listing", "[multi_backend]") {
  int totalDevices = 0;

  SECTION("NVIDIA GPUs via CUDA") {
    int result = cudaInit();
    if (result != 0) {
      WARN("CUDA initialization failed (code " << result
                                               << "), skipping CUDA devices");
      return;
    }

    int count = cudaDeviceCount();
    CHECK(count >= 0);
    std::cout << "NVIDIA devices: " << count << std::endl;
    cudaListDevices();
    totalDevices += count;
  }

  SECTION("AMD GPUs via HIP") {
    int result = hipInitWrapper();
    if (result != 0) {
      WARN("HIP initialization failed (code " << result
                                              << "), skipping HIP devices");
      return;
    }

    int count = hipDeviceCount();
    CHECK(count >= 0);
    std::cout << "AMD devices: " << count << std::endl;
    hipListDevices();
    totalDevices += count;
  }
}

TEST_CASE("Test combined GPU count", "[multi_backend]") {
  int cudaCount = 0;
  int hipCount = 0;

  if (cudaInit() == 0) {
    cudaCount = cudaDeviceCount();
  }
  if (hipInitWrapper() == 0) {
    hipCount = hipDeviceCount();
  }

  int total = cudaCount + hipCount;
  std::cout << "Total GPUs: " << total << " (CUDA: " << cudaCount
            << ", HIP: " << hipCount << ")" << std::endl;

  CHECK(total >= 1);
}
