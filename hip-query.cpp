#include <iostream>

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

int main() {
  hipDeviceProp_t prop;
  int deviceCount;

  hipGetDeviceCount(&deviceCount);
  std::cout << "Number of HIP devices: " << deviceCount << std::endl;

  for (int i = 0; i < deviceCount; ++i) {
    hipGetDeviceProperties(&prop, i);
    std::cout << "Device " << i << " Name: " << prop.name << std::endl;
    std::cout << "      Architecture: " << prop.gcnArchName << std::endl;
    std::cout << "      Compute Capability: " << prop.major << "." << prop.minor
              << std::endl;
    std::cout << std::endl;
  }

  return 0;
}
