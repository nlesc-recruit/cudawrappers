#include <hip/hip_runtime.h>

#include <cstdio>

int hipInitWrapper() {
  hipError_t result = hipInit(0);
  return static_cast<int>(result);
}

int hipDeviceCount() {
  int count = 0;
  static_cast<void>(hipGetDeviceCount(&count));
  return count;
}

void hipListDevices() {
  int count = 0;
  static_cast<void>(hipGetDeviceCount(&count));

  for (int i = 0; i < count; i++) {
    hipDeviceProp_t prop;
    static_cast<void>(hipGetDeviceProperties(&prop, i));

    printf("  Device %d: %s\n", i, prop.name);
    printf("    Memory:          %zu MiB\n",
           prop.totalGlobalMem / (1024 * 1024));
    printf("    Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("    Multi-processors: %d\n", prop.multiProcessorCount);
  }
}
