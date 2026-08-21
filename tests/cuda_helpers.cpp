#include <cuda.h>

#include <array>
#include <cstdio>
#include <string>

int cudaInit() {
  CUresult result = cuInit(0);
  return static_cast<int>(result);
}

int cudaDeviceCount() {
  int count = 0;
  cuDeviceGetCount(&count);
  return count;
}

void cudaListDevices() {
  int count = 0;
  cuDeviceGetCount(&count);

  for (int i = 0; i < count; i++) {
    CUdevice device;
    cuDeviceGet(&device, i);

    std::array<char, 256> name{};
    cuDeviceGetName(name.data(), static_cast<int>(name.size()), device);

    size_t totalMem = 0;
    cuDeviceTotalMem(&totalMem, device);

    int major = 0;
    int minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                         device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                         device);

    std::array<char, 256> pciBusId{};
    cuDeviceGetPCIBusId(pciBusId.data(), static_cast<int>(pciBusId.size()),
                        device);

    printf("  Device %d: %s\n", i, name.data());
    printf("    Memory:          %zu MiB\n", totalMem / (1024 * 1024));
    printf("    Compute capability: %d.%d\n", major, minor);
    printf("    PCI bus ID:      %s\n", pciBusId.data());
  }
}
