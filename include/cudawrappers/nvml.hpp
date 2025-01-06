#if !defined NVML_H
#define NVML_H
// NVML is not avalailable on AMD
#if !defined(__HIP_PLATFORM_AMD__)

#include <nvml.h>

#include <exception>

#include <cudawrappers/cu.hpp>

namespace nvml {
class Error : public std::exception {
 public:
  explicit Error(nvmlReturn_t result) : _result(result) {}

  const char* what() const noexcept { return nvmlErrorString(_result); }

  operator nvmlReturn_t() const { return _result; }

 private:
  nvmlReturn_t _result;
};

inline void checkNvmlCall(nvmlReturn_t result) {
  if (result != NVML_SUCCESS) throw Error(result);
}

class Context {
 public:
  Context() { checkNvmlCall(nvmlInit()); }

  ~Context() { checkNvmlCall(nvmlShutdown()); }
};

class Device {
 public:
  Device(int index) {
    checkNvmlCall(nvmlDeviceGetHandleByIndex(index, &device_));
  }

  Device(cu::Device& device) {
    const std::string uuid = device.getUuid();
    checkNvmlCall(nvmlDeviceGetHandleByUUID(uuid.c_str(), &device_));
  }

  void getFieldValues(int valuesCount, nvmlFieldValue_t* values) {
    checkNvmlCall(nvmlDeviceGetFieldValues(device_, valuesCount, values));
  }

  unsigned int getClock(nvmlClockType_t clockType, nvmlClockId_t clockId) {
    unsigned int clockMhz;
    checkNvmlCall(nvmlDeviceGetClock(device_, clockType, clockId, &clockMhz));
    return clockMhz;
  }

  unsigned int getPower() {
    unsigned int power;
    checkNvmlCall(nvmlDeviceGetPowerUsage(device_, &power));
    return power;
  }

 private:
  nvmlDevice_t device_;
};
}  // namespace nvml

#endif  //  __HIP_PLATFORM_AMD__
#endif  // NVML_H
