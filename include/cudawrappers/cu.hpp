#if !defined CU_WRAPPER_H
#define CU_WRAPPER_H

#include <array>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(__HIP__)
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
#include <cudawrappers/macros.hpp>

typedef uint32_t cuuint32_t;

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif

namespace cu {
class Error : public std::exception {
 public:
  explicit Error(CUresult result) : _result(result) {}

  const char *what() const noexcept {
    const char *str{};
    return cuGetErrorString(_result, &str) != CUDA_ERROR_INVALID_VALUE
               ? str
               : "unknown error";
  }

  operator CUresult() const { return _result; }

 private:
  CUresult _result;
};

inline void checkCudaCall(CUresult result) {
  if (result != CUDA_SUCCESS) throw Error(result);
}

inline void init(unsigned flags = 0) { checkCudaCall(cuInit(flags)); }

inline int driverGetVersion() {
  int version{};
  checkCudaCall(cuDriverGetVersion(&version));
  return version;
}

inline void memcpyHtoD(CUdeviceptr dst, const void *src, size_t size) {
#if defined(__HIP__)
  // const_cast is a temp fix for https://github.com/ROCm/ROCm/issues/2977
  checkCudaCall(cuMemcpyHtoD(dst, const_cast<void *>(src), size));
#else
  checkCudaCall(cuMemcpyHtoD(dst, src, size));
#endif
}

inline void memcpyDtoH(void *dst, CUdeviceptr src, size_t size) {
  checkCudaCall(cuMemcpyDtoH(dst, src, size));
}

class Context;
class Stream;

template <typename T>
class Wrapper {
 public:
  // conversion to C-style T

  operator T() const { return _obj; }

  operator T() { return _obj; }

  bool operator==(const Wrapper<T> &other) { return _obj == other._obj; }

  bool operator!=(const Wrapper<T> &other) { return _obj != other._obj; }

 protected:
  Wrapper() = default;

  Wrapper(const Wrapper<T> &other) : _obj(other._obj), manager(other.manager) {}

  Wrapper(Wrapper<T> &&other)
      : _obj(other._obj), manager(std::move(other.manager)) {
    other._obj = 0;
  }

  explicit Wrapper(T &obj) : _obj(obj) {}

  T _obj{};
  std::shared_ptr<T> manager;
};

class Device : public Wrapper<CUdevice> {
 public:
  // Device Management

  explicit Device(int ordinal) : _ordinal(ordinal) {
    checkCudaCall(cuDeviceGet(&_obj, ordinal));
  }

  struct CUdeviceArg {
  };  // int and CUdevice are the same type, but we need two constructors
  Device(CUdeviceArg, CUdevice device) : Wrapper(device) {}

  int getAttribute(CUdevice_attribute attribute) const {
    int value{};
    checkCudaCall(cuDeviceGetAttribute(&value, attribute, _obj));
    return value;
  }

  template <CUdevice_attribute attribute>
  int getAttribute() const {
    return getAttribute(attribute);
  }

  static int getCount() {
    int nrDevices{};
    checkCudaCall(cuDeviceGetCount(&nrDevices));
    return nrDevices;
  }

  std::string getName() const {
    const size_t max_device_name_length{64};
    std::array<char, max_device_name_length> name{};
    checkCudaCall(cuDeviceGetName(name.data(), name.size(), _obj));
    return {name.data()};
  }

  std::string getUuid() const {
    CUuuid uuid;
    checkCudaCall(cuDeviceGetUuid(&uuid, _obj));

    // Convert a CUuuid to CUDA's string representation.
    // The CUuuid contains an array of 16 bytes, the UUID has
    // the form 'GPU-XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXX', with every
    // X being an alphanumeric character.
    std::stringstream result;
    result << "GPU";

    for (int i = 0; i < 16; ++i) {
      if (i == 0 || i == 4 || i == 6 || i == 8 || i == 10) {
        result << "-";
      }
      result << std::hex << std::setfill('0') << std::setw(2)
             << static_cast<unsigned>(
                    static_cast<unsigned char>(uuid.bytes[i]));
    }

    return result.str();
  }

  std::string getArch() const {
#if defined(__HIP__)
    hipDeviceProp_t prop;
    checkCudaCall(hipGetDeviceProperties(&prop, _ordinal));
    return prop.gcnArchName;
#else
    return std::to_string(
        10 *

            getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>() +
        getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>());

#endif
  }

  size_t totalMem() const {
    size_t size{};
    checkCudaCall(cuDeviceTotalMem(&size, _obj));
    return size;
  }

  // Primary Context Management
#if !defined(__HIP__)
  std::pair<unsigned, bool> primaryCtxGetState() const {
    unsigned flags{};
    int active{};
    checkCudaCall(cuDevicePrimaryCtxGetState(_obj, &flags, &active));
    return {flags, active};
  }
#endif

  // void primaryCtxRelease() not available; it is released on destruction of
  // the Context returned by Device::primaryContextRetain()

#if !defined(__HIP__)
  void primaryCtxReset() { checkCudaCall(cuDevicePrimaryCtxReset(_obj)); }
#endif

  Context primaryCtxRetain();  // retain this context until the primary context
                               // can be released

#if !defined(__HIP__)
  void primaryCtxSetFlags(unsigned flags) {
    checkCudaCall(cuDevicePrimaryCtxSetFlags(_obj, flags));
  }
#endif

 private:
  int _ordinal;
};

class Context : public Wrapper<CUcontext> {
 public:
  // Context Management

  Context(int flags, Device &device) : _primaryContext(false), _device(device) {
#if !defined(__HIP__)
    checkCudaCall(cuCtxCreate(&_obj, flags, device));
    manager =
        std::shared_ptr<CUcontext>(new CUcontext(_obj), [](CUcontext *ptr) {
          if (*ptr) cuCtxDestroy(*ptr);
          delete ptr;
        });
#endif
  }

#if !defined(__HIP__)
  unsigned getApiVersion() const {
    unsigned version{};
    checkCudaCall(cuCtxGetApiVersion(_obj, &version));
    return version;
  }
#endif

#if !defined(__HIP__)
  static CUfunc_cache getCacheConfig() {
    CUfunc_cache config{};
    checkCudaCall(cuCtxGetCacheConfig(&config));
    return config;
  }
#endif

#if !defined(__HIP__)
  static void setCacheConfig(CUfunc_cache config) {
    checkCudaCall(cuCtxSetCacheConfig(config));
  }
#endif

  Context getCurrent() {
    CUcontext context{};
#if !defined(__HIP__)
    checkCudaCall(cuCtxGetCurrent(&context));
#endif
    return Context(context, _device);
  }

#if !defined(__HIP__)
  void setCurrent() const { checkCudaCall(cuCtxSetCurrent(_obj)); }
#endif

#if !defined(__HIP__)
  void pushCurrent() { checkCudaCall(cuCtxPushCurrent(_obj)); }
#endif

  Device getDevice() {
    CUdevice device;
#if !defined(__HIP__)
    checkCudaCall(cuCtxGetDevice(&device));
#else
    device = _device;
#endif
    return Device(Device::CUdeviceArg(), device);
  }

  static size_t getLimit(CUlimit limit) {
    size_t value{};
    checkCudaCall(cuCtxGetLimit(&value, limit));
    return value;
  }

  template <CUlimit limit>
  static size_t getLimit() {
    return getLimit(limit);
  }

  static void setLimit(CUlimit limit, size_t value) {
    checkCudaCall(cuCtxSetLimit(limit, value));
  }

  template <CUlimit limit>
  static void setLimit(size_t value) {
    setLimit(limit, value);
  }

  size_t getFreeMemory() const {
    size_t free;
    size_t total;
    checkCudaCall(cuMemGetInfo(&free, &total));
    return free;
  }

  size_t getTotalMemory() const {
    size_t free;
    size_t total;
    checkCudaCall(cuMemGetInfo(&free, &total));
    return total;
  }

#if !defined(__HIP__)
  static void synchronize() { checkCudaCall(cuCtxSynchronize()); }
#endif

 private:
  friend class Device;
  Context(CUcontext context, Device &device)
      : Wrapper<CUcontext>(context), _primaryContext(true), _device(device) {}

  bool _primaryContext;
  cu::Device &_device;
};

class HostMemory : public Wrapper<void *> {
 public:
  explicit HostMemory(size_t size, unsigned int flags = 0) : _size(size) {
    checkCudaCall(cuMemHostAlloc(&_obj, size, flags));
    manager = std::shared_ptr<void *>(new (void *)(_obj), [](void **ptr) {
      checkCudaCall(cuMemFreeHost(*ptr));
      delete ptr;
    });
  }

  explicit HostMemory(void *ptr, size_t size, unsigned int flags = 0)
      : _size(size) {
    _obj = ptr;
    checkCudaCall(cuMemHostRegister(_obj, size, flags));
    manager = std::shared_ptr<void *>(new (void *)(_obj), [](void **ptr) {
      checkCudaCall(cuMemHostUnregister(*ptr));
    });
  }

  template <typename T>
  operator T *() {
    return static_cast<T *>(_obj);
  }

  size_t size() const { return _size; }

 private:
  size_t _size;
};

class Array : public Wrapper<CUarray> {
 public:
  Array(unsigned width, CUarray_format format, unsigned numChannels) {
    create2DArray(width, 0, format, numChannels);
  }

  Array(unsigned width, unsigned height, CUarray_format format,
        unsigned numChannels) {
    create2DArray(width, height, format, numChannels);
  }

  Array(unsigned width, unsigned height, unsigned depth, CUarray_format format,
        unsigned numChannels) {
    CUDA_ARRAY3D_DESCRIPTOR descriptor;
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.Depth = depth;
    descriptor.Format = format;
    descriptor.NumChannels = numChannels;
    descriptor.Flags = 0;
    checkCudaCall(cuArray3DCreate(&_obj, &descriptor));
    createManager();
  }

  explicit Array(CUarray &array) : Wrapper(array) {}

 private:
  void create2DArray(unsigned width, unsigned height, CUarray_format format,
                     unsigned numChannels) {
    CUDA_ARRAY_DESCRIPTOR descriptor;
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.Format = format;
    descriptor.NumChannels = numChannels;
    checkCudaCall(cuArrayCreate(&_obj, &descriptor));
    createManager();
  }

  void createManager() {
    manager = std::shared_ptr<CUarray>(new CUarray(_obj), [](CUarray *ptr) {
      checkCudaCall(cuArrayDestroy(*ptr));
      delete ptr;
    });
  }
};

class Module : public Wrapper<CUmodule> {
 public:
  explicit Module(const char *file_name) {
#if defined TEGRA_QUIRKS  // cuModuleLoad broken on Jetson TX1
    std::ifstream file(file_name);
    std::string program((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    checkCudaCall(cuModuleLoadData(&_obj, program.c_str()));
#else
    checkCudaCall(cuModuleLoad(&_obj, file_name));
#endif
    manager = std::shared_ptr<CUmodule>(new CUmodule(_obj), [](CUmodule *ptr) {
      checkCudaCall(cuModuleUnload(*ptr));
      delete ptr;
    });
  }

  explicit Module(const void *data) {
    checkCudaCall(cuModuleLoadData(&_obj, data));
    manager = std::shared_ptr<CUmodule>(new CUmodule(_obj), [](CUmodule *ptr) {
      checkCudaCall(cuModuleUnload(*ptr));
      delete ptr;
    });
  }

  typedef std::map<CUjit_option, void *> optionmap_t;
  explicit Module(const void *image, Module::optionmap_t &options) {
    std::vector<CUjit_option> keys;
    std::vector<void *> values;

    for (const std::pair<CUjit_option, void *> &i : options) {
      keys.push_back(i.first);
      values.push_back(i.second);
    }

    checkCudaCall(cuModuleLoadDataEx(&_obj, image, options.size(), keys.data(),
                                     values.data()));

    for (size_t i = 0; i < keys.size(); ++i) {
      options[keys[i]] = values[i];
    }
  }

  explicit Module(CUmodule &module) : Wrapper(module) {}

  CUdeviceptr getGlobal(const char *name) const {
    CUdeviceptr deviceptr{};
    checkCudaCall(cuModuleGetGlobal(&deviceptr, nullptr, _obj, name));
    return deviceptr;
  }
};

class Function : public Wrapper<CUfunction> {
 public:
  Function(const Module &module, const char *name) : _name(name) {
    checkCudaCall(cuModuleGetFunction(&_obj, module, name));
  }

  explicit Function(CUfunction &function) : Wrapper(function) {}

#if defined(__HIP__)
  int getAttribute(hipFunction_attribute attribute) const {
    int value{};
    checkCudaCall(cuFuncGetAttribute(&value, attribute, _obj));
    return value;
  }
#else
  int getAttribute(CUfunction_attribute attribute) const {
    int value{};
    checkCudaCall(cuFuncGetAttribute(&value, attribute, _obj));
    return value;
  }
#endif

  void setAttribute(CUfunction_attribute attribute, int value) {
    checkCudaCall(cuFuncSetAttribute(_obj, attribute, value));
  }

  int occupancyMaxActiveBlocksPerMultiprocessor(int blockSize,
                                                size_t dynamicSMemSize) {
    int numBlocks;
    checkCudaCall(cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, _obj, blockSize, dynamicSMemSize));
    return numBlocks;
  }

  void setCacheConfig(CUfunc_cache config) {
    checkCudaCall(cuFuncSetCacheConfig(_obj, config));
  }

  const char *name() const { return _name; }

 private:
  const char *_name;
};

class Event : public Wrapper<CUevent> {
 public:
  explicit Event(unsigned int flags = CU_EVENT_DEFAULT) {
    checkCudaCall(cuEventCreate(&_obj, flags));
    manager = std::shared_ptr<CUevent>(new CUevent(_obj), [](CUevent *ptr) {
      checkCudaCall(cuEventDestroy(*ptr));
      delete ptr;
    });
  }

  explicit Event(CUevent &event) : Wrapper(event) {}

  float elapsedTime(const Event &start) const {
    float ms{};
    checkCudaCall(cuEventElapsedTime(&ms, start, _obj));
    return ms;
  }

  void query() const {
    checkCudaCall(cuEventQuery(_obj));  // unsuccessful result throws cu::Error
  }

  void record() { checkCudaCall(cuEventRecord(_obj, 0)); }

  void record(Stream &);

  void synchronize() { checkCudaCall(cuEventSynchronize(_obj)); }
};

class DeviceMemory : public Wrapper<CUdeviceptr> {
 public:
  explicit DeviceMemory(size_t size, CUmemorytype type = CU_MEMORYTYPE_DEVICE,
                        unsigned int flags = 0)
      : _size(size) {
    if (size == 0) {
      _obj = 0;
      return;
    } else if (type == CU_MEMORYTYPE_DEVICE && !flags) {
      checkCudaCall(cuMemAlloc(&_obj, size));
    } else if (type == CU_MEMORYTYPE_UNIFIED) {
      checkCudaCall(cuMemAllocManaged(&_obj, size, flags));
    } else {
      throw Error(CUDA_ERROR_INVALID_VALUE);
    }
    manager = std::shared_ptr<CUdeviceptr>(new CUdeviceptr(_obj),
                                           [](CUdeviceptr *ptr) {
                                             cuMemFree(*ptr);
                                             delete ptr;
                                           });
  }

  explicit DeviceMemory(CUdeviceptr ptr) : Wrapper(ptr) {}

  explicit DeviceMemory(CUdeviceptr ptr, size_t size)
      : Wrapper(ptr), _size(size) {}

  explicit DeviceMemory(const HostMemory &hostMemory) {
    checkCudaCall(cuMemHostGetDevicePointer(&_obj, hostMemory, 0));
  }

  void zero(size_t size) { checkCudaCall(cuMemsetD8(_obj, 0, size)); }

  const void *parameter()
      const  // used to construct parameter list for launchKernel();
  {
    return &_obj;
  }

  template <typename T>
  operator T *() {
    int data;
    checkCudaCall(
        cuPointerGetAttribute(&data, CU_POINTER_ATTRIBUTE_IS_MANAGED, _obj));
    if (data) {
      return reinterpret_cast<T *>(_obj);
    } else {
      throw std::runtime_error(
          "Cannot return memory of type CU_MEMORYTYPE_DEVICE as pointer.");
    }
  }

  size_t size() const { return _size; }

 private:
  size_t _size;
};

class Stream : public Wrapper<CUstream> {
  friend class Event;

 public:
  explicit Stream(unsigned int flags = CU_STREAM_DEFAULT) {
    checkCudaCall(cuStreamCreate(&_obj, flags));
    manager = std::shared_ptr<CUstream>(new CUstream(_obj), [](CUstream *ptr) {
      checkCudaCall(cuStreamDestroy(*ptr));
      delete ptr;
    });
  }

  explicit Stream(CUstream stream) : Wrapper<CUstream>(stream) {}

  DeviceMemory memAllocAsync(size_t size) {
    CUdeviceptr ptr;
    checkCudaCall(cuMemAllocAsync(&ptr, size, _obj));
    return DeviceMemory(ptr, size);
  }

  void memFreeAsync(DeviceMemory &devMem) {
    checkCudaCall(cuMemFreeAsync(devMem, _obj));
  }

  void memcpyHtoHAsync(void *dstPtr, const void *srcPtr, size_t size) {
#if defined(__HIP__)
    checkCudaCall(hipMemcpyAsync(
        reinterpret_cast<CUdeviceptr>(dstPtr),
        reinterpret_cast<CUdeviceptr>(const_cast<void *>(srcPtr)), size,
        hipMemcpyDefault, _obj));
#else
    checkCudaCall(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dstPtr),
                                reinterpret_cast<CUdeviceptr>(srcPtr), size,
                                _obj));
#endif
  }

  void memcpyHtoDAsync(DeviceMemory &devPtr, const void *hostPtr, size_t size) {
#if defined(__HIP__)
    checkCudaCall(
        hipMemcpyHtoDAsync(devPtr, const_cast<void *>(hostPtr), size, _obj));
#else
    checkCudaCall(cuMemcpyHtoDAsync(devPtr, hostPtr, size, _obj));
#endif
  }

  void memcpyHtoDAsync(CUdeviceptr devPtr, const void *hostPtr, size_t size) {
#if defined(__HIP__)
    checkCudaCall(
        hipMemcpyHtoDAsync(devPtr, const_cast<void *>(hostPtr), size, _obj));
#else
    checkCudaCall(cuMemcpyHtoDAsync(devPtr, hostPtr, size, _obj));
#endif
  }

  void memcpyDtoHAsync(void *hostPtr, const DeviceMemory &devPtr, size_t size) {
    checkCudaCall(cuMemcpyDtoHAsync(hostPtr, devPtr, size, _obj));
  }

  void memcpyDtoHAsync(void *hostPtr, CUdeviceptr devPtr, size_t size) {
    checkCudaCall(cuMemcpyDtoHAsync(hostPtr, devPtr, size, _obj));
  }

  void memcpyDtoDAsync(DeviceMemory &dstPtr, DeviceMemory &srcPtr,
                       size_t size) {
#if defined(__HIP__)
    checkCudaCall(hipMemcpyAsync(dstPtr, srcPtr, size, hipMemcpyDefault, _obj));
#else
    checkCudaCall(cuMemcpyAsync(dstPtr, srcPtr, size, _obj));
#endif
  }

  void memPrefetchAsync(DeviceMemory &devPtr, size_t size) {
    checkCudaCall(cuMemPrefetchAsync(devPtr, size, CU_DEVICE_CPU, _obj));
  }

  void memPrefetchAsync(DeviceMemory &devPtr, size_t size, Device &dstDevice) {
    checkCudaCall(cuMemPrefetchAsync(devPtr, size, dstDevice, _obj));
  }

  void zero(DeviceMemory &devPtr, size_t size) {
    checkCudaCall(cuMemsetD8Async(devPtr, 0, size, _obj));
  }

  void launchKernel(Function &function, unsigned gridX, unsigned gridY,
                    unsigned gridZ, unsigned blockX, unsigned blockY,
                    unsigned blockZ, unsigned sharedMemBytes,
                    const std::vector<const void *> &parameters) {
    checkCudaCall(cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY,
                                 blockZ, sharedMemBytes, _obj,
                                 const_cast<void **>(&parameters[0]), nullptr));
  }

#if CUDART_VERSION >= 9000
  void launchCooperativeKernel(Function &function, unsigned gridX,
                               unsigned gridY, unsigned gridZ, unsigned blockX,
                               unsigned blockY, unsigned blockZ,
                               unsigned sharedMemBytes,
                               const std::vector<const void *> &parameters) {
    checkCudaCall(cuLaunchCooperativeKernel(
        function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
        _obj, const_cast<void **>(&parameters[0])));
  }
#endif

  void query() {
    checkCudaCall(cuStreamQuery(_obj));  // unsuccessful result throws cu::Error
  }

  void synchronize() { checkCudaCall(cuStreamSynchronize(_obj)); }

  void wait(Event &event) { checkCudaCall(cuStreamWaitEvent(_obj, event, 0)); }

  void addCallback(CUstreamCallback callback, void *userData,
                   unsigned int flags = 0) {
    checkCudaCall(cuStreamAddCallback(_obj, callback, userData, flags));
  }

  void record(Event &event) { checkCudaCall(cuEventRecord(event, _obj)); }

#if !defined(__HIP__)
  void batchMemOp(unsigned count, CUstreamBatchMemOpParams *paramArray,
                  unsigned flags) {
    checkCudaCall(cuStreamBatchMemOp(_obj, count, paramArray, flags));
  }
#endif

  void waitValue32(CUdeviceptr addr, cuuint32_t value, unsigned flags) const {
    checkCudaCall(cuStreamWaitValue32(_obj, addr, value, flags));
  }

  void writeValue32(CUdeviceptr addr, cuuint32_t value, unsigned flags) {
    checkCudaCall(cuStreamWriteValue32(_obj, addr, value, flags));
  }
};

inline void Event::record(Stream &stream) {
  checkCudaCall(cuEventRecord(_obj, stream._obj));
}
}  // namespace cu

#endif
