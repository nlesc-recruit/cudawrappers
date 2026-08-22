#if !defined CU_WRAPPER_H
#define CU_WRAPPER_H

#include "cudawrappers/cu_backend.hpp"

#include <array>
#include <cstddef>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// When CUDA headers are available (and not building for HIP), include
// <cuda.h> to get the real CUDA types. Otherwise, define manual types
// compatible with the backend abstraction (for HIP-only or headerless builds).
#if __has_include(<cuda.h>) && !defined(__HIP__)
#include <cuda.h>
#if !defined(CU_GRAPH_DEFAULT)
#define CU_GRAPH_DEFAULT 0
#endif
#if !defined(CU_GREEN_CTX_DEFAULT_STREAM)
#define CU_GREEN_CTX_DEFAULT_STREAM 0x01
#endif
#if !defined(CUDA_MEM_HANDLE_TYPE_FABRIC)
#define CUDA_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x20)
#endif
#else
// Manual types compatible with CUDA driver API (for HIP and headerless builds).
// cu.hpp does not include macros.hpp — those mappings are for end-user code.
// cu.hpp only uses its own types and the backend handles HIP conversion.
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef void* CUevent;
typedef unsigned long long CUdeviceptr;
typedef void* CUmemoryPool;
typedef void* CUgraph;
typedef void* CUgraphExec;
typedef void* CUgraphNode;
typedef void* CUarray;
typedef void* CUgreenCtx;
struct CUdevResource;
typedef const CUdevResource* CUdevResourceDesc;

struct CUuuid {
  char bytes[16];
};

struct CUdevprop {
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int sharedMemPerBlock;
  int totalConstantMemory;
  int SIMDWidth;
  int memPitch;
  int regsPerBlock;
  int clockRate;
  int textureAlign;
};

typedef void (*CUhostFn)(void*);

#ifndef CUDA_SUCCESS
constexpr CUresult CUDA_SUCCESS = 0;
#endif
constexpr CUresult CUDA_ERROR_NOT_FOUND = 500;

// --- Shared type definitions (used by both HIP and non-HIP builds) ---

enum CUdevice_attribute {
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
  CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
  CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
  CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
  CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
  CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
  CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
  CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
  CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 9,
  CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
  CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
  CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 35,
  CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 67,
  CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 39,
  CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 40,
  CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
  CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 6,
  CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 31,
  CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 32,
  CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
  CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
  CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 15,
  CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 53,
  CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 68,
  CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 42,
  CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 43,
  CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 44,
  CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 45,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 46,
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 47,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 48,
  CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 58,
  CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 59,
  CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 56,
  CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 49,
  CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 50,
  CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 51,
  CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 54,
  CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 55,
  CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 52,
  CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 93,
};

enum CUfunc_cache {
  CU_FUNC_CACHE_PREFER_NONE = 0,
  CU_FUNC_CACHE_PREFER_SHARED = 1,
  CU_FUNC_CACHE_PREFER_L1 = 2,
  CU_FUNC_CACHE_PREFER_EQUAL = 3
};

enum CUfunction_attribute {
  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 0,
  CU_FUNC_ATTRIBUTE_MAX_SHARED_MEMORY_BYTES = 1,
  CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
  CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
  CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
  CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
  CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
  CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 8,
  CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
  CU_FUNC_ATTRIBUTE_MAX = 10,
};

enum CUmemorytype {
  CU_MEMORYTYPE_HOST = 1,
  CU_MEMORYTYPE_DEVICE = 2,
  CU_MEMORYTYPE_ARRAY = 3,
  CU_MEMORYTYPE_UNIFIED = 4,
};

enum CUlimit {
  CU_LIMITStackSize = 0,
  CU_LIMITPrintfFifoSize = 1,
  CU_LIMITMallocHeapSize = 2,
};

enum CUarray_format {
  CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
  CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
  CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
  CU_AD_FORMAT_SIGNED_INT8 = 0x08,
  CU_AD_FORMAT_SIGNED_INT16 = 0x09,
  CU_AD_FORMAT_SIGNED_INT32 = 0x0a,
  CU_AD_FORMAT_HALF = 0x10,
  CU_AD_FORMAT_FLOAT = 0x20,
};

enum CUpointer_attribute {
  CU_POINTER_ATTRIBUTE_CONTEXT = 1,
  CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
  CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
  CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
  CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
  CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
  CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
  CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
  CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 9,
  CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 10,
  CU_POINTER_ATTRIBUTE_RANGE_SIZE = 11,
  CU_POINTER_ATTRIBUTE_MAPPED = 12,
  CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 13,
  CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 14,
  CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 15,
  CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 16,
  CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 17,
};

enum CUgraphDebugDot_flags {
  CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE = 1,
  CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = 2,
  CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = 4,
  CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = 8,
  CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = 16,
  CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = 32,
  CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = 64,
  CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = 128,
  CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = 256,
  CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = 512,
  CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES = 1024,
  CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS = 2048,
  CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS = 4096,
  CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS = 8192,
  CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO = 16384,
  CU_GRAPH_DEBUG_DOT_FLAGS_CONDITIONAL_NODE_PARAMS = 32768,
};

enum CUgraphMem_attribute {
  CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT = 0,
  CU_GRAPH_MEM_ATTR_USED_MEM_HIGH = 1,
  CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT = 2,
  CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH = 3,
};

enum CUexecAffinityType { CU_EXEC_AFFINITY_TYPE_SM_COUNT = 0 };

enum CUdevice_P2PAttribute {
  CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 0,
  CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 1,
  CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 2,
  CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = 3,
  CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = 4,
};

constexpr unsigned int CU_EVENT_DEFAULT = 0x00;
constexpr unsigned int CU_EVENT_BLOCKING_SYNC = 0x01;
constexpr unsigned int CU_EVENT_DISABLE_TIMING = 0x02;
constexpr unsigned int CU_EVENT_INTERPROCESS = 0x04;

constexpr unsigned int CU_STREAM_DEFAULT = 0x00;
constexpr unsigned int CU_STREAM_NON_BLOCKING = 0x01;

constexpr unsigned int CU_MEMHOSTALLOC_PORTABLE = 0x01;
constexpr unsigned int CU_MEMHOSTALLOC_DEVICEMAP = 0x02;
constexpr unsigned int CU_MEMHOSTALLOC_WRITECOMBINED = 0x04;

constexpr unsigned int CU_MEMHOSTREGISTER_PORTABLE = 0x01;
constexpr unsigned int CU_MEMHOSTREGISTER_DEVICEMAP = 0x02;
constexpr unsigned int CU_MEMHOSTREGISTER_READ_ONLY = 0x04;

constexpr unsigned int CU_MEM_ATTACH_GLOBAL = 0x01;
constexpr unsigned int CU_MEM_ATTACH_SINGLE = 0x02;
constexpr unsigned int CU_MEM_ATTACH_HOST = 0x04;

constexpr unsigned int CU_CTX_SCHED_BLOCKING_SYNC = 0x04;
constexpr unsigned int CU_CTX_SCHED_SPIN = 0x01;
constexpr unsigned int CU_CTX_SCHED_YIELD = 0x02;
constexpr unsigned int CU_CTX_SCHED_AUTO = 0x00;
constexpr unsigned int CU_CTX_SCHED_MASK = 0x07;
constexpr unsigned int CU_CTX_MAP_HOST = 0x08;
constexpr unsigned int CU_CTX_LMEM_RESIZE_TO_MAX = 0x10;

constexpr unsigned int CU_GREEN_CTX_DEFAULT_STREAM = 0x01;

enum CUmemAllocationType {
  CU_MEM_ALLOCATION_TYPE_PINNED = 0x01,
};

enum CUmemAllocationHandleType {
  CU_MEM_HANDLE_TYPE_NONE = 0x00,
  CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x01,
  CU_MEM_HANDLE_TYPE_WIN32 = 0x02,
  CU_MEM_HANDLE_TYPE_WIN32_KMT = 0x04,
  CU_MEM_HANDLE_TYPE_MAX = 0x7FFFFFFF,
};

constexpr int CU_MEM_LOCATION_TYPE_DEVICE = 1;

struct CUmemLocation {
  int type;
  int id;
};

struct CUmemAllocationProp {
  int type;
  int handleTypes;
  CUmemLocation location;
  size_t allocGranularity;
  int flags;
};

struct CUdevResource {
  char _opaque[128];
};

constexpr int CU_DEV_RESOURCE_TYPE_SM = 0;
constexpr int CU_DEV_RESOURCE_TYPE_COMPUTE = 1;
constexpr int CU_DEV_RESOURCE_TYPE_MEMORY = 2;
constexpr int CU_DEV_RESOURCE_TYPE_NVLINK = 3;
constexpr int CU_DEV_RESOURCE_TYPE_MAX = 0x7FFFFFFF;

struct CUDA_MEMCPY3D {
  unsigned int srcXInBytes;
  unsigned int srcY;
  unsigned int srcZ;
  unsigned int srcLOD;
  int srcMemoryType;
  const void* srcHost;
  CUdeviceptr srcDevice;
  CUarray srcArray;
  unsigned int srcPitch;
  unsigned int srcHeight;
  unsigned int dstXInBytes;
  unsigned int dstY;
  unsigned int dstZ;
  unsigned int dstLOD;
  int dstMemoryType;
  void* dstHost;
  CUdeviceptr dstDevice;
  CUarray dstArray;
  unsigned int dstPitch;
  unsigned int dstHeight;
  unsigned int WidthInBytes;
  unsigned int Height;
  unsigned int Depth;
};

struct CUDA_KERNEL_NODE_PARAMS {
  CUfunction func;
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  void** kernelParams;
  void** extra;
};

struct CUDA_HOST_NODE_PARAMS {
  void (*fn)(void*);
  void* userData;
};

struct CUDA_MEM_ALLOC_NODE_PARAMS {
  CUmemAllocationProp poolProps;
  size_t bytesize;
  CUdeviceptr dptr;
};

struct CUDA_ARRAY_DESCRIPTOR {
  unsigned int Width;
  unsigned int Height;
  CUarray_format Format;
  unsigned int NumChannels;
};

struct CUDA_ARRAY3D_DESCRIPTOR {
  unsigned int Width;
  unsigned int Height;
  unsigned int Depth;
  CUarray_format Format;
  unsigned int NumChannels;
  unsigned int Flags;
};

enum CUjit_option {
  CU_JIT_MAX_REGISTERS = 0,
  CU_JIT_THREADS_PER_BLOCK = 1,
  CU_JIT_WALL_TIME = 2,
  CU_JIT_INFO_LOG_BUFFER = 3,
  CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
  CU_JIT_ERROR_LOG_BUFFER = 5,
  CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
  CU_JIT_OPTIMIZATION_LEVEL = 7,
  CU_JIT_TARGET = 8,
  CU_JIT_TARGET_FROM_CUCONTEXT = 9,
  CU_JIT_FALLBACK_STRATEGY = 10,
  CU_JIT_GENERATE_DEBUG_INFO = 11,
  CU_JIT_LOG_VERBOSE = 12,
  CU_JIT_GENERATE_LINE_INFO = 13,
  CU_JIT_CACHE_MODE = 14,
  CU_JIT_NEW_SM3X_OPT = 15,
  CU_JIT_FAST_COMPILE = 16,
  CU_JIT_NUM_OPTIONS = 17,
};

constexpr unsigned int CU_GRAPH_DEFAULT = 0;

#endif  // __has_include(<cuda.h>) && !defined(__HIP__)

// When real CUDA types are used (CUctx_st*, etc.), the backend expects void*.
// These helpers safely convert between the two representations.
namespace cu_backend_cast {
template <typename T>
inline void** toVoidPP(T& ptr) {
  return reinterpret_cast<void**>(&ptr);
}
template <typename T>
inline void* toVoidP(T ptr) {
  return reinterpret_cast<void*>(ptr);
}
template <typename T>
inline T fromVoidP(void* ptr) {
  return reinterpret_cast<T>(ptr);
}
}  // namespace cu_backend_cast

namespace cu {

class Error : public std::exception {
 public:
  explicit Error(CUresult result);
  const char* what() const noexcept;
  operator CUresult() const;

 private:
  CUresult _result;
};

void checkCudaCall(int result);

void init(unsigned flags = 0);
int driverGetVersion();
const char* getErrorName(CUresult result);
void memcpyHtoD(CUdeviceptr dst, const void* src, size_t size);
void memcpyDtoH(void* dst, CUdeviceptr src, size_t size);
void pointerSetAttribute(const void* value, CUpointer_attribute attribute,
                          CUdeviceptr ptr);

template <typename T>
class Wrapper {
 public:
  operator T() const { return _obj; }
  operator T() { return _obj; }
  bool operator==(const Wrapper<T>& other) { return _obj == other._obj; }
  bool operator!=(const Wrapper<T>& other) { return _obj != other._obj; }
  Wrapper& operator=(const Wrapper<T>& other) {
    _obj = other._obj;
    manager = other.manager;
    _backendIdx = other._backendIdx;
    return *this;
  }
  Wrapper& operator=(Wrapper<T>&& other) {
    _obj = other._obj;
    manager = std::move(other.manager);
    _backendIdx = other._backendIdx;
    other._obj = 0;
    return *this;
  }
  int getBackendIdx() const { return _backendIdx; }

  void** ptr() { return cu_backend_cast::toVoidPP(_obj); }

 protected:
  Wrapper() = default;
  Wrapper(const Wrapper<T>& other)
      : _obj(other._obj), manager(other.manager), _backendIdx(other._backendIdx) {}
  Wrapper(Wrapper<T>&& other)
      : _obj(other._obj), manager(std::move(other.manager)), _backendIdx(other._backendIdx) {
    other._obj = 0;
  }
  explicit Wrapper(T& obj) : _obj(obj) {}

  T _obj{};
  std::shared_ptr<T> manager;
  int _backendIdx{0};
};

class Device : public Wrapper<CUdevice> {
 public:
  explicit Device(unsigned int ordinal);
  explicit Device(CUdevice device);

  int getAttribute(CUdevice_attribute attribute) const;
  template <CUdevice_attribute attribute>
  int getAttribute() const {
    return getAttribute(attribute);
  }
  static int getCount();
  static int getCount(int backendIdx);
  static int getDeviceOffset(int backendIdx);
  std::string getName() const;
  std::string getUuid() const;
  std::string getArch() const;
  void getComputeCapability(int& major, int& minor) const;
  static Device getByPCIBusId(const std::string& pciBusId);
  std::string getPCIBusId() const;
  void getDefaultMemPool(CUmemoryPool& pool) const;
  void getMemPool(CUmemoryPool& pool) const;
  void setMemPool(CUmemoryPool pool) const;
  size_t totalMem() const;
  int getOrdinal() const;

#if !defined(__HIP__)
  size_t getTexture1DLinearMaxWidth(CUarray_format format,
                                    unsigned numChannels) const;
  void getExecAffinitySupport(int& pi, CUexecAffinityType type) const;
  void getProperties(CUdevprop& prop) const;
  void getDevResource(CUdevResource& resource, CUdevResourceType type) const;
#endif

 private:
  int _ordinal{-1};
};

class Context;

class HostMemory : public Wrapper<void*> {
 public:
  explicit HostMemory(size_t size, unsigned int flags = 0);
  explicit HostMemory(void* ptr, size_t size, unsigned int flags = 0);
  template <typename T>
  operator T*() {
    return static_cast<T*>(_obj);
  }
  size_t size() const;

 protected:
  size_t _size{0};
};

class UnmanagedMemory : public HostMemory {
 public:
  UnmanagedMemory(void* ptr, size_t size);
};

class Array : public Wrapper<CUarray> {
 public:
  Array(unsigned width, CUarray_format format, unsigned numChannels);
  Array(unsigned width, unsigned height, CUarray_format format,
        unsigned numChannels);
  Array(unsigned width, unsigned height, unsigned depth, CUarray_format format,
        unsigned numChannels);
  explicit Array(CUarray& array);
};

class Module : public Wrapper<CUmodule> {
 public:
  typedef std::map<CUjit_option, void*> optionmap_t;
  explicit Module(const char* file_name);
  explicit Module(const void* data);
  explicit Module(const void* image, Module::optionmap_t& options);
  explicit Module(CUmodule& module);
  CUdeviceptr getGlobal(const char* name) const;
};

class Function : public Wrapper<CUfunction> {
 public:
  Function(const Module& module, const char* name);
  explicit Function(CUfunction& function);
  int getAttribute(CUfunction_attribute attribute) const;
  void setAttribute(CUfunction_attribute attribute, int value);
  int occupancyMaxActiveBlocksPerMultiprocessor(int blockSize,
                                                size_t dynamicSMemSize);
  void setCacheConfig(CUfunc_cache config);
};

class Stream;

class Event : public Wrapper<CUevent> {
 public:
  explicit Event(unsigned int flags = CU_EVENT_DEFAULT);
  explicit Event(CUevent& event);
  float elapsedTime(const Event& start) const;
  void query() const;
  void record();
  void record(Stream& stream);
  void record(Stream& stream, unsigned int flags);
  void synchronize();
};

class DeviceMemory : public Wrapper<CUdeviceptr> {
 public:
  explicit DeviceMemory(size_t size,
                        CUmemorytype type = CU_MEMORYTYPE_DEVICE,
                        unsigned int flags = 0);
  explicit DeviceMemory(CUdeviceptr ptr);
  explicit DeviceMemory(CUdeviceptr ptr, size_t size);
  explicit DeviceMemory(const HostMemory& hostMemory);
  explicit DeviceMemory(const DeviceMemory& other, size_t offset, size_t size);
  void memset(unsigned char value, size_t size);
  void memset(unsigned short value, size_t size);
  void memset(unsigned int value, size_t size);
  void memset2D(unsigned char value, size_t pitch, size_t width, size_t height);
  void memset2D(unsigned short value, size_t pitch, size_t width,
                size_t height);
  void memset2D(unsigned int value, size_t pitch, size_t width, size_t height);
  void zero(size_t size);
  const void* parameter() const;
  template <typename T>
  operator T*() {
    return reinterpret_cast<T*>(_obj);
  }
  template <typename T>
  operator T*() const {
    return reinterpret_cast<T*>(_obj);
  }
  size_t size() const;

 private:
  size_t _size{0};
};

class GraphNode : public Wrapper<CUgraphNode> {
 public:
  GraphNode() = default;
  GraphNode(CUgraphNode& node);
  CUgraphNode* getNode();
};

class GraphKernelNodeParams : public Wrapper<CUDA_KERNEL_NODE_PARAMS> {
 public:
  GraphKernelNodeParams(const Function& function, unsigned gridDimX,
                        unsigned gridDimY, unsigned gridDimZ,
                        unsigned blockDimX, unsigned blockDimY,
                        unsigned blockDimZ, unsigned sharedMemBytes,
                        const std::vector<const void*>& kernelParams);
  const void* parameter() const { return &_obj; }
};

class GraphHostNodeParams : public Wrapper<CUDA_HOST_NODE_PARAMS> {
 public:
  GraphHostNodeParams(void (*fn)(void*), void* data);
  const void* parameter() const { return &_obj; }
};

class GraphDevMemAllocNodeParams
    : public Wrapper<CUDA_MEM_ALLOC_NODE_PARAMS> {
 public:
  GraphDevMemAllocNodeParams(const Device& dev, size_t size);
  const CUdeviceptr& getDevPtr() const;
  const void* parameter();
  const DeviceMemory getDeviceMemory();
};

class GraphMemCopyToDeviceNodeParams : public Wrapper<CUDA_MEMCPY3D> {
 public:
  GraphMemCopyToDeviceNodeParams(const CUdeviceptr& dst, const void* src,
                                 size_t size_x, size_t size_y, size_t size_z,
                                 size_t element_size, size_t pitch = 0);
  const void* parameter() const { return &_obj; }
};

class GraphMemCopyToHostNodeParams : public Wrapper<CUDA_MEMCPY3D> {
 public:
  GraphMemCopyToHostNodeParams(void* host, const CUdeviceptr& src,
                               size_t size_x, size_t size_y, size_t size_z,
                               size_t element_size, size_t pitch = 0);
  const void* parameter() const { return &_obj; }
};

#if !defined(__HIP__)
class GreenContext;
#endif

class Context : public Wrapper<CUcontext> {
 public:
  Context(int flags, Device& device);
  unsigned getApiVersion() const;
  static CUfunc_cache getCacheConfig();
  static void setCacheConfig(CUfunc_cache config);
  Context getCurrent();
  void setCurrent() const;
  Context popCurrent();
  void pushCurrent();
  void enablePeerAccess(Context& peerContext, unsigned int flags = 0);
  void disablePeerAccess(Context& peerContext);
  Device getDevice();
  size_t getFreeMemory() const;
  size_t getTotalMemory() const;
  static size_t getLimit(CUlimit limit);
  template <CUlimit limit>
  static size_t getLimit() {
    return getLimit(limit);
  }
  static void setLimit(CUlimit limit, size_t value);
  template <CUlimit limit>
  static void setLimit(size_t value) {
    setLimit(limit, value);
  }
  static void synchronize();

#if !defined(__HIP__)
  void getDevResource(CUdevResource& resource, CUdevResourceType type) const;
  static Context fromGreenCtx(GreenContext& greenContext);
#endif

 private:
  Context(CUcontext context, Device& device);
  Device* _device{nullptr};
};

class GraphExec;

#if !defined(__HIP__)
class GreenContext : public Wrapper<CUgreenCtx> {
 public:
  GreenContext(CUdevResourceDesc desc, Device& device,
               unsigned int flags = CU_GREEN_CTX_DEFAULT_STREAM);
  void getDevResource(CUdevResource& resource, CUdevResourceType type) const;
  Stream createStream(unsigned int flags = 0, int priority = 0) const;
  void recordEvent(Event& event) const;
  void waitEvent(Event& event) const;
  Device& getDevice();
  const Device& getDevice() const;

 private:
  Device& _device;
};
#endif

class Graph : public Wrapper<CUgraph> {
 public:
  Graph(Context& context, CUgraph& graph);
  Graph(Context& context, unsigned int flags = CU_GRAPH_DEFAULT);
  void addKernelNode(GraphNode& node,
                     const std::vector<CUgraphNode>& dependencies,
                     GraphKernelNodeParams& params);
  void addHostNode(GraphNode& node,
                   const std::vector<CUgraphNode>& dependencies,
                   GraphHostNodeParams& params);
  void addDevMemFreeNode(GraphNode& node,
                         const std::vector<CUgraphNode>& dependencies,
                         const CUdeviceptr& devPtr);
  void addMemAllocNode(GraphNode& node,
                       const std::vector<CUgraphNode>& dependencies,
                       GraphDevMemAllocNodeParams& params);
  void addMemCpyNode(GraphNode& node,
                     const std::vector<CUgraphNode>& dependencies,
                     GraphMemCopyToDeviceNodeParams& params);
  void addMemCpyNode(GraphNode& node,
                     const std::vector<CUgraphNode>& dependencies,
                     GraphMemCopyToHostNodeParams& params);
  void debugDotPrint(std::string path,
                     CUgraphDebugDot_flags flags = CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE);
  CUgraphExec instantiateWithFlags(unsigned int flags = CU_GRAPH_DEFAULT);

 private:
  Context* _context{nullptr};
};

class GraphExec : public Wrapper<CUgraphExec> {
 public:
  explicit GraphExec(CUgraphExec& graphExec);
  explicit GraphExec(GraphExec& graphExec) = default;
  explicit GraphExec(const Graph& graph,
                     unsigned int flags = CU_GRAPH_DEFAULT);
};

class Stream : public Wrapper<CUstream> {
 public:
  explicit Stream(unsigned int flags = CU_STREAM_DEFAULT);
  explicit Stream(CUstream stream);
  Stream(CUstream stream, bool takeOwnership);
  DeviceMemory memAllocAsync(size_t size);
  void memFreeAsync(DeviceMemory& devMem);
  void memcpyHtoHAsync(void* dstPtr, const void* srcPtr, size_t size);
  void memcpyHtoDAsync(DeviceMemory& dst, const void* src, size_t size);
  void memcpyHtoD2DAsync(DeviceMemory& dst, size_t dpitch, const void* src,
                         size_t spitch, size_t width, size_t height);
  void memcpyDtoH2DAsync(void* dst, size_t dpitch, const DeviceMemory& src,
                         size_t spitch, size_t width, size_t height);
  void memcpyHtoDAsync(CUdeviceptr dst, const void* src, size_t size);
  void memcpyDtoHAsync(void* dst, const DeviceMemory& src, size_t size);
  void memcpyDtoHAsync(void* dst, CUdeviceptr src, size_t size);
  void memcpyDtoDAsync(DeviceMemory& dst, const DeviceMemory& src, size_t size);
  void memPrefetchAsync(DeviceMemory& devMem, size_t size);
  void memPrefetchAsync(DeviceMemory& devMem, size_t size, Device& device);
  void memsetAsync(DeviceMemory& dst, unsigned char value, size_t count);
  void memsetAsync(DeviceMemory& dst, unsigned short value, size_t count);
  void memsetAsync(DeviceMemory& dst, unsigned int value, size_t count);
  void memset2DAsync(DeviceMemory& dst, unsigned char value, size_t pitch,
                     size_t width, size_t height);
  void memset2DAsync(DeviceMemory& dst, unsigned short value, size_t pitch,
                     size_t width, size_t height);
  void memset2DAsync(DeviceMemory& dst, unsigned int value, size_t pitch,
                     size_t width, size_t height);
  void zero(DeviceMemory& dst, size_t size);
  void zero2D(DeviceMemory& dst, size_t pitch, size_t width, size_t height);
  void launchKernel(Function& function, unsigned gridX, unsigned gridY,
                    unsigned gridZ, unsigned blockX, unsigned blockY,
                    unsigned blockZ, unsigned sharedMemBytes,
                    const std::vector<const void*>& parameters);
  void graphLaunch(GraphExec& graphExec);
  void query();
  void synchronize();
  void wait(Event& event);
  void record(Event& event);
  void record(Event& event, unsigned int flags);
  void launchHostFunc(CUhostFn fn, void* userData = nullptr);
#if !defined(__HIP__)
  void getDevResource(CUdevResource& resource, CUdevResourceType type) const;
#endif
};

#if !defined(__HIP__)
inline void devResourceGenerateDesc(CUdevResourceDesc* phDesc,
                                    CUdevResource* resources,
                                    unsigned int nbResources) {
  Backend& b = getBackend();
  checkCudaCall(b.devResourceGenerateDesc(
      reinterpret_cast<void**>(const_cast<CUdevResourceDesc*>(phDesc)),
      const_cast<CUdevResource*>(resources), nbResources));
}
inline void devSmResourceSplitByCount(
    CUdevResource* result, unsigned int* nbGroups, const CUdevResource* input,
    CUdevResource* remaining, unsigned int useFlags, unsigned int minCount) {
  Backend& b = getBackend();
  checkCudaCall(b.devSmResourceSplitByCount(
      result, nbGroups, input, remaining, useFlags, minCount));
}
#endif

// ============================================================================
// Inline implementations (header-only library)
// ============================================================================

// --- Error ---

inline Error::Error(CUresult result) : _result(result) {}
inline const char* Error::what() const noexcept {
  Backend& b = getBackend(0);
  if (b.getErrorString) {
    const char* str{};
    b.getErrorString(static_cast<CUresult_b>(_result), &str);
    if (str) return str;
  }
  return "unknown error";
}
inline Error::operator CUresult() const { return _result; }

// --- Free functions ---

inline void checkCudaCall(int result) {
  if (result != CUDA_SUCCESS) throw Error(static_cast<CUresult>(result));
}

inline void init(unsigned flags) {
  for (size_t i = 0; i < getBackendCount(); ++i) {
    Backend& b = getBackend(i);
    if (b.init) checkCudaCall(b.init(flags));
  }
}

inline int driverGetVersion() {
  int version{};
  checkCudaCall(getBackend(0).driverGetVersion(&version));
  return version;
}

inline const char* getErrorName(CUresult result) {
  Backend& b = getBackend(0);
  const char* str{};
  if (b.getErrorName) b.getErrorName(result, &str);
  return str ? str : "unknown";
}

inline void memcpyHtoD(CUdeviceptr dst, const void* src, size_t size) {
  checkCudaCall(getBackend(0).memcpyHtoD(dst, src, size));
}

inline void memcpyDtoH(void* dst, CUdeviceptr src, size_t size) {
  checkCudaCall(getBackend(0).memcpyDtoH(dst, src, size));
}

inline void pointerSetAttribute(const void* value, CUpointer_attribute attribute,
                                 CUdeviceptr ptr) {
  checkCudaCall(getBackend(0).pointerSetAttribute(value, attribute, ptr));
}

// --- Device ---

inline int Device::getCount(int backendIdx) {
  int nrDevices{};
  checkCudaCall(getBackend(backendIdx).deviceGetCount(&nrDevices));
  return nrDevices;
}

inline int Device::getDeviceOffset(int backendIdx) {
  int offset = 0;
  for (int i = 0; i < backendIdx; ++i) {
    offset += getCount(i);
  }
  return offset;
}

inline int Device::getCount() {
  int total = 0;
  for (size_t i = 0; i < getBackendCount(); ++i) {
    total += getCount(i);
  }
  return total;
}

inline Device::Device(unsigned int ordinal) : _ordinal(ordinal) {
  int globalOffset = 0;
  for (size_t i = 0; i < getBackendCount(); ++i) {
    int count = getCount(i);
    if (ordinal < globalOffset + count) {
      _backendIdx = static_cast<int>(i);
      int localOrdinal = ordinal - globalOffset;
      checkCudaCall(getBackend(_backendIdx).deviceGet(&_obj, localOrdinal));
      return;
    }
    globalOffset += count;
  }
  throw std::runtime_error("Device ordinal out of range");
}

inline Device::Device(CUdevice device) : Wrapper<CUdevice>(device), _ordinal(-1) {
  for (size_t i = 0; i < getBackendCount(); ++i) {
    int count = getCount(i);
    for (int ordinal = 0; ordinal < count; ordinal++) {
      CUdevice current_device;
      checkCudaCall(getBackend(i).deviceGet(&current_device, ordinal));
      if (current_device == device) {
        _backendIdx = static_cast<int>(i);
        _ordinal = ordinal;
        return;
      }
    }
  }
}

inline int Device::getAttribute(CUdevice_attribute attribute) const {
  int value{};
  int r = getBackend(_backendIdx).deviceGetAttribute(&value, attribute, _obj);
  checkCudaCall(r);
  return value;
}

inline std::string Device::getName() const {
  const size_t max_device_name_length{64};
  std::array<char, max_device_name_length> name{};
  int r = getBackend(_backendIdx).deviceGetName(name.data(), name.size(), _obj);
  checkCudaCall(r);
  return {name.data()};
}

inline std::string Device::getUuid() const {
  CUuuid uuid;
  checkCudaCall(getBackend(_backendIdx).deviceGetUuid(
      reinterpret_cast<CUuuid_b*>(&uuid), _obj));
  std::stringstream result;
  result << "GPU";
  for (int i = 0; i < 16; ++i) {
    if (i == 0 || i == 4 || i == 6 || i == 8 || i == 10) result << "-";
    result << std::hex << std::setfill('0') << std::setw(2)
           << static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[i]));
  }
  return result.str();
}

inline std::string Device::getArch() const {
  const size_t max_arch_length{64};
  std::array<char, max_arch_length> arch{};
  int r = getBackend(_backendIdx).deviceGetArchName(arch.data(), arch.size(), _obj);
  checkCudaCall(r);
  return {arch.data()};
}

inline void Device::getComputeCapability(int& major, int& minor) const {
  major = getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>();
  minor = getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();
}

inline Device Device::getByPCIBusId(const std::string& pciBusId) {
  for (size_t i = 0; i < getBackendCount(); ++i) {
    CUdevice device{};
    int result = getBackend(i).deviceGetByPCIBusId(&device, pciBusId.c_str());
    if (result == CUDA_SUCCESS) {
      Device dev(device);
      return dev;
    }
  }
  throw Error(CUDA_ERROR_NOT_FOUND);
}

inline std::string Device::getPCIBusId() const {
  const size_t pciBusIdLength{64};
  std::array<char, pciBusIdLength> pciBusId{};
  checkCudaCall(
      getBackend(_backendIdx).deviceGetPCIBusId(pciBusId.data(), pciBusId.size(), _obj));
  return std::string(pciBusId.data());
}

inline void Device::getDefaultMemPool(CUmemoryPool& pool) const {
  checkCudaCall(getBackend(_backendIdx).deviceGetDefaultMemPool(
      cu_backend_cast::toVoidPP(pool), _obj));
}

inline void Device::getMemPool(CUmemoryPool& pool) const {
  checkCudaCall(getBackend(_backendIdx).deviceGetMemPool(
      cu_backend_cast::toVoidPP(pool), _obj));
}

inline void Device::setMemPool(CUmemoryPool pool) const {
  checkCudaCall(getBackend(_backendIdx).deviceSetMemPool(
      _obj, cu_backend_cast::toVoidP(pool)));
}

inline size_t Device::totalMem() const {
  size_t total_mem{};
  checkCudaCall(getBackend(_backendIdx).deviceTotalMem(&total_mem, _obj));
  return total_mem;
}

inline int Device::getOrdinal() const { return _ordinal; }

// --- Context ---

inline Context::Context(int flags, Device& device) : _device(&device) {
  _backendIdx = device.getBackendIdx();
  checkCudaCall(
      getBackend(_backendIdx).ctxCreate(ptr(), flags, device));
  int bIdx = _backendIdx;
  manager = std::shared_ptr<CUcontext>(new CUcontext(_obj),
                                        [bIdx](CUcontext* ptr) {
                                          if (*ptr)
                                            getBackend(bIdx).ctxDestroy(*ptr);
                                          delete ptr;
                                        });
}

inline Context::Context(CUcontext context, Device& device)
    : Wrapper<CUcontext>(context), _device(&device) {
  _backendIdx = device.getBackendIdx();
}

inline unsigned Context::getApiVersion() const {
  unsigned version{};
  checkCudaCall(getBackend(_backendIdx).ctxGetApiVersion(_obj, &version));
  return version;
}

inline CUfunc_cache Context::getCacheConfig() {
  int config{};
  checkCudaCall(getBackend(0).ctxGetCacheConfig(&config));
  return static_cast<CUfunc_cache>(config);
}

inline void Context::setCacheConfig(CUfunc_cache config) {
  checkCudaCall(getBackend(0).ctxSetCacheConfig(config));
}

inline Context Context::getCurrent() {
  for (size_t i = 0; i < getBackendCount(); ++i) {
    CUcontext ctx{};
    int result = getBackend(i).ctxGetCurrent(cu_backend_cast::toVoidPP(ctx));
    if (result == CUDA_SUCCESS && ctx) {
      Device dev(0);
      for (size_t j = 0; j < getBackendCount(); ++j) {
        int count = Device::getCount(j);
        for (int ordinal = 0; ordinal < count; ordinal++) {
          CUdevice current_device;
          checkCudaCall(getBackend(j).deviceGet(&current_device, ordinal));
          Device d(current_device);
          if (d.getBackendIdx() == static_cast<int>(j)) {
            return Context(ctx, d);
          }
        }
      }
    }
  }
  return Context(0, *new Device(0));
}

inline void Context::setCurrent() const {
  checkCudaCall(getBackend(_backendIdx).ctxSetCurrent(_obj));
}

inline Context Context::popCurrent() {
  for (size_t i = 0; i < getBackendCount(); ++i) {
    CUcontext ctx{};
    int result = getBackend(i).ctxPopCurrent(cu_backend_cast::toVoidPP(ctx));
    if (result == CUDA_SUCCESS && ctx) {
      Device dev(0);
      return Context(ctx, dev);
    }
  }
  return Context(0, *new Device(0));
}

inline void Context::pushCurrent() {
  checkCudaCall(getBackend(_backendIdx).ctxPushCurrent(_obj));
}

inline void Context::enablePeerAccess(Context& peerContext, unsigned int flags) {
  checkCudaCall(getBackend(_backendIdx).ctxEnablePeerAccess(peerContext._obj, flags));
}

inline void Context::disablePeerAccess(Context& peerContext) {
  checkCudaCall(getBackend(_backendIdx).ctxDisablePeerAccess(peerContext._obj));
}

inline Device Context::getDevice() {
  CUdevice dev{};
  checkCudaCall(getBackend(_backendIdx).ctxGetDevice(&dev));
  return Device(dev);
}

inline size_t Context::getFreeMemory() const {
  size_t free{};
  size_t total{};
  checkCudaCall(getBackend(_backendIdx).memGetInfo(&free, &total));
  return free;
}

inline size_t Context::getTotalMemory() const {
  size_t free{};
  size_t total{};
  checkCudaCall(getBackend(_backendIdx).memGetInfo(&free, &total));
  return total;
}

inline size_t Context::getLimit(CUlimit limit) {
  size_t value{};
  checkCudaCall(getBackend(0).ctxGetLimit(&value, limit));
  return value;
}

inline void Context::setLimit(CUlimit limit, size_t value) {
  checkCudaCall(getBackend(0).ctxSetLimit(limit, value));
}

inline void Context::synchronize() {
  checkCudaCall(getBackend(0).ctxSynchronize());
}

// --- HostMemory ---

inline HostMemory::HostMemory(size_t size, unsigned int flags) : _size(size) {
  checkCudaCall(getBackend(_backendIdx).memHostAlloc(&_obj, size, flags));
  int bIdx = _backendIdx;
  manager = std::shared_ptr<void*>(new void*(_obj), [bIdx](void** ptr) {
    getBackend(bIdx).memHostFree(*ptr);
    delete ptr;
  });
}

inline HostMemory::HostMemory(void* ptr, size_t size, unsigned int flags)
    : _size(size) {
  _obj = ptr;
  checkCudaCall(getBackend(_backendIdx).memHostRegister(ptr, size, flags));
  int bIdx = _backendIdx;
  manager = std::shared_ptr<void*>(new void*(_obj), [bIdx](void** ptr) {
    getBackend(bIdx).memHostUnregister(*ptr);
    delete ptr;
  });
}

inline size_t HostMemory::size() const { return _size; }

// --- UnmanagedMemory ---

inline UnmanagedMemory::UnmanagedMemory(void* ptr, size_t size)
    : HostMemory(ptr, size, 0) {}

// --- Array ---

inline Array::Array(unsigned width, CUarray_format format, unsigned numChannels) {
  CUDA_ARRAY_DESCRIPTOR desc{};
  desc.Width = width;
  desc.Height = 0;
  desc.Format = format;
  desc.NumChannels = numChannels;
  checkCudaCall(getBackend(_backendIdx).arrayCreate(ptr(), &desc));
}

inline Array::Array(unsigned width, unsigned height, CUarray_format format,
             unsigned numChannels) {
  CUDA_ARRAY_DESCRIPTOR desc{};
  desc.Width = width;
  desc.Height = height;
  desc.Format = format;
  desc.NumChannels = numChannels;
  checkCudaCall(getBackend(_backendIdx).arrayCreate(ptr(), &desc));
}

inline Array::Array(unsigned width, unsigned height, unsigned depth,
             CUarray_format format, unsigned numChannels) {
  CUDA_ARRAY3D_DESCRIPTOR desc{};
  desc.Width = width;
  desc.Height = height;
  desc.Depth = depth;
  desc.Format = format;
  desc.NumChannels = numChannels;
  checkCudaCall(getBackend(_backendIdx).array3DCreate(ptr(), &desc));
}

inline Array::Array(CUarray& array) : Wrapper<CUarray>(array) {}

// --- Module ---

inline Module::Module(const char* file_name) {
  checkCudaCall(getBackend(_backendIdx).moduleLoad(ptr(), file_name));
}

inline Module::Module(const void* data) {
  checkCudaCall(getBackend(_backendIdx).moduleLoadData(ptr(), data));
}

inline Module::Module(const void* image, Module::optionmap_t& options) {
  std::vector<CUjit_option> optionKeys;
  std::vector<void*> optionValues;
  for (auto& [key, value] : options) {
    optionKeys.push_back(key);
    optionValues.push_back(value);
  }
  checkCudaCall(getBackend(_backendIdx).moduleLoadDataEx(
      ptr(), image, optionKeys.size(),
      reinterpret_cast<int*>(optionKeys.data()), optionValues.data()));
}

inline Module::Module(CUmodule& module) : Wrapper<CUmodule>(module) {}

inline CUdeviceptr Module::getGlobal(const char* name) const {
  CUdeviceptr ptr{};
  size_t bytes{};
  checkCudaCall(getBackend(_backendIdx).moduleGetGlobal(&ptr, &bytes, _obj, name));
  return ptr;
}

// --- Function ---

inline Function::Function(const Module& module, const char* name) {
  checkCudaCall(
      getBackend(_backendIdx).moduleGetFunction(ptr(), module, name));
}

inline Function::Function(CUfunction& function) : Wrapper<CUfunction>(function) {}

inline int Function::getAttribute(CUfunction_attribute attribute) const {
  int value{};
  checkCudaCall(getBackend(_backendIdx).funcGetAttribute(&value, attribute, _obj));
  return value;
}

inline void Function::setAttribute(CUfunction_attribute attribute, int value) {
  checkCudaCall(getBackend(_backendIdx).funcSetAttribute(_obj, attribute, value));
}

inline int Function::occupancyMaxActiveBlocksPerMultiprocessor(int blockSize,
                                                        size_t dynamicSMemSize) {
  int numBlocks{};
  checkCudaCall(getBackend(_backendIdx).occupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocks, _obj, blockSize, dynamicSMemSize));
  return numBlocks;
}

inline void Function::setCacheConfig(CUfunc_cache config) {
  checkCudaCall(getBackend(_backendIdx).funcSetCacheConfig(_obj, config));
}

// --- Event ---

inline Event::Event(unsigned int flags) {
  checkCudaCall(getBackend(_backendIdx).eventCreate(ptr(), flags));
  int bIdx = _backendIdx;
  manager = std::shared_ptr<CUevent>(new CUevent(_obj), [bIdx](CUevent* ptr) {
    if (*ptr) getBackend(bIdx).eventDestroy(*ptr);
    delete ptr;
  });
}

inline Event::Event(CUevent& event) : Wrapper<CUevent>(event) {}

inline float Event::elapsedTime(const Event& start) const {
  float ms{};
  checkCudaCall(getBackend(_backendIdx).eventElapsedTime(&ms, start._obj, _obj));
  return ms;
}

inline void Event::query() const {
  checkCudaCall(getBackend(_backendIdx).eventQuery(_obj));
}

inline void Event::synchronize() {
  checkCudaCall(getBackend(_backendIdx).eventSynchronize(_obj));
}

inline void Event::record() {
  checkCudaCall(getBackend(_backendIdx).eventRecord(_obj, nullptr));
}

inline void Event::record(Stream& stream) {
  checkCudaCall(getBackend(_backendIdx).eventRecord(_obj, stream));
}

inline void Event::record(Stream& stream, unsigned int flags) {
  checkCudaCall(getBackend(_backendIdx).eventRecordWithFlags(_obj, stream, flags));
}

// --- DeviceMemory ---

inline DeviceMemory::DeviceMemory(size_t size, CUmemorytype type, unsigned int flags) {
  _size = size;
  if (type == CU_MEMORYTYPE_DEVICE) {
    if (flags != 0) {
      throw std::runtime_error(
          "DeviceMemory: flags are not supported with CU_MEMORYTYPE_DEVICE");
    }
    checkCudaCall(getBackend(_backendIdx).memAlloc(&_obj, size));
  } else if (type == CU_MEMORYTYPE_UNIFIED) {
    checkCudaCall(getBackend(_backendIdx).memAllocManaged(&_obj, size, flags));
  } else {
    throw std::runtime_error("Invalid memory type for DeviceMemory");
  }
}

inline DeviceMemory::DeviceMemory(CUdeviceptr ptr) { _obj = ptr; }

inline DeviceMemory::DeviceMemory(CUdeviceptr ptr, size_t size) {
  _obj = ptr;
  _size = size;
}

inline DeviceMemory::DeviceMemory(const HostMemory& hostMemory) {
  _size = hostMemory.size();
  checkCudaCall(getBackend(_backendIdx).memAlloc(&_obj, _size));
  checkCudaCall(getBackend(_backendIdx).memcpyHtoD(_obj, hostMemory, _size));
}

inline DeviceMemory::DeviceMemory(const DeviceMemory& other, size_t offset,
                           size_t size)
    : _size(size) {
  if (offset + size > other._size) {
    throw std::runtime_error("DeviceMemory: offset + size exceeds allocation size");
  }
  _obj = other._obj + offset;
}

inline void DeviceMemory::memset(unsigned char value, size_t size) {
  checkCudaCall(getBackend(_backendIdx).memsetD8(_obj, value, size));
}

inline void DeviceMemory::memset(unsigned short value, size_t size) {
  checkCudaCall(getBackend(_backendIdx).memsetD16(_obj, value, size));
}

inline void DeviceMemory::memset(unsigned int value, size_t size) {
  checkCudaCall(getBackend(_backendIdx).memsetD32(_obj, value, size));
}

inline void DeviceMemory::memset2D(unsigned char value, size_t pitch, size_t width,
                             size_t height) {
  checkCudaCall(getBackend(_backendIdx).memsetD2D8(_obj, pitch, value, width, height));
}

inline void DeviceMemory::memset2D(unsigned short value, size_t pitch, size_t width,
                             size_t height) {
  checkCudaCall(getBackend(_backendIdx).memsetD2D16(_obj, pitch, value, width, height));
}

inline void DeviceMemory::memset2D(unsigned int value, size_t pitch, size_t width,
                             size_t height) {
  checkCudaCall(getBackend(_backendIdx).memsetD2D32(_obj, pitch, value, width, height));
}

inline void DeviceMemory::zero(size_t size) {
  checkCudaCall(getBackend(_backendIdx).memsetD8(_obj, 0, size));
}

inline const void* DeviceMemory::parameter() const { return &_obj; }

inline size_t DeviceMemory::size() const { return _size; }

// --- Stream ---

inline Stream::Stream(unsigned int flags) {
  checkCudaCall(getBackend(_backendIdx).streamCreate(ptr(), flags));
  int bIdx = _backendIdx;
  manager = std::shared_ptr<CUstream>(new CUstream(_obj), [bIdx](CUstream* ptr) {
    if (*ptr) getBackend(bIdx).streamDestroy(*ptr);
    delete ptr;
  });
}

inline Stream::Stream(CUstream stream) { _obj = stream; }

inline Stream::Stream(CUstream stream, bool takeOwnership) {
  _obj = stream;
  if (takeOwnership) {
    int bIdx = _backendIdx;
    manager = std::shared_ptr<CUstream>(new CUstream(_obj),
                                         [bIdx](CUstream* ptr) {
                                           if (*ptr)
                                             getBackend(bIdx).streamDestroy(*ptr);
                                           delete ptr;
                                         });
  }
}

inline DeviceMemory Stream::memAllocAsync(size_t size) {
  CUdeviceptr ptr{};
  checkCudaCall(getBackend(_backendIdx).memAllocAsync(&ptr, size, _obj));
  return DeviceMemory(ptr, size);
}

inline void Stream::memFreeAsync(DeviceMemory& devMem) {
  checkCudaCall(getBackend(_backendIdx).memFreeAsync(devMem, _obj));
  devMem = DeviceMemory(CUdeviceptr{});
}

inline void Stream::memcpyHtoHAsync(void* dstPtr, const void* srcPtr, size_t size) {
  checkCudaCall(getBackend(_backendIdx).memcpyAsync(
      reinterpret_cast<CUdeviceptr>(dstPtr),
      reinterpret_cast<const void*>(srcPtr), size, _obj));
}

inline void Stream::memcpyHtoDAsync(DeviceMemory& dst, const void* src, size_t size) {
  checkCudaCall(getBackend(_backendIdx).memcpyHtoDAsync(dst, src, size, _obj));
}

inline void Stream::memcpyHtoDAsync(CUdeviceptr dst, const void* src, size_t size) {
  checkCudaCall(getBackend(_backendIdx).memcpyHtoDAsync(dst, src, size, _obj));
}

inline void Stream::memcpyDtoHAsync(void* dst, const DeviceMemory& src, size_t size) {
  checkCudaCall(getBackend(_backendIdx).memcpyDtoHAsync(dst, src, size, _obj));
}

inline void Stream::memcpyDtoHAsync(void* dst, CUdeviceptr src, size_t size) {
  checkCudaCall(getBackend(_backendIdx).memcpyDtoHAsync(dst, src, size, _obj));
}

inline void Stream::memcpyHtoD2DAsync(DeviceMemory& dst, size_t dpitch, const void* src,
                               size_t spitch, size_t width, size_t height) {
  checkCudaCall(getBackend(_backendIdx).memcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                           1, _obj));
}

inline void Stream::memcpyDtoH2DAsync(void* dst, size_t dpitch, const DeviceMemory& src,
                               size_t spitch, size_t width, size_t height) {
  checkCudaCall(getBackend(_backendIdx).memcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                           2, _obj));
}

inline void Stream::memcpyDtoDAsync(DeviceMemory& dst, const DeviceMemory& src,
                             size_t size) {
  checkCudaCall(getBackend(_backendIdx).memcpyAsync(dst, src, size, _obj));
}

inline void Stream::memPrefetchAsync(DeviceMemory& devMem, size_t size) {
  checkCudaCall(getBackend(_backendIdx).memPrefetchAsync(devMem, size, -1, _obj));
}

inline void Stream::memPrefetchAsync(DeviceMemory& devMem, size_t size,
                               Device& device) {
  checkCudaCall(getBackend(_backendIdx).memPrefetchAsync(devMem, size, device, _obj));
}

inline void Stream::memsetAsync(DeviceMemory& dst, unsigned char value, size_t count) {
  checkCudaCall(getBackend(_backendIdx).memsetD8Async(dst, value, count, _obj));
}

inline void Stream::memsetAsync(DeviceMemory& dst, unsigned short value,
                         size_t count) {
  checkCudaCall(getBackend(_backendIdx).memsetD16Async(dst, value, count, _obj));
}

inline void Stream::memsetAsync(DeviceMemory& dst, unsigned int value, size_t count) {
  checkCudaCall(getBackend(_backendIdx).memsetD32Async(dst, value, count, _obj));
}

inline void Stream::memset2DAsync(DeviceMemory& dst, unsigned char value,
                            size_t pitch, size_t width, size_t height) {
  checkCudaCall(
      getBackend(_backendIdx).memsetD2D8Async(dst, pitch, value, width, height, _obj));
}

inline void Stream::memset2DAsync(DeviceMemory& dst, unsigned short value,
                            size_t pitch, size_t width, size_t height) {
  checkCudaCall(
      getBackend(_backendIdx).memsetD2D16Async(dst, pitch, value, width, height, _obj));
}

inline void Stream::memset2DAsync(DeviceMemory& dst, unsigned int value,
                            size_t pitch, size_t width, size_t height) {
  checkCudaCall(
      getBackend(_backendIdx).memsetD2D32Async(dst, pitch, value, width, height, _obj));
}

inline void Stream::zero(DeviceMemory& dst, size_t size) {
  checkCudaCall(getBackend(_backendIdx).memsetD8Async(dst, 0, size, _obj));
}

inline void Stream::zero2D(DeviceMemory& dst, size_t pitch, size_t width,
                     size_t height) {
  checkCudaCall(
      getBackend(_backendIdx).memsetD2D8Async(dst, pitch, 0, width, height, _obj));
}

inline void Stream::launchKernel(Function& function, unsigned gridX, unsigned gridY,
                          unsigned gridZ, unsigned blockX, unsigned blockY,
                          unsigned blockZ, unsigned sharedMemBytes,
                          const std::vector<const void*>& parameters) {
  checkCudaCall(getBackend(_backendIdx).launchKernel(
      function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
      _obj, const_cast<void**>(parameters.data()), nullptr));
}

inline void Stream::graphLaunch(GraphExec& graphExec) {
  checkCudaCall(getBackend(_backendIdx).graphLaunch(graphExec, _obj));
}

inline void Stream::query() {
  checkCudaCall(getBackend(_backendIdx).streamQuery(_obj));
}

inline void Stream::synchronize() {
  checkCudaCall(getBackend(_backendIdx).streamSynchronize(_obj));
}

inline void Stream::wait(Event& event) {
  checkCudaCall(getBackend(_backendIdx).streamWaitEvent(_obj, event));
}

inline void Stream::record(Event& event) {
  checkCudaCall(getBackend(_backendIdx).streamRecordEvent(_obj, event));
}

inline void Stream::record(Event& event, unsigned int flags) {
  (void)flags;
  checkCudaCall(getBackend(_backendIdx).streamRecordEvent(_obj, event));
}

inline void Stream::launchHostFunc(CUhostFn fn, void* userData) {
  checkCudaCall(getBackend(_backendIdx).streamLaunchHostFunc(_obj, fn, userData));
}

// --- Graph ---

inline Graph::Graph(Context& context, CUgraph& graph) : Wrapper<CUgraph>(graph), _context(&context) {
  _backendIdx = context.getBackendIdx();
}

inline Graph::Graph(Context& context, unsigned int flags) : _context(&context) {
  _backendIdx = context.getBackendIdx();
  checkCudaCall(getBackend(_backendIdx).graphCreate(ptr(), flags));
  int bIdx = _backendIdx;
  manager = std::shared_ptr<CUgraph>(new CUgraph(_obj), [bIdx](CUgraph* ptr) {
    if (*ptr) getBackend(bIdx).graphDestroy(*ptr);
    delete ptr;
  });
}

inline void Graph::addKernelNode(GraphNode& node,
                           const std::vector<CUgraphNode>& dependencies,
                           GraphKernelNodeParams& params) {
  checkCudaCall(getBackend(_backendIdx).graphAddKernelNode(
      reinterpret_cast<void**>(node.getNode()), cu_backend_cast::toVoidP(_obj),
      reinterpret_cast<void* const*>(dependencies.data()), dependencies.size(),
      reinterpret_cast<const CUDA_KERNEL_NODE_PARAMS_b*>(params.parameter())));
}

inline void Graph::addHostNode(GraphNode& node,
                         const std::vector<CUgraphNode>& dependencies,
                         GraphHostNodeParams& params) {
  checkCudaCall(getBackend(_backendIdx).graphAddHostNode(
      reinterpret_cast<void**>(node.getNode()), cu_backend_cast::toVoidP(_obj),
      reinterpret_cast<void* const*>(dependencies.data()), dependencies.size(),
      reinterpret_cast<const CUDA_HOST_NODE_PARAMS_b*>(params.parameter())));
}

inline void Graph::addDevMemFreeNode(GraphNode& node,
                               const std::vector<CUgraphNode>& dependencies,
                               const CUdeviceptr& devPtr) {
  checkCudaCall(getBackend(_backendIdx).graphAddMemFreeNode(
      reinterpret_cast<void**>(node.getNode()), cu_backend_cast::toVoidP(_obj),
      reinterpret_cast<void* const*>(dependencies.data()), dependencies.size(), devPtr));
}

inline void Graph::addMemAllocNode(GraphNode& node,
                             const std::vector<CUgraphNode>& dependencies,
                             GraphDevMemAllocNodeParams& params) {
  checkCudaCall(getBackend(_backendIdx).graphAddMemAllocNode(
      reinterpret_cast<void**>(node.getNode()), cu_backend_cast::toVoidP(_obj),
      reinterpret_cast<void* const*>(dependencies.data()), dependencies.size(),
      reinterpret_cast<CUDA_MEM_ALLOC_NODE_PARAMS_b*>(const_cast<void*>(params.parameter()))));
}

inline void Graph::addMemCpyNode(GraphNode& node,
                           const std::vector<CUgraphNode>& dependencies,
                           GraphMemCopyToDeviceNodeParams& params) {
  checkCudaCall(getBackend(_backendIdx).graphAddMemcpyNode(
      reinterpret_cast<void**>(node.getNode()), cu_backend_cast::toVoidP(_obj),
      reinterpret_cast<void* const*>(dependencies.data()), dependencies.size(),
      reinterpret_cast<const CUDA_MEMCPY3D_b*>(params.parameter()),
      cu_backend_cast::toVoidP(static_cast<CUcontext>(*_context))));
}

inline void Graph::addMemCpyNode(GraphNode& node,
                           const std::vector<CUgraphNode>& dependencies,
                           GraphMemCopyToHostNodeParams& params) {
  checkCudaCall(getBackend(_backendIdx).graphAddMemcpyNode(
      reinterpret_cast<void**>(node.getNode()), cu_backend_cast::toVoidP(_obj),
      reinterpret_cast<void* const*>(dependencies.data()), dependencies.size(),
      reinterpret_cast<const CUDA_MEMCPY3D_b*>(params.parameter()),
      cu_backend_cast::toVoidP(static_cast<CUcontext>(*_context))));
}

inline void Graph::debugDotPrint(std::string path, CUgraphDebugDot_flags flags) {
  checkCudaCall(getBackend(_backendIdx).graphDebugDotPrint(_obj, path.c_str(), flags));
}

inline CUgraphExec Graph::instantiateWithFlags(unsigned int flags) {
  CUgraphExec exec{};
  checkCudaCall(
      getBackend(_backendIdx).graphInstantiate(reinterpret_cast<CUgraphExec_b*>(&exec), _obj, flags));
  return exec;
}

// --- GraphExec ---

inline GraphExec::GraphExec(CUgraphExec& graphExec) { _obj = graphExec; }
inline GraphExec::GraphExec(const Graph& graph, unsigned int flags) {
  CUgraphExec exec{};
  checkCudaCall(getBackend(_backendIdx).graphInstantiate(reinterpret_cast<CUgraphExec_b*>(&exec), graph, flags));
  _obj = exec;
}

// --- GraphNode ---

inline GraphNode::GraphNode(CUgraphNode& node) { _obj = node; }
inline CUgraphNode* GraphNode::getNode() { return &_obj; }

// --- GraphKernelNodeParams ---

inline GraphKernelNodeParams::GraphKernelNodeParams(
    const Function& function, unsigned gridDimX, unsigned gridDimY,
    unsigned gridDimZ, unsigned blockDimX, unsigned blockDimY,
    unsigned blockDimZ, unsigned sharedMemBytes,
    const std::vector<const void*>& kernelParams) {
  _obj.func = function;
  _obj.gridDimX = gridDimX;
  _obj.gridDimY = gridDimY;
  _obj.gridDimZ = gridDimZ;
  _obj.blockDimX = blockDimX;
  _obj.blockDimY = blockDimY;
  _obj.blockDimZ = blockDimZ;
  _obj.sharedMemBytes = sharedMemBytes;
  _obj.kernelParams = const_cast<void**>(kernelParams.data());
  _obj.extra = nullptr;
}

// --- GraphHostNodeParams ---

inline GraphHostNodeParams::GraphHostNodeParams(void (*fn)(void*), void* data) {
  _obj.fn = fn;
  _obj.userData = data;
}

// --- GraphDevMemAllocNodeParams ---

inline GraphDevMemAllocNodeParams::GraphDevMemAllocNodeParams(const Device& dev,
                                                        size_t size) {
  memset(&_obj, 0, sizeof(_obj));
  _obj.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  _obj.poolProps.location.id = static_cast<int>(dev);
  _obj.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
  _obj.poolProps.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
  _obj.bytesize = size;
  _obj.dptr = 0;
}

inline const CUdeviceptr& GraphDevMemAllocNodeParams::getDevPtr() const {
  return _obj.dptr;
}

inline const void* GraphDevMemAllocNodeParams::parameter() { return &_obj; }

inline const DeviceMemory GraphDevMemAllocNodeParams::getDeviceMemory() {
  return DeviceMemory(_obj.dptr, _obj.bytesize);
}

// --- GraphMemCopyToDeviceNodeParams ---

inline GraphMemCopyToDeviceNodeParams::GraphMemCopyToDeviceNodeParams(
    const CUdeviceptr& dst, const void* src, size_t size_x, size_t size_y,
    size_t size_z, size_t element_size, size_t pitch) {
  memset(&_obj, 0, sizeof(_obj));
  _obj.srcMemoryType = CU_MEMORYTYPE_HOST;
  _obj.srcHost = src;
  _obj.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  _obj.dstDevice = dst;
  _obj.WidthInBytes = size_x * element_size;
  _obj.Height = size_y;
  _obj.Depth = size_z;
  if (pitch > 0) {
    _obj.dstPitch = pitch;
    _obj.srcPitch = size_x * element_size;
  }
}

// --- GraphMemCopyToHostNodeParams ---

inline GraphMemCopyToHostNodeParams::GraphMemCopyToHostNodeParams(
    void* host, const CUdeviceptr& src, size_t size_x, size_t size_y,
    size_t size_z, size_t element_size, size_t pitch) {
  memset(&_obj, 0, sizeof(_obj));
  _obj.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  _obj.srcDevice = src;
  _obj.dstMemoryType = CU_MEMORYTYPE_HOST;
  _obj.dstHost = host;
  _obj.WidthInBytes = size_x * element_size;
  _obj.Height = size_y;
  _obj.Depth = size_z;
  if (pitch > 0) {
    _obj.srcPitch = pitch;
    _obj.dstPitch = size_x * element_size;
  }
}

// --- NVIDIA-only implementations ---

#if !defined(__HIP__)

inline size_t Device::getTexture1DLinearMaxWidth(CUarray_format format,
                                                   unsigned numChannels) const {
  size_t maxWidth{};
  Backend& b = getBackend(_backendIdx);
  checkCudaCall(b.deviceGetTexture1DLinearMaxWidth(&maxWidth,
                static_cast<int>(format), numChannels, static_cast<CUdevice_b>(_obj)));
  return maxWidth;
}

inline void Device::getExecAffinitySupport(int& pi,
                                             CUexecAffinityType type) const {
  Backend& b = getBackend(_backendIdx);
  checkCudaCall(b.deviceGetExecAffinitySupport(&pi, static_cast<int>(type),
                static_cast<CUdevice_b>(_obj)));
}

inline void Device::getProperties(CUdevprop& prop) const {
  prop.maxThreadsPerBlock =
      getAttribute<CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK>();
  prop.maxThreadsDim[0] = getAttribute<CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X>();
  prop.maxThreadsDim[1] = getAttribute<CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y>();
  prop.maxThreadsDim[2] = getAttribute<CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z>();
  prop.maxGridSize[0] = getAttribute<CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X>();
  prop.maxGridSize[1] = getAttribute<CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y>();
  prop.maxGridSize[2] = getAttribute<CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z>();
  prop.sharedMemPerBlock =
      getAttribute<CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK>();
  prop.totalConstantMemory =
      getAttribute<CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY>();
  prop.SIMDWidth = getAttribute<CU_DEVICE_ATTRIBUTE_WARP_SIZE>();
  prop.memPitch = getAttribute<CU_DEVICE_ATTRIBUTE_MAX_PITCH>();
  prop.regsPerBlock =
      getAttribute<CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK>();
  prop.clockRate = getAttribute<CU_DEVICE_ATTRIBUTE_CLOCK_RATE>();
  prop.textureAlign = getAttribute<CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT>();
}

inline void Device::getDevResource(CUdevResource& resource,
                                     CUdevResourceType type) const {
  Backend& b = getBackend(_backendIdx);
  checkCudaCall(b.deviceGetDevResource(static_cast<CUdevice_b>(_obj),
               &resource, static_cast<int>(type)));
}

inline void Context::getDevResource(CUdevResource& resource,
                                      CUdevResourceType type) const {
  Backend& b = getBackend(_backendIdx);
  checkCudaCall(b.ctxGetDevResource(_obj, &resource, static_cast<int>(type)));
}

inline Context Context::fromGreenCtx(GreenContext& greenContext) {
  CUcontext context{};
  Backend& b = getBackend(greenContext.getBackendIdx());
  checkCudaCall(b.ctxFromGreenCtx(cu_backend_cast::toVoidPP(context), greenContext));
  return Context(context, greenContext.getDevice());
}

// --- GreenContext ---

inline GreenContext::GreenContext(CUdevResourceDesc desc, Device& device,
                                   unsigned int flags)
    : _device(device) {
  _backendIdx = device.getBackendIdx();
  Backend& b = getBackend(_backendIdx);
  checkCudaCall(b.greenCtxCreate(ptr(),
               const_cast<void*>(static_cast<const void*>(desc)),
               static_cast<CUdevice_b>(device), flags));
  int bIdx = _backendIdx;
  manager = std::shared_ptr<CUgreenCtx>(new CUgreenCtx(_obj),
      [bIdx](CUgreenCtx* ptr) {
        if (*ptr) getBackend(bIdx).greenCtxDestroy(*ptr);
        delete ptr;
      });
}

inline void GreenContext::getDevResource(CUdevResource& resource,
                                           CUdevResourceType type) const {
  Backend& b = getBackend(_backendIdx);
  checkCudaCall(b.greenCtxGetDevResource(_obj, &resource,
               static_cast<int>(type)));
}

inline Stream GreenContext::createStream(unsigned int flags,
                                           int priority) const {
  CUstream stream;
  Backend& b = getBackend(_backendIdx);
  checkCudaCall(b.greenCtxStreamCreate(cu_backend_cast::toVoidPP(stream), _obj, flags, priority));
  return Stream(stream, true);
}

inline void GreenContext::recordEvent(Event& event) const {
  Backend& b = getBackend(_backendIdx);
  checkCudaCall(b.greenCtxRecordEvent(_obj, event));
}

inline void GreenContext::waitEvent(Event& event) const {
  Backend& b = getBackend(_backendIdx);
  checkCudaCall(b.greenCtxWaitEvent(_obj, event));
}

inline Device& GreenContext::getDevice() { return _device; }
inline const Device& GreenContext::getDevice() const { return _device; }

inline void Stream::getDevResource(CUdevResource& resource,
                                     CUdevResourceType type) const {
  Backend& b = getBackend(_backendIdx);
  checkCudaCall(b.streamGetDevResource(_obj, &resource,
               static_cast<int>(type)));
}

#endif  // !defined(__HIP__)

}  // namespace cu

#endif
