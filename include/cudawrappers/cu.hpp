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

// When building cudawrappers internally (CUDAWRAPPERS_INTERNAL) or for HIP,
// define manual types compatible with the backend abstraction.
// For consumers on CUDA, include the real CUDA headers.
#if !defined(__HIP__) && !defined(CUDAWRAPPERS_INTERNAL)
#include <cuda_runtime.h>
#include <cuda.h>
#else
// Unified CUDA types (compatible with HIP and backend abstraction)
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

// CUDA constants (same values on both CUDA and HIP)
constexpr CUresult CUDA_SUCCESS = 0;
constexpr CUresult CUDA_ERROR_NOT_FOUND = 500;

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

struct CUDA_MEMCPY3D {
  unsigned int srcXInBytes;
  unsigned int srcY;
  unsigned int srcZ;
  unsigned int srcLOD;
  CUmemorytype srcMemoryType;
  const void* srcHost;
  CUdeviceptr srcDevice;
  CUarray srcArray;
  unsigned int srcPitch;
  unsigned int srcHeight;
  unsigned int dstXInBytes;
  unsigned int dstY;
  unsigned int dstZ;
  unsigned int dstLOD;
  CUmemorytype dstMemoryType;
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
  CUfunction fn;
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
  int locationType;
  int deviceId;
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

#endif  // !defined(__HIP__) && !defined(CUDAWRAPPERS_INTERNAL)

constexpr unsigned int CU_GRAPH_DEFAULT = 0;

namespace cu {

class Error : public std::exception {
 public:
  explicit Error(CUresult result);
  const char* what() const noexcept;
  operator CUresult() const;

 private:
  CUresult _result;
};

void checkCudaCall(CUresult result);

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

 private:
  Context(CUcontext context, Device& device);
  Device* _device{nullptr};
};

class GraphExec;

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
  void debugDotPrint(std::string path, CUgraphDebugDot_flags flags);
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
};

}  // namespace cu

#endif
