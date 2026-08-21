#if !defined CU_BACKEND_H
#define CU_BACKEND_H

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <vector>

typedef int CUresult_b;
typedef int CUdevice_b;
typedef void* CUcontext_b;
typedef void* CUmodule_b;
typedef void* CUfunction_b;
typedef void* CUstream_b;
typedef void* CUevent_b;
typedef unsigned long long CUdeviceptr_b;
typedef void* CUmemoryPool_b;
typedef void* CUgraph_b;
typedef void* CUgraphExec_b;
typedef void* CUgraphNode_b;

struct CUuuid_b {
  char bytes[16];
};

struct CUdevprop_b {
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

struct CUmemPoolProps_b {
  int type;
  int location;
  void* handle;
  unsigned long long reserved;
};

struct CUDA_MEMCPY2D_b {
  unsigned int srcXInBytes;
  unsigned int srcY;
  unsigned int srcZ;
  unsigned int srcLOD;
  int srcMemoryType;
  const void* srcHost;
  CUdeviceptr_b srcDevice;
  void* srcArray;
  unsigned int srcPitch;
  unsigned int srcHeight;
  unsigned int dstXInBytes;
  unsigned int dstY;
  unsigned int dstZ;
  unsigned int dstLOD;
  int dstMemoryType;
  void* dstHost;
  CUdeviceptr_b dstDevice;
  void* dstArray;
  unsigned int dstPitch;
  unsigned int dstHeight;
  unsigned int WidthInBytes;
  unsigned int Height;
  unsigned int Depth;
};

struct CUDA_MEMCPY3D_b {
  unsigned int srcXInBytes;
  unsigned int srcY;
  unsigned int srcZ;
  unsigned int srcLOD;
  int srcMemoryType;
  const void* srcHost;
  CUdeviceptr_b srcDevice;
  void* srcArray;
  unsigned int srcPitch;
  unsigned int srcHeight;
  unsigned int dstXInBytes;
  unsigned int dstY;
  unsigned int dstZ;
  unsigned int dstLOD;
  int dstMemoryType;
  void* dstHost;
  CUdeviceptr_b dstDevice;
  void* dstArray;
  unsigned int dstPitch;
  unsigned int dstHeight;
  unsigned int WidthInBytes;
  unsigned int Height;
  unsigned int Depth;
};

struct CUDA_KERNEL_NODE_PARAMS_b {
  CUfunction_b fn;
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

struct CUDA_HOST_NODE_PARAMS_b {
  void (*fn)(void*);
  void* userData;
};

struct CUDA_MEM_ALLOC_NODE_PARAMS_b {
  char _opaque[48];
};

typedef void (*CUhostFn_b)(void*);
typedef void (*CUstreamCallback_b)(CUstream_b, CUresult_b, void*);

struct Backend {
  void* lib;

  CUresult_b (*init)(unsigned int);
  CUresult_b (*driverGetVersion)(int*);
  CUresult_b (*getErrorString)(CUresult_b, const char**);
  CUresult_b (*getErrorName)(CUresult_b, const char**);

  CUresult_b (*deviceGetCount)(int*);
  CUresult_b (*deviceGet)(CUdevice_b*, int);
  CUresult_b (*deviceGetAttribute)(int*, int, CUdevice_b);
  CUresult_b (*deviceGetName)(char*, int, CUdevice_b);
  CUresult_b (*deviceGetArchName)(char*, int, CUdevice_b);
  CUresult_b (*deviceGetUuid)(CUuuid_b*, CUdevice_b);
  CUresult_b (*deviceGetPCIBusId)(char*, int, CUdevice_b);
  CUresult_b (*deviceGetByPCIBusId)(CUdevice_b*, const char*);
  CUresult_b (*deviceTotalMem)(size_t*, CUdevice_b);
  CUresult_b (*deviceGetDefaultMemPool)(CUmemoryPool_b*, CUdevice_b);
  CUresult_b (*deviceGetMemPool)(CUmemoryPool_b*, CUdevice_b);
  CUresult_b (*deviceSetMemPool)(CUdevice_b, CUmemoryPool_b);
  CUresult_b (*deviceGetLuid)(char*, unsigned int*, CUdevice_b);
  CUresult_b (*deviceGetP2PAttribute)(int*, int, CUdevice_b, CUdevice_b);
  CUresult_b (*deviceCanAccessPeer)(int*, CUdevice_b, CUdevice_b);
  CUresult_b (*deviceGetTexture1DLinearMaxWidth)(size_t*, int, unsigned int,
                                                   CUdevice_b);
  CUresult_b (*deviceGraphMemTrim)(CUdevice_b);
  CUresult_b (*deviceGetGraphMemAttribute)(CUdevice_b, int, void*);
  CUresult_b (*deviceSetGraphMemAttribute)(CUdevice_b, int, void*);
  CUresult_b (*deviceGetExecAffinitySupport)(int*, int, CUdevice_b);
  CUresult_b (*deviceGetDevResource)(CUdevice_b, void*, int);

  CUresult_b (*ctxCreate)(CUcontext_b*, unsigned int, CUdevice_b);
  CUresult_b (*ctxDestroy)(CUcontext_b);
  CUresult_b (*ctxGetApiVersion)(CUcontext_b, unsigned int*);
  CUresult_b (*ctxGetCacheConfig)(int*);
  CUresult_b (*ctxSetCacheConfig)(int);
  CUresult_b (*ctxGetCurrent)(CUcontext_b*);
  CUresult_b (*ctxSetCurrent)(CUcontext_b);
  CUresult_b (*ctxPopCurrent)(CUcontext_b*);
  CUresult_b (*ctxPushCurrent)(CUcontext_b);
  CUresult_b (*ctxEnablePeerAccess)(CUcontext_b, unsigned int);
  CUresult_b (*ctxDisablePeerAccess)(CUcontext_b);
  CUresult_b (*ctxGetDevice)(CUdevice_b*);
  CUresult_b (*ctxGetLimit)(size_t*, int);
  CUresult_b (*ctxSetLimit)(int, size_t);
  CUresult_b (*ctxSynchronize)();
  CUresult_b (*ctxGetDevResource)(CUcontext_b, void*, int);
  CUresult_b (*ctxFromGreenCtx)(CUcontext_b*, void*);

  CUresult_b (*memAlloc)(CUdeviceptr_b*, size_t);
  CUresult_b (*memAllocManaged)(CUdeviceptr_b*, size_t, unsigned int);
  CUresult_b (*memAllocAsync)(CUdeviceptr_b*, size_t, CUstream_b);
  CUresult_b (*memFree)(CUdeviceptr_b);
  CUresult_b (*memFreeAsync)(CUdeviceptr_b, CUstream_b);
  CUresult_b (*memHostAlloc)(void**, size_t, unsigned int);
  CUresult_b (*memHostFree)(void*);
  CUresult_b (*memHostRegister)(void*, size_t, unsigned long long);
  CUresult_b (*memHostUnregister)(void*);
  CUresult_b (*memHostGetDevicePointer)(void**, void*, unsigned int);
  CUresult_b (*memGetInfo)(size_t*, size_t*);
  CUresult_b (*memPrefetchAsync)(const void*, size_t, int, CUstream_b);

  CUresult_b (*memcpyHtoD)(CUdeviceptr_b, const void*, size_t);
  CUresult_b (*memcpyDtoH)(void*, CUdeviceptr_b, size_t);
  CUresult_b (*memcpyAsync)(CUdeviceptr_b, const void*, size_t, CUstream_b);
  CUresult_b (*memcpyHtoDAsync)(CUdeviceptr_b, const void*, size_t,
                                 CUstream_b);
  CUresult_b (*memcpyDtoHAsync)(void*, CUdeviceptr_b, size_t, CUstream_b);
  CUresult_b (*memcpy2DAsync)(void*, size_t, const void*, size_t, size_t,
                               size_t, int, CUstream_b);

  CUresult_b (*memsetD8)(CUdeviceptr_b, unsigned char, size_t);
  CUresult_b (*memsetD16)(CUdeviceptr_b, unsigned short, size_t);
  CUresult_b (*memsetD32)(CUdeviceptr_b, unsigned int, size_t);
  CUresult_b (*memsetD2D8)(CUdeviceptr_b, size_t, unsigned char, size_t,
                            size_t);
  CUresult_b (*memsetD2D16)(CUdeviceptr_b, size_t, unsigned short, size_t,
                             size_t);
  CUresult_b (*memsetD2D32)(CUdeviceptr_b, size_t, unsigned int, size_t,
                             size_t);
  CUresult_b (*memsetD8Async)(CUdeviceptr_b, unsigned char, size_t,
                               CUstream_b);
  CUresult_b (*memsetD16Async)(CUdeviceptr_b, unsigned short, size_t,
                                CUstream_b);
  CUresult_b (*memsetD32Async)(CUdeviceptr_b, unsigned int, size_t,
                                CUstream_b);
  CUresult_b (*memsetD2D8Async)(CUdeviceptr_b, size_t, unsigned char, size_t,
                                 size_t, CUstream_b);
  CUresult_b (*memsetD2D16Async)(CUdeviceptr_b, size_t, unsigned short, size_t,
                                  size_t, CUstream_b);
  CUresult_b (*memsetD2D32Async)(CUdeviceptr_b, size_t, unsigned int, size_t,
                                  size_t, CUstream_b);

  CUresult_b (*pointerSetAttribute)(const void*, int, CUdeviceptr_b);
  CUresult_b (*pointerGetAttribute)(void*, int, CUdeviceptr_b);
  CUresult_b (*pointerGetAttributes)(void*, int, CUdeviceptr_b);

  CUresult_b (*arrayCreate)(void**, void*);
  CUresult_b (*array3DCreate)(void**, void*);
  CUresult_b (*arrayDestroy)(void*);

  CUresult_b (*streamCreate)(CUstream_b*, unsigned int);
  CUresult_b (*streamCreateWithPriority)(CUstream_b*, unsigned int, int);
  CUresult_b (*streamDestroy)(CUstream_b);
  CUresult_b (*streamSynchronize)(CUstream_b);
  CUresult_b (*streamQuery)(CUstream_b);
  CUresult_b (*streamWaitEvent)(CUstream_b, CUevent_b);
  CUresult_b (*streamGetFlags)(CUstream_b, unsigned int*);
  CUresult_b (*streamGetPriority)(CUstream_b, int*);
  CUresult_b (*streamAddCallback)(CUstream_b, CUstreamCallback_b, void*,
                                   unsigned int);
  CUresult_b (*streamLaunchHostFunc)(CUstream_b, CUhostFn_b, void*);
  CUresult_b (*streamRecordEvent)(CUstream_b, CUevent_b);
  CUresult_b (*streamWaitValue32)(CUstream_b, CUdeviceptr_b, unsigned int,
                                   unsigned int);
  CUresult_b (*streamWriteValue32)(CUstream_b, CUdeviceptr_b, unsigned int,
                                    unsigned int);
  CUresult_b (*streamBatchMemOp)(CUstream_b, unsigned int, void*, unsigned int);
  CUresult_b (*streamGetDevResource)(CUstream_b, void*, int);
  CUresult_b (*streamGetGreenCtx)(CUstream_b, void**);

  CUresult_b (*eventCreate)(CUevent_b*, unsigned int);
  CUresult_b (*eventDestroy)(CUevent_b);
  CUresult_b (*eventRecord)(CUevent_b, CUevent_b);
  CUresult_b (*eventRecordWithFlags)(CUevent_b, CUevent_b, unsigned int);
  CUresult_b (*eventSynchronize)(CUevent_b);
  CUresult_b (*eventElapsedTime)(float*, CUevent_b, CUevent_b);
  CUresult_b (*eventQuery)(CUevent_b);

  CUresult_b (*moduleLoad)(CUmodule_b*, const char*);
  CUresult_b (*moduleLoadData)(CUmodule_b*, const void*);
  CUresult_b (*moduleLoadDataEx)(CUmodule_b*, const void*, unsigned int,
                                  int*, void**);
  CUresult_b (*moduleUnload)(CUmodule_b);
  CUresult_b (*moduleGetFunction)(CUfunction_b*, CUmodule_b, const char*);
  CUresult_b (*moduleGetGlobal)(CUdeviceptr_b*, size_t*, CUmodule_b,
                                 const char*);

  CUresult_b (*funcGetAttribute)(int*, int, CUfunction_b);
  CUresult_b (*funcSetAttribute)(const void*, int, int);
  CUresult_b (*funcSetCacheConfig)(const void*, int);
  CUresult_b (*occupancyMaxActiveBlocksPerMultiprocessor)(int*, CUfunction_b,
                                                           int, size_t);

  CUresult_b (*graphCreate)(CUgraph_b*, unsigned int);
  CUresult_b (*graphDestroy)(CUgraph_b);
  CUresult_b (*graphInstantiate)(CUgraphExec_b*, CUgraph_b, unsigned int);
  CUresult_b (*graphLaunch)(CUgraphExec_b, CUstream_b);
  CUresult_b (*graphDestroyExec)(CUgraphExec_b);
  CUresult_b (*graphDebugDotPrint)(CUgraph_b, const char*, unsigned int);

  CUresult_b (*graphAddKernelNode)(CUgraphNode_b*, CUgraph_b,
                                    const CUgraphNode_b*, size_t,
                                    const CUDA_KERNEL_NODE_PARAMS_b*);
  CUresult_b (*graphAddHostNode)(CUgraphNode_b*, CUgraph_b,
                                  const CUgraphNode_b*, size_t,
                                  const CUDA_HOST_NODE_PARAMS_b*);
  CUresult_b (*graphAddMemFreeNode)(CUgraphNode_b*, CUgraph_b,
                                     const CUgraphNode_b*, size_t,
                                     CUdeviceptr_b);
  CUresult_b (*graphAddMemAllocNode)(CUgraphNode_b*, CUgraph_b,
                                      const CUgraphNode_b*, size_t,
                                      CUDA_MEM_ALLOC_NODE_PARAMS_b*);
  CUresult_b (*graphAddMemcpyNode)(CUgraphNode_b*, CUgraph_b,
                                    const CUgraphNode_b*, size_t,
                                    const CUDA_MEMCPY3D_b*, CUcontext_b);

  CUresult_b (*launchKernel)(void*, unsigned int, unsigned int, unsigned int,
                              unsigned int, unsigned int, unsigned int,
                              unsigned int, CUstream_b, void**, void**);
  CUresult_b (*launchCooperativeKernel)(void*, unsigned int, unsigned int,
                                         unsigned int, unsigned int,
                                         unsigned int, unsigned int,
                                         unsigned int, unsigned int,
                                         CUstream_b, void**);

  CUresult_b (*getExportTable)(void*, const CUuuid_b*);

  CUresult_b (*greenCtxCreate)(void**, void*, CUdevice_b, unsigned int);
  CUresult_b (*greenCtxDestroy)(void*);
  CUresult_b (*greenCtxGetDevResource)(void*, void*, int);
  CUresult_b (*greenCtxGetId)(void*, unsigned long long*);
  CUresult_b (*greenCtxStreamCreate)(CUstream_b*, void*, unsigned int, int);
  CUresult_b (*greenCtxRecordEvent)(void*, CUevent_b);
  CUresult_b (*greenCtxWaitEvent)(void*, CUevent_b);

  CUresult_b (*devResourceGenerateDesc)(void**, void*, unsigned int);
  CUresult_b (*devSmResourceSplitByCount)(void*, unsigned int*, const void*,
                                           void*, unsigned int, unsigned int);

  int is_cuda;
};

// --- Backend loader declarations ---
inline Backend loadCudaBackend();
inline Backend loadHipBackend();

// --- Backend management (inline, header-only) ---

inline std::vector<Backend>& getBackends() {
  static std::vector<Backend> backends;
  static bool loaded = false;
  if (!loaded) {
    loaded = true;

    Backend cuda = loadCudaBackend();
    if (cuda.lib && cuda.init) {
      backends.push_back(cuda);
    } else if (cuda.lib) {
      dlclose(cuda.lib);
    }

    Backend hip = loadHipBackend();
    if (hip.lib && hip.init) {
      backends.push_back(hip);
    } else if (hip.lib) {
      dlclose(hip.lib);
    }
  }
  return backends;
}

inline size_t getBackendCount() { return getBackends().size(); }

inline Backend& getBackend(int idx) { return getBackends().at(idx); }

inline Backend& getBackend() {
  static Backend empty{};
  auto& backends = getBackends();
  if (!backends.empty()) {
    return backends.front();
  }
  return empty;
}

// --- CUDA backend (loaded entirely via dlsym, no CUDA headers needed) ---

#if __has_include(<cuda.h>) && !defined(__HIP__)

namespace {

// Layout-compatible with CUDA_MEMCPY2D from cuda.h (must match exactly)
struct CUDA_MEMCPY2D_compat {
  size_t srcXInBytes;
  size_t srcY;
  int srcMemoryType;
  const void* srcHost;
  unsigned long long srcDevice;
  void* srcArray;
  size_t srcPitch;
  size_t dstXInBytes;
  size_t dstY;
  int dstMemoryType;
  void* dstHost;
  unsigned long long dstDevice;
  void* dstArray;
  size_t dstPitch;
  size_t WidthInBytes;
  size_t Height;
};

inline int cudaMemcpy2DAsyncWrapper(void* dst, size_t dpitch, const void* src,
                                     size_t spitch, size_t width, size_t height,
                                     int kind, void* stream) {
  Backend& b = getBackend();
  if (!b.lib) return 1;
  CUDA_MEMCPY2D_compat copyParams = {};
  copyParams.WidthInBytes = width;
  copyParams.Height = height;
  copyParams.srcPitch = spitch;
  copyParams.dstPitch = dpitch;
  if (kind == 1) {
    copyParams.srcMemoryType = 1;
    copyParams.srcHost = src;
    copyParams.dstMemoryType = 2;
    copyParams.dstDevice = reinterpret_cast<unsigned long long>(dst);
  } else {
    copyParams.srcMemoryType = 2;
    copyParams.srcDevice = reinterpret_cast<unsigned long long>(src);
    copyParams.dstMemoryType = 1;
    copyParams.dstHost = dst;
  }
  using Fn = CUresult_b (*)(const void*, void*);
  static Fn fn = reinterpret_cast<Fn>(dlsym(b.lib, "cuMemcpy2DAsync_v2"));
  if (!fn) return 1;
  return fn(&copyParams, stream);
}

inline int cuEventRecordWrapper(void* stream, void* event) {
  Backend& b = getBackend();
  if (!b.lib) return 1;
  using Fn = CUresult_b (*)(void*, void*);
  static Fn fn = reinterpret_cast<Fn>(dlsym(b.lib, "cuEventRecord"));
  if (!fn) return 1;
  return fn(event, stream);
}

inline int cuStreamWaitEventWrapper(void* stream, void* event) {
  Backend& b = getBackend();
  if (!b.lib) return 1;
  using Fn = CUresult_b (*)(void*, void*, unsigned int);
  static Fn fn = reinterpret_cast<Fn>(dlsym(b.lib, "cuStreamWaitEvent"));
  if (!fn) return 1;
  return fn(stream, event, 0);
}

}  // anonymous namespace

inline Backend loadCudaBackend() {
  Backend b{};
  memset(&b, 0, sizeof(b));

  b.lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
  if (!b.lib) {
    b.lib = dlopen("libcuda.so", RTLD_LAZY | RTLD_GLOBAL);
  }
  if (!b.lib) {
    return b;
  }

#define LOAD(name, cuda_name) \
  b.name = reinterpret_cast<decltype(Backend::name)>(dlsym(b.lib, cuda_name))

  LOAD(init, "cuInit");
  LOAD(driverGetVersion, "cuDriverGetVersion");
  LOAD(getErrorString, "cuGetErrorString");
  LOAD(getErrorName, "cuGetErrorName");
  LOAD(deviceGetCount, "cuDeviceGetCount");
  LOAD(deviceGet, "cuDeviceGet");
  LOAD(deviceGetAttribute, "cuDeviceGetAttribute");
  LOAD(deviceGetName, "cuDeviceGetName");
  LOAD(deviceGetUuid, "cuDeviceGetUuid");
  LOAD(deviceGetPCIBusId, "cuDeviceGetPCIBusId");
  LOAD(deviceGetByPCIBusId, "cuDeviceGetByPCIBusId");
  LOAD(deviceTotalMem, "cuDeviceTotalMem");
  LOAD(deviceGetDefaultMemPool, "cuDeviceGetDefaultMemPool");
  LOAD(deviceGetMemPool, "cuDeviceGetMemPool");
  LOAD(deviceSetMemPool, "cuDeviceSetMemPool");
  LOAD(deviceGetLuid, "cuDeviceGetLuid");
  LOAD(deviceGetP2PAttribute, "cuDeviceGetP2PAttribute");
  LOAD(deviceCanAccessPeer, "cuDeviceCanAccessPeer");
  LOAD(deviceGetTexture1DLinearMaxWidth, "cuDeviceGetTexture1DLinearMaxWidth");
  LOAD(deviceGraphMemTrim, "cuDeviceGraphMemTrim");
  LOAD(deviceGetGraphMemAttribute, "cuDeviceGetGraphMemAttribute");
  LOAD(deviceSetGraphMemAttribute, "cuDeviceSetGraphMemAttribute");
  LOAD(deviceGetExecAffinitySupport, "cuDeviceGetExecAffinitySupport");
  LOAD(deviceGetDevResource, "cuDeviceGetDevResource");

  b.ctxCreate = reinterpret_cast<decltype(Backend::ctxCreate)>(
      dlsym(b.lib, "cuCtxCreate_v2"));
  if (!b.ctxCreate) {
    b.ctxCreate = reinterpret_cast<decltype(Backend::ctxCreate)>(
        dlsym(b.lib, "cuCtxCreate"));
  }
  LOAD(ctxDestroy, "cuCtxDestroy");
  LOAD(ctxGetApiVersion, "cuCtxGetApiVersion");
  LOAD(ctxGetCacheConfig, "cuCtxGetCacheConfig");
  LOAD(ctxSetCacheConfig, "cuCtxSetCacheConfig");
  LOAD(ctxGetCurrent, "cuCtxGetCurrent");
  LOAD(ctxSetCurrent, "cuCtxSetCurrent");
  LOAD(ctxPopCurrent, "cuCtxPopCurrent");
  LOAD(ctxPushCurrent, "cuCtxPushCurrent");
  LOAD(ctxEnablePeerAccess, "cuCtxEnablePeerAccess");
  LOAD(ctxDisablePeerAccess, "cuCtxDisablePeerAccess");
  LOAD(ctxGetDevice, "cuCtxGetDevice");
  LOAD(ctxGetLimit, "cuCtxGetLimit");
  LOAD(ctxSetLimit, "cuCtxSetLimit");
  LOAD(ctxSynchronize, "cuCtxSynchronize");
  LOAD(ctxGetDevResource, "cuCtxGetDevResource");
  LOAD(ctxFromGreenCtx, "cuCtxFromGreenCtx");

  LOAD(memAlloc, "cuMemAlloc_v2");
  LOAD(memAllocManaged, "cuMemAllocManaged");
  LOAD(memAllocAsync, "cuMemAllocAsync");
  LOAD(memFree, "cuMemFree_v2");
  LOAD(memFreeAsync, "cuMemFreeAsync");
  LOAD(memHostAlloc, "cuMemHostAlloc");
  LOAD(memHostFree, "cuMemFreeHost");
  LOAD(memHostRegister, "cuMemHostRegister_v2");
  LOAD(memHostUnregister, "cuMemHostUnregister");
  LOAD(memHostGetDevicePointer, "cuMemHostGetDevicePointer_v2");
  LOAD(memGetInfo, "cuMemGetInfo_v2");
  LOAD(memPrefetchAsync, "cuMemPrefetchAsync");

  LOAD(memcpyHtoD, "cuMemcpyHtoD_v2");
  LOAD(memcpyDtoH, "cuMemcpyDtoH_v2");
  LOAD(memcpyAsync, "cuMemcpyAsync");
  LOAD(memcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
  LOAD(memcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
  b.memcpy2DAsync =
      reinterpret_cast<decltype(Backend::memcpy2DAsync)>(cudaMemcpy2DAsyncWrapper);

  LOAD(memsetD8, "cuMemsetD8_v2");
  LOAD(memsetD16, "cuMemsetD16_v2");
  LOAD(memsetD32, "cuMemsetD32_v2");
  LOAD(memsetD2D8, "cuMemsetD2D8_v2");
  LOAD(memsetD2D16, "cuMemsetD2D16_v2");
  LOAD(memsetD2D32, "cuMemsetD2D32_v2");
  LOAD(memsetD8Async, "cuMemsetD8Async");
  LOAD(memsetD16Async, "cuMemsetD16Async");
  LOAD(memsetD32Async, "cuMemsetD32Async");
  LOAD(memsetD2D8Async, "cuMemsetD2D8Async");
  LOAD(memsetD2D16Async, "cuMemsetD2D16Async");
  LOAD(memsetD2D32Async, "cuMemsetD2D32Async");

  LOAD(pointerSetAttribute, "cuPointerSetAttribute");
  LOAD(pointerGetAttribute, "cuPointerGetAttribute");
  LOAD(pointerGetAttributes, "cuPointerGetAttributes");

  LOAD(arrayCreate, "cuArrayCreate");
  LOAD(array3DCreate, "cuArray3DCreate_v2");
  LOAD(arrayDestroy, "cuArrayDestroy");

  LOAD(streamCreate, "cuStreamCreate");
  LOAD(streamCreateWithPriority, "cuStreamCreateWithPriority");
  LOAD(streamDestroy, "cuStreamDestroy");
  LOAD(streamSynchronize, "cuStreamSynchronize");
  LOAD(streamQuery, "cuStreamQuery");
  LOAD(streamGetFlags, "cuStreamGetFlags");
  LOAD(streamGetPriority, "cuStreamGetPriority");
  LOAD(streamGetDevResource, "cuStreamGetDevResource");
  LOAD(streamGetGreenCtx, "cuStreamGetGreenCtx");
  b.streamWaitEvent = cuStreamWaitEventWrapper;

  LOAD(eventCreate, "cuEventCreate");
  LOAD(eventDestroy, "cuEventDestroy");
  LOAD(eventRecord, "cuEventRecord_v2");
  LOAD(eventRecordWithFlags, "cuEventRecordWithFlags");
  LOAD(eventSynchronize, "cuEventSynchronize");
  LOAD(eventElapsedTime, "cuEventElapsedTime");
  LOAD(eventQuery, "cuEventQuery");
  b.streamRecordEvent = cuEventRecordWrapper;
  LOAD(streamLaunchHostFunc, "cuLaunchHostFunc");

  LOAD(moduleLoad, "cuModuleLoad");
  LOAD(moduleLoadData, "cuModuleLoadData");
  LOAD(moduleLoadDataEx, "cuModuleLoadDataEx");
  LOAD(moduleUnload, "cuModuleUnload");
  LOAD(moduleGetFunction, "cuModuleGetFunction");
  LOAD(moduleGetGlobal, "cuModuleGetGlobal_v2");

  LOAD(funcGetAttribute, "cuFuncGetAttribute");
  LOAD(funcSetAttribute, "cuFuncSetAttribute");
  LOAD(funcSetCacheConfig, "cuFuncSetCacheConfig");
  LOAD(occupancyMaxActiveBlocksPerMultiprocessor,
       "cuOccupancyMaxActiveBlocksPerMultiprocessor");

  LOAD(graphCreate, "cuGraphCreate");
  LOAD(graphDestroy, "cuGraphDestroy");
  LOAD(graphInstantiate, "cuGraphInstantiateWithFlags");
  LOAD(graphLaunch, "cuGraphLaunch");
  LOAD(graphDestroyExec, "cuGraphExecDestroy");
  LOAD(graphDebugDotPrint, "cuGraphDebugDotPrint");
  LOAD(graphAddKernelNode, "cuGraphAddKernelNode");
  LOAD(graphAddHostNode, "cuGraphAddHostNode");
  LOAD(graphAddMemFreeNode, "cuGraphAddMemFreeNode");
  LOAD(graphAddMemAllocNode, "cuGraphAddMemAllocNode");
  LOAD(graphAddMemcpyNode, "cuGraphAddMemcpyNode");

  LOAD(launchKernel, "cuLaunchKernel");
  LOAD(launchCooperativeKernel, "cuLaunchCooperativeKernel");

  LOAD(getExportTable, "cuGetExportTable");

  LOAD(greenCtxCreate, "cuGreenCtxCreate");
  LOAD(greenCtxDestroy, "cuGreenCtxDestroy");
  LOAD(greenCtxGetDevResource, "cuGreenCtxGetDevResource");
  LOAD(greenCtxGetId, "cuGreenCtxGetId");
  LOAD(greenCtxStreamCreate, "cuGreenCtxStreamCreate");
  LOAD(greenCtxRecordEvent, "cuGreenCtxRecordEvent");
  LOAD(greenCtxWaitEvent, "cuGreenCtxWaitEvent");

  LOAD(devResourceGenerateDesc, "cuDevResourceGenerateDesc");
  LOAD(devSmResourceSplitByCount, "cuDevSmResourceSplitByCount");

#undef LOAD

  {
    struct ArchNameHelper {
      static CUresult_b getArchName(char* name, int maxLen,
                                     CUdevice_b device) {
        Backend& bb = getBackend();
        auto getAttr = reinterpret_cast<CUresult_b (*)(int*, int, CUdevice_b)>(
            dlsym(bb.lib, "cuDeviceGetAttribute"));
        if (!getAttr) return 500;
        int major = 0, minor = 0;
        CUresult_b r1 = getAttr(&major, 75, device);
        CUresult_b r2 = getAttr(&minor, 76, device);
        if (r1 != 0 || r2 != 0) return r1 ? r1 : r2;
        int written = snprintf(name, maxLen, "sm_%d", 10 * major + minor);
        if (written < 0 || written >= maxLen) return 500;
        return 0;
      }
    };
    b.deviceGetArchName = ArchNameHelper::getArchName;
  }

  b.is_cuda = 1;
  return b;
}

#else  // CUDA not available or HIP

inline Backend loadCudaBackend() {
  Backend b{};
  memset(&b, 0, sizeof(b));
  return b;
}

#endif  // __has_include(<cuda.h>) && !defined(__HIP__)

// --- CUDA-to-HIP attribute mapping (no HIP headers needed) ---

namespace {

inline int cudaToHipDeviceAttribute(int cudaAttr) {
  switch (cudaAttr) {
    case 1: return 55;   // MAX_THREADS_PER_BLOCK
    case 2: return 25;   // MAX_BLOCK_DIM_X
    case 3: return 26;   // MAX_BLOCK_DIM_Y
    case 4: return 27;   // MAX_BLOCK_DIM_Z
    case 5: return 28;   // MAX_GRID_DIM_X
    case 6: return 29;   // MAX_GRID_DIM_Y
    case 7: return 30;   // MAX_GRID_DIM_Z
    case 8: return 73;   // MAX_SHARED_MEMORY_PER_BLOCK
    case 9: return 5;    // COMPUTE_MODE
    case 10: return 86;  // WARP_SIZE
    case 11: return 57;  // MAX_PITCH
    case 12: return 70;  // MAX_REGISTERS_PER_BLOCK
    case 13: return 4;   // CLOCK_RATE
    case 14: return 80;  // TEXTURE_ALIGNMENT
    case 15: return 17;  // KERNEL_EXEC_TIMEOUT
    case 16: return 62;  // MULTIPROCESSOR_COUNT
    case 18: return 15;  // INTEGRATED
    case 31: return 255; // ECC_ENABLED
    case 32: return 1;   // ASYNC_ENGINE_COUNT
    case 33: return 66;  // PCI_BUS_ID
    case 34: return 67;  // PCI_DEVICE_ID
    case 35: return 68;  // PCI_DOMAIN_ID
    case 36: return 59;  // MEMORY_CLOCK_RATE
    case 37: return 58;  // GLOBAL_MEMORY_BUS_WIDTH
    case 38: return 18;  // L2_CACHE_SIZE
    case 39: return 23;  // MANAGED_MEMORY
    case 40: return 8;   // CONCURRENT_MANAGED_ACCESS
    case 42: return 14;  // HOST_NATIVE_ATOMIC_SUPPORTED
    case 43: return 13;  // GLOBAL_L1_CACHE_SUPPORTED
    case 44: return 19;  // LOCAL_L1_CACHE_SUPPORTED
    case 45: return 71;  // MAX_REGISTERS_PER_MULTIPROCESSOR
    case 46: return 92;  // MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
    case 47: return 56;  // MAX_THREADS_PER_MULTIPROCESSOR
    case 53: return 3;   // CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
    case 67: return 87;  // MEMORY_POOLS_SUPPORTED
    case 68: return 6;   // COMPUTE_PREEMPTION_SUPPORTED
    case 75: return 23;  // COMPUTE_CAPABILITY_MAJOR
    case 76: return 61;  // COMPUTE_CAPABILITY_MINOR
    default: return cudaAttr;  // pass through as-is
  }
}

}  // anonymous namespace

// --- HIP wrapper functions (only available when compiling with HIP) ---

#if defined(__HIP__)

#include <hip/hip_runtime.h>

namespace {

inline int hipModuleLoadDataEx_wrap(CUmodule_b* module, const void* image,
                                     unsigned int numOptions, int* options,
                                     void** optionValues) {
  hipJitOption* jitOpts = nullptr;
  if (numOptions > 0 && options) {
    jitOpts = reinterpret_cast<hipJitOption*>(options);
  }
  return hipModuleLoadDataEx(reinterpret_cast<hipModule_t*>(module), image,
                              numOptions, jitOpts, optionValues);
}

inline int hipGraphAddMemcpyNode_wrap(CUgraphNode_b* node, CUgraph_b graph,
                                       const CUgraphNode_b* deps, size_t numDeps,
                                       const CUDA_MEMCPY3D_b* p,
                                       CUcontext_b ctx) {
  hipMemcpy3DParms par{};
  memset(&par, 0, sizeof(par));

  if (p->srcMemoryType == 1 && p->dstMemoryType == 2) {
    par.srcPtr = make_hipPitchedPtr(const_cast<void*>(p->srcHost),
                                     p->srcPitch, p->WidthInBytes, p->Height);
    par.dstPtr = make_hipPitchedPtr(reinterpret_cast<void*>(p->dstDevice),
                                     p->dstPitch, p->WidthInBytes, p->Height);
    par.extent = make_hipExtent(p->WidthInBytes, p->Height, p->Depth);
    par.kind = hipMemcpyHostToDevice;
  } else if (p->srcMemoryType == 2 && p->dstMemoryType == 1) {
    par.srcPtr = make_hipPitchedPtr(reinterpret_cast<void*>(p->srcDevice),
                                     p->srcPitch, p->WidthInBytes, p->Height);
    par.dstPtr = make_hipPitchedPtr(p->dstHost, p->dstPitch, p->WidthInBytes,
                                     p->Height);
    par.extent = make_hipExtent(p->WidthInBytes, p->Height, p->Depth);
    par.kind = hipMemcpyDeviceToHost;
  } else {
    return 1;
  }

  (void)ctx;
  return hipGraphAddMemcpyNode(
      reinterpret_cast<hipGraphNode_t*>(node),
      reinterpret_cast<hipGraph_t>(graph),
      reinterpret_cast<const hipGraphNode_t*>(deps), numDeps, &par);
}

inline int hipDeviceGetAttribute_wrap(int* pi, int attr, int device) {
  return hipDeviceGetAttribute(
      pi, static_cast<hipDeviceAttribute_t>(cudaToHipDeviceAttribute(attr)),
      device);
}

inline int hipMemsetD2D8_wrap(CUdeviceptr_b dst, size_t pitch,
                               unsigned char value, size_t width,
                               size_t height) {
  return hipMemset2D(reinterpret_cast<void*>(dst), pitch, value, width, height);
}

inline int hipMemsetD2D16_wrap(CUdeviceptr_b dst, size_t pitch,
                                unsigned short value, size_t width,
                                size_t height) {
  for (size_t row = 0; row < height; ++row) {
    hipError_t err = hipMemsetD16(
        reinterpret_cast<void*>(dst + row * pitch), value, width);
    if (err != hipSuccess) return static_cast<int>(err);
  }
  return 0;
}

inline int hipMemsetD2D32_wrap(CUdeviceptr_b dst, size_t pitch,
                                unsigned int value, size_t width,
                                size_t height) {
  for (size_t row = 0; row < height; ++row) {
    hipError_t err = hipMemsetD32(
        reinterpret_cast<void*>(dst + row * pitch), value, width);
    if (err != hipSuccess) return static_cast<int>(err);
  }
  return 0;
}

inline int hipMemsetD2D8Async_wrap(CUdeviceptr_b dst, size_t pitch,
                                    unsigned char value, size_t width,
                                    size_t height, CUstream_b stream) {
  return hipMemset2DAsync(reinterpret_cast<void*>(dst), pitch, value, width,
                           height, reinterpret_cast<hipStream_t>(stream));
}

inline int hipMemsetD2D16Async_wrap(CUdeviceptr_b dst, size_t pitch,
                                     unsigned short value, size_t width,
                                     size_t height, CUstream_b stream) {
  for (size_t row = 0; row < height; ++row) {
    hipError_t err = hipMemsetD16Async(
        reinterpret_cast<void*>(dst + row * pitch), value, width,
        reinterpret_cast<hipStream_t>(stream));
    if (err != hipSuccess) return static_cast<int>(err);
  }
  return 0;
}

inline int hipMemsetD2D32Async_wrap(CUdeviceptr_b dst, size_t pitch,
                                     unsigned int value, size_t width,
                                     size_t height, CUstream_b stream) {
  for (size_t row = 0; row < height; ++row) {
    hipError_t err = hipMemsetD32Async(
        reinterpret_cast<void*>(dst + row * pitch), value, width,
        reinterpret_cast<hipStream_t>(stream));
    if (err != hipSuccess) return static_cast<int>(err);
  }
  return 0;
}

}  // anonymous namespace

#endif  // defined(__HIP__)

// --- HIP backend loader (uses dlsym, works without HIP headers) ---

inline Backend loadHipBackend() {
  Backend b{};
  memset(&b, 0, sizeof(b));

  b.lib = dlopen("libamdhip64.so", RTLD_LAZY | RTLD_GLOBAL);
  if (!b.lib) {
    b.lib = dlopen("libamdhip64.so.5", RTLD_LAZY | RTLD_GLOBAL);
  }
  if (!b.lib) {
    return b;
  }

#define LOAD(name, hip_name) \
  b.name = reinterpret_cast<decltype(Backend::name)>(dlsym(b.lib, hip_name))

  LOAD(init, "hipInit");
  LOAD(driverGetVersion, "hipDriverGetVersion");
  LOAD(getErrorString, "hipDrvGetErrorString");
  LOAD(getErrorName, "hipDrvGetErrorName");
  LOAD(deviceGetCount, "hipGetDeviceCount");
  LOAD(deviceGet, "hipDeviceGet");
#if defined(__HIP__)
  b.deviceGetAttribute = reinterpret_cast<decltype(Backend::deviceGetAttribute)>(
      hipDeviceGetAttribute_wrap);
#else
  {
    using hipGetAttrFn = int (*)(int*, int, int);
    static hipGetAttrFn rawGetAttr =
        reinterpret_cast<hipGetAttrFn>(dlsym(b.lib, "hipDeviceGetAttribute"));
    b.deviceGetAttribute = [](int* pi, int attr, int device) -> CUresult_b {
      if (!rawGetAttr) return 1;
      return rawGetAttr(pi, cudaToHipDeviceAttribute(attr), device);
    };
  }
#endif
  LOAD(deviceGetName, "hipDeviceGetName");
  LOAD(deviceGetUuid, "hipDeviceGetUuid");
  LOAD(deviceGetPCIBusId, "hipDeviceGetPCIBusId");
  LOAD(deviceGetByPCIBusId, "hipDeviceGetByPCIBusId");
  LOAD(deviceTotalMem, "hipDeviceTotalMem");
  LOAD(deviceGetDefaultMemPool, "hipDeviceGetDefaultMemPool");
  LOAD(deviceGetMemPool, "hipDeviceGetMemPool");
  LOAD(deviceSetMemPool, "hipDeviceSetMemPool");
  LOAD(deviceGetLuid, "hipDeviceGetLuid");
  LOAD(deviceGetP2PAttribute, "hipDeviceGetP2PAttribute");
  LOAD(deviceCanAccessPeer, "hipDeviceCanAccessPeer");
  LOAD(deviceGetTexture1DLinearMaxWidth, "hipDeviceGetTexture1DLinearMaxWidth");
  LOAD(deviceGraphMemTrim, "hipDeviceGraphMemTrim");
  LOAD(deviceGetGraphMemAttribute, "hipDeviceGetGraphMemAttribute");
  LOAD(deviceSetGraphMemAttribute, "hipDeviceSetGraphMemAttribute");
  LOAD(deviceGetExecAffinitySupport, "hipDeviceGetExecAffinitySupport");

  LOAD(ctxCreate, "hipCtxCreate");
  LOAD(ctxDestroy, "hipCtxDestroy");
  LOAD(ctxGetApiVersion, "hipCtxGetApiVersion");
  LOAD(ctxGetCacheConfig, "hipCtxGetCacheConfig");
  LOAD(ctxSetCacheConfig, "hipCtxSetCacheConfig");
  LOAD(ctxGetCurrent, "hipCtxGetCurrent");
  LOAD(ctxSetCurrent, "hipCtxSetCurrent");
  LOAD(ctxPopCurrent, "hipCtxPopCurrent");
  LOAD(ctxPushCurrent, "hipCtxPushCurrent");
  LOAD(ctxEnablePeerAccess, "hipCtxEnablePeerAccess");
  LOAD(ctxDisablePeerAccess, "hipCtxDisablePeerAccess");
  LOAD(ctxGetDevice, "hipCtxGetDevice");
  LOAD(ctxGetLimit, "hipDeviceGetLimit");
  LOAD(ctxSetLimit, "hipDeviceSetLimit");
  LOAD(ctxSynchronize, "hipCtxSynchronize");

  LOAD(memAlloc, "hipMalloc");
  LOAD(memAllocManaged, "hipMallocManaged");
  LOAD(memAllocAsync, "hipMallocAsync");
  LOAD(memFree, "hipFree");
  LOAD(memFreeAsync, "hipFreeAsync");
  LOAD(memHostAlloc, "hipHostMalloc");
  LOAD(memHostFree, "hipHostFree");
  LOAD(memHostRegister, "hipHostRegister");
  LOAD(memHostUnregister, "hipHostUnregister");
  LOAD(memHostGetDevicePointer, "hipHostGetDevicePointer");
  LOAD(memGetInfo, "hipMemGetInfo");
  LOAD(memPrefetchAsync, "hipMemPrefetchAsync");

  LOAD(memcpyHtoD, "hipMemcpyHtoD");
  LOAD(memcpyDtoH, "hipMemcpyDtoH");
  LOAD(memcpyAsync, "hipMemcpyDtoDAsync");
  LOAD(memcpyHtoDAsync, "hipMemcpyHtoDAsync");
  LOAD(memcpyDtoHAsync, "hipMemcpyDtoHAsync");
  LOAD(memcpy2DAsync, "hipMemcpy2DAsync");

  LOAD(memsetD8, "hipMemsetD8");
  LOAD(memsetD16, "hipMemsetD16");
  LOAD(memsetD32, "hipMemsetD32");
  LOAD(memsetD8Async, "hipMemsetD8Async");
  LOAD(memsetD16Async, "hipMemsetD16Async");
  LOAD(memsetD32Async, "hipMemsetD32Async");
#if defined(__HIP__)
  b.memsetD2D8 = reinterpret_cast<decltype(Backend::memsetD2D8)>(hipMemsetD2D8_wrap);
  b.memsetD2D16 = reinterpret_cast<decltype(Backend::memsetD2D16)>(hipMemsetD2D16_wrap);
  b.memsetD2D32 = reinterpret_cast<decltype(Backend::memsetD2D32)>(hipMemsetD2D32_wrap);
  b.memsetD2D8Async = reinterpret_cast<decltype(Backend::memsetD2D8Async)>(hipMemsetD2D8Async_wrap);
  b.memsetD2D16Async = reinterpret_cast<decltype(Backend::memsetD2D16Async)>(hipMemsetD2D16Async_wrap);
  b.memsetD2D32Async = reinterpret_cast<decltype(Backend::memsetD2D32Async)>(hipMemsetD2D32Async_wrap);
#endif

  LOAD(pointerSetAttribute, "hipPointerSetAttribute");
  LOAD(pointerGetAttribute, "hipPointerGetAttribute");
  LOAD(pointerGetAttributes, "hipPointerGetAttributes");

  LOAD(arrayCreate, "hipArrayCreate");
  LOAD(array3DCreate, "hipArray3DCreate");
  LOAD(arrayDestroy, "hipArrayDestroy");

  LOAD(streamCreate, "hipStreamCreateWithFlags");
  LOAD(streamCreateWithPriority, "hipStreamCreateWithPriority");
  LOAD(streamDestroy, "hipStreamDestroy");
  LOAD(streamSynchronize, "hipStreamSynchronize");
  LOAD(streamQuery, "hipStreamQuery");
  LOAD(streamWaitEvent, "hipStreamWaitEvent");
  LOAD(streamGetFlags, "hipStreamGetFlags");
  LOAD(streamGetPriority, "hipStreamGetPriority");
  LOAD(streamAddCallback, "hipStreamAddCallback");
  LOAD(streamLaunchHostFunc, "hipLaunchHostFunc");

  LOAD(eventCreate, "hipEventCreateWithFlags");
  LOAD(eventDestroy, "hipEventDestroy");
  LOAD(eventRecord, "hipEventRecord");
  LOAD(eventRecordWithFlags, "hipEventRecordWithFlags");
  LOAD(eventSynchronize, "hipEventSynchronize");
  LOAD(eventElapsedTime, "hipEventElapsedTime");
  LOAD(eventQuery, "hipEventQuery");
  LOAD(streamRecordEvent, "hipEventRecord");

  LOAD(moduleLoad, "hipModuleLoad");
  LOAD(moduleLoadData, "hipModuleLoadData");
  LOAD(moduleUnload, "hipModuleUnload");
  LOAD(moduleGetFunction, "hipModuleGetFunction");
  LOAD(moduleGetGlobal, "hipModuleGetGlobal");

  LOAD(funcGetAttribute, "hipFuncGetAttribute");
  LOAD(funcSetAttribute, "hipFuncSetAttribute");
  LOAD(funcSetCacheConfig, "hipFuncSetCacheConfig");
  LOAD(occupancyMaxActiveBlocksPerMultiprocessor,
       "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");

  LOAD(graphCreate, "hipGraphCreate");
  LOAD(graphDestroy, "hipGraphDestroy");
  LOAD(graphInstantiate, "hipGraphInstantiateWithFlags");
  LOAD(graphLaunch, "hipGraphLaunch");
  LOAD(graphDestroyExec, "hipGraphExecDestroy");
  LOAD(graphDebugDotPrint, "hipGraphDebugDotPrint");
  LOAD(graphAddKernelNode, "hipGraphAddKernelNode");
  LOAD(graphAddHostNode, "hipGraphAddHostNode");
  LOAD(graphAddMemFreeNode, "hipGraphAddMemFreeNode");
  LOAD(graphAddMemAllocNode, "hipGraphAddMemAllocNode");

  LOAD(launchKernel, "hipModuleLaunchKernel");
  LOAD(launchCooperativeKernel, "hipModuleLaunchCooperativeKernel");

  LOAD(getExportTable, "hipGetExportTable");

#undef LOAD

#if defined(__HIP__)
  b.moduleLoadDataEx = reinterpret_cast<decltype(Backend::moduleLoadDataEx)>(
      hipModuleLoadDataEx_wrap);
  b.graphAddMemcpyNode = reinterpret_cast<decltype(Backend::graphAddMemcpyNode)>(
      hipGraphAddMemcpyNode_wrap);
  b.deviceGetArchName = [](char* name, int maxLen, CUdevice_b device) -> CUresult_b {
    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, device);
    if (err != hipSuccess) return static_cast<CUresult_b>(err);
    int written = snprintf(name, maxLen, "%s", prop.gcnArchName);
    if (written < 0 || written >= maxLen) return 1;
    return 0;
  };
#else
  b.moduleLoadDataEx = reinterpret_cast<decltype(Backend::moduleLoadDataEx)>(
      dlsym(b.lib, "hipModuleLoadDataEx"));
  b.graphAddMemcpyNode = reinterpret_cast<decltype(Backend::graphAddMemcpyNode)>(
      dlsym(b.lib, "hipGraphAddMemcpyNode"));
  {
    struct ArchNameHelper {
      static CUresult_b getArchName(char* name, int maxLen,
                                     CUdevice_b device) {
        auto getProps = reinterpret_cast<int (*)(void*, int)>(
            dlsym(hipLib(), "hipGetDevicePropertiesR0600"));
        if (!getProps) return 1;
        char propBuf[1472] = {};
        int err = getProps(propBuf, device);
        if (err != 0) return err;
        const char* archName = propBuf + 1160;
        int written = snprintf(name, maxLen, "%s", archName);
        if (written < 0 || written >= maxLen) return 1;
        return 0;
      }
      static void*& hipLib() {
        static void* lib = nullptr;
        return lib;
      }
    };
    ArchNameHelper::hipLib() = b.lib;
    b.deviceGetArchName = ArchNameHelper::getArchName;
  }
#endif

  b.is_cuda = 0;
  return b;
}

#endif  // CU_BACKEND_H
