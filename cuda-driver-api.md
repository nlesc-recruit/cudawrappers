# CUDA Driver API Coverage for cudawrappers

This document summarizes CUDA Driver API coverage for `cudawrappers` against CUDA 13.3.0. It maps user-facing APIs to wrapper support and highlights internal-only variants that are excluded from coverage percentages.

Reference: https://docs.nvidia.com/cuda/cuda-driver-api/index.html

| Section | Coverage |
|---|---|
| Error Handling | ✅ |
| Initialization | ✅ |
| Version Management | ✅ |
| Device Management | ✅ |
| Unified Addressing | ✅ |
| Event Management | ✅ |
| Context Management | ⌛ 48% |
| Stream Memory Operations | ⌛ 43% |
| Module Management | ⌛ 42% |
| Peer Context Memory Access | ⌛ 40% |
| Memory Management | ⌛ 32% |
| Stream Management | ⌛ 28% |
| Execution Control | ⌛ 22% |
| Occupancy | ⌛ 14% |
| Graph Management | ⌛ 10% |
| Other | ⌛ 6% |

> Internal-only variants such as `_v2`, `_v3`, and `_ptsz` are listed in each section and excluded from the coverage percentage calculations.

### Not Implemented Sections
- Primary Context Management
- Library Management
- Virtual Memory Management
- Stream Ordered Memory Allocator
- Multicast Object Management
- Logical Endpoint
- External Resource Interoperability
- Texture Object Management
- Surface Object Management
- Tensor Map Object Management
- Graphics Interoperability
- Driver Entry Point Access
- Coredump Attributes Control API
- Green Contexts
- Error Log Management Functions
- CUDA Checkpointing
- Profiler Control

### Deprecated APIs
- Texture Reference Management [DEPRECATED] — not implemented

## Error Handling

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuGetErrorName` | `cu::getErrorName()` |
| `cuGetErrorString` | Error::what() (internal cu::Error wrapper) |

## Initialization

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuInit` | cu::init() |

## Version Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuDriverGetVersion` | cu::driverGetVersion() |

## Device Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuDeviceComputeCapability` | Device::getComputeCapability() |
| `cuDeviceGet` | Device::Device(unsigned int) |
| `cuDeviceGetAttribute` | Device::getAttribute() |
| `cuDeviceGetByPCIBusId` | Device::getByPCIBusId() |
| `cuDeviceGetCount` | Device::getCount() |
| `cuDeviceGetDefaultMemPool` | Device::getDefaultMemPool() |
| `cuDeviceGetDevResource` | Device::getDevResource() |
| `cuDeviceGetExecAffinitySupport` | Device::getExecAffinitySupport() |
| `cuDeviceGetGraphMemAttribute` | Device::getGraphMemAttribute() |
| `cuDeviceGetLuid` | Device::getLuid() |
| `cuDeviceGetMemPool` | Device::getMemPool() |
| `cuDeviceGetName` | Device::getName() |
| `cuDeviceGetNvSciSyncAttributes` | Device::getNvSciSyncAttributes() |
| `cuDeviceGetPCIBusId` | Device::getPCIBusId() |
| `cuDeviceGetProperties` | Device::getProperties() |
| `cuDeviceGetTexture1DLinearMaxWidth` | Device::getTexture1DLinearMaxWidth() |
| `cuDeviceGetUuid` | Device::getUuid() |
| `cuDeviceGraphMemTrim` | Device::graphMemTrim() |
| `cuDeviceRegisterAsyncNotification` | Device::registerAsyncNotification() |
| `cuDeviceSetGraphMemAttribute` | Device::setGraphMemAttribute() |
| `cuDeviceSetMemPool` | Device::setMemPool() |
| `cuDeviceTotalMem` | Device::totalMem() |
| `cuDeviceUnregisterAsyncNotification` | Device::unregisterAsyncNotification() |

## Primary Context Management
| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuDevicePrimaryCtxGetState` | Missing |
| `cuDevicePrimaryCtxRelease` | Missing |
| `cuDevicePrimaryCtxReset` | Missing |
| `cuDevicePrimaryCtxRetain` | Missing |
| `cuDevicePrimaryCtxSetFlags` | Missing |

## Context Management

> Internal APIs not counted in coverage: `cuCtxCreate_v2`, `cuCtxCreate_v3`, `cuCtxGetDevice_v2`, `cuCtxSynchronize_v2`

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuCtxAttach` | Missing |
| `cuCtxCreate` | Context::Context(int flags, Device &device) |
| `cuCtxCreate_v2` | Missing |
| `cuCtxCreate_v3` | Missing |
| `cuCtxDestroy` | Context destructor |
| `cuCtxDetach` | Missing |
| `cuCtxFromGreenCtx` | Missing |
| `cuCtxGetApiVersion` | Context::getApiVersion() |
| `cuCtxGetCacheConfig` | Context::getCacheConfig() |
| `cuCtxGetCurrent` | Context::getCurrent() |
| `cuCtxGetDevResource` | Missing |
| `cuCtxGetDevice` | Context::getDevice() |
| `cuCtxGetDevice_v2` | Missing |
| `cuCtxGetExecAffinity` | Missing |
| `cuCtxGetFlags` | Missing |
| `cuCtxGetId` | Missing |
| `cuCtxGetLimit` | Context::getLimit() |
| `cuCtxGetSharedMemConfig` | Missing |
| `cuCtxGetStreamPriorityRange` | Missing |
| `cuCtxPopCurrent` | Context::popCurrent() |
| `cuCtxPushCurrent` | Context::pushCurrent() |
| `cuCtxRecordEvent` | Missing |
| `cuCtxResetPersistingL2Cache` | Missing |
| `cuCtxSetCacheConfig` | Context::setCacheConfig() |
| `cuCtxSetCurrent` | Context::setCurrent() |
| `cuCtxSetFlags` | Missing |
| `cuCtxSetLimit` | Context::setLimit() |
| `cuCtxSetSharedMemConfig` | Missing |
| `cuCtxSynchronize` | Context::synchronize() |
| `cuCtxSynchronize_v2` | Missing |
| `cuCtxWaitEvent` | Missing |

## Module Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuLinkComplete` | Missing |
| `cuLinkDestroy` | Missing |
| `cuModuleEnumerateFunctions` | Missing |
| `cuModuleGetFunction` | Function::Function(const Module &, const char *) |
| `cuModuleGetFunctionCount` | Missing |
| `cuModuleGetGlobal` | Module::getGlobal() |
| `cuModuleGetLoadingMode` | Missing |
| `cuModuleGetSurfRef` | Missing |
| `cuModuleGetTexRef` | Missing |
| `cuModuleLoad` | Module::Module(const char *) |
| `cuModuleLoadData` | Module::Module(const void *) |
| `cuModuleLoadDataEx` | Module::Module(const void *, Module::optionmap_t &) |
| `cuModuleLoadFatBinary` | Missing |
| `cuModuleUnload` | Module destructor |

## Library Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuLibraryEnumerateKernels` | Missing |
| `cuLibraryGetGlobal` | Missing |
| `cuLibraryGetKernel` | Missing |
| `cuLibraryGetKernelCount` | Missing |
| `cuLibraryGetManaged` | Missing |
| `cuLibraryGetModule` | Missing |
| `cuLibraryGetUnifiedFunction` | Missing |
| `cuLibraryLoadData` | Missing |
| `cuLibraryLoadFromFile` | Missing |
| `cuLibraryUnload` | Missing |

## Memory Management

> Internal APIs not counted in coverage: `cuMemPrefetchAsync_ptsz`, `cuMemPrefetchAsync_v2`

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuMemAlloc` | DeviceMemory::DeviceMemory(size_t, CUmemorytype, unsigned int) |
| `cuMemAllocAsync` | Stream::memAllocAsync() |
| `cuMemAllocFromPoolAsync` | Missing |
| `cuMemAllocHost` | Missing |
| `cuMemAllocManaged` | DeviceMemory::DeviceMemory(size_t, CUmemorytype, unsigned int) |
| `cuMemAllocPitch` | Missing |
| `cuMemBatchDecompressAsync` | Missing |
| `cuMemCreate` | Missing |
| `cuMemDiscardAndPrefetchBatchAsync` | Missing |
| `cuMemDiscardBatchAsync` | Missing |
| `cuMemExportToShareableHandle` | Missing |
| `cuMemFree` | DeviceMemory destructor |
| `cuMemFreeAsync` | Stream::memFreeAsync() |
| `cuMemFreeHost` | HostMemory destructor |
| `cuMemGetAccess` | Missing |
| `cuMemGetAddressRange` | Missing |
| `cuMemGetAllocationGranularity` | Missing |
| `cuMemGetAllocationPropertiesFromHandle` | Missing |
| `cuMemGetDefaultMemPool` | Missing |
| `cuMemGetHandleForAddressRange` | Missing |
| `cuMemGetInfo` | Context::getFreeMemory()/Context::getTotalMemory() |
| `cuMemGetMemPool` | Missing |
| `cuMemHostAlloc` | HostMemory::HostMemory(size_t, unsigned int) |
| `cuMemHostGetDevicePointer` | DeviceMemory::DeviceMemory(const HostMemory &) |
| `cuMemHostGetFlags` | Missing |
| `cuMemHostRegister` | HostMemory::HostMemory(void *, size_t, unsigned int) |
| `cuMemHostUnregister` | HostMemory destructor |
| `cuMemImportFromShareableHandle` | Missing |
| `cuMemMapArrayAsync` | Missing |
| `cuMemPrefetchAsync` | Stream::memPrefetchAsync(DeviceMemory &, size_t) / Stream::memPrefetchAsync(DeviceMemory &, size_t, Device &) |
| `cuMemPrefetchAsync_ptsz` | Missing |
| `cuMemPrefetchAsync_v2` | Missing |
| `cuMemPrefetchBatchAsync` | Missing |
| `cuMemRangeGetAttribute` | Missing |
| `cuMemRangeGetAttributes` | Missing |
| `cuMemRelease` | Missing |
| `cuMemRetainAllocationHandle` | Missing |
| `cuMemSetAccess` | Missing |
| `cuMemSetMemPool` | Missing |

## Virtual Memory Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuMemAddressFree` | Missing |
| `cuMemAddressReserve` | Missing |
| `cuMemMap` | Missing |
| `cuMemUnmap` | Missing |

## Stream Ordered Memory Allocator

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuMemPoolCreate` | Missing |
| `cuMemPoolDestroy` | Missing |
| `cuMemPoolExportPointer` | Missing |
| `cuMemPoolExportToShareableHandle` | Missing |
| `cuMemPoolGetAccess` | Missing |
| `cuMemPoolGetAttribute` | Missing |
| `cuMemPoolImportFromShareableHandle` | Missing |
| `cuMemPoolImportPointer` | Missing |
| `cuMemPoolSetAccess` | Missing |
| `cuMemPoolSetAttribute` | Missing |
| `cuMemPoolTrimTo` | Missing |

## Multicast Object Management

> Internal APIs not counted in coverage: `cuMulticastBindAddr_v2`, `cuMulticastBindMem_v2`

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuMulticastAddDevice` | Missing |
| `cuMulticastBindAddr` | Missing |
| `cuMulticastBindAddr_v2` | Missing |
| `cuMulticastBindMem` | Missing |
| `cuMulticastBindMem_v2` | Missing |
| `cuMulticastCreate` | Missing |
| `cuMulticastGetGranularity` | Missing |
| `cuMulticastUnbind` | Missing |

## Logical Endpoint

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuLogicalEndpointAddDevice` | Missing |
| `cuLogicalEndpointBindAddr` | Missing |
| `cuLogicalEndpointBindMem` | Missing |
| `cuLogicalEndpointCreate` | Missing |
| `cuLogicalEndpointDestroy` | Missing |
| `cuLogicalEndpointExport` | Missing |
| `cuLogicalEndpointGetLimits` | Missing |
| `cuLogicalEndpointIdRelease` | Missing |
| `cuLogicalEndpointIdReserve` | Missing |
| `cuLogicalEndpointImport` | Missing |
| `cuLogicalEndpointQuery` | Missing |
| `cuLogicalEndpointUnbind` | Missing |

## Unified Addressing

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuDeviceGetHostAtomicCapabilities` | Device::getHostAtomicCapabilities() |
| `cuPointerGetAttribute` | Wrapper::checkPointerAccess() |
| `cuPointerGetAttributes` | Wrapper::checkPointerAccess() |
| `cuPointerSetAttribute` | pointerSetAttribute() |

## Stream Management

> Internal APIs not counted in coverage: `cuStreamBatchMemOp_ptsz`, `cuStreamBatchMemOp_v2`, `cuStreamBeginCapture_ptsz`, `cuStreamBeginCapture_v2`, `cuStreamGetCaptureInfo_ptsz`, `cuStreamGetCaptureInfo_v2`, `cuStreamGetCaptureInfo_v2_ptsz`, `cuStreamGetCaptureInfo_v3`, `cuStreamGetCtx_v2`, `cuStreamUpdateCaptureDependencies_ptsz`, `cuStreamUpdateCaptureDependencies_v2`, `cuStreamWaitValue32_ptsz`, `cuStreamWaitValue32_v2`, `cuStreamWaitValue64_ptsz`, `cuStreamWaitValue64_v2`, `cuStreamWriteValue32_ptsz`, `cuStreamWriteValue32_v2`, `cuStreamWriteValue64_ptsz`, `cuStreamWriteValue64_v2`

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuStreamAddCallback` | Stream::addCallback() |
| `cuStreamAttachMemAsync` | Missing |
| `cuStreamBatchMemOp` | Stream::batchMemOp() |
| `cuStreamBatchMemOp_ptsz` | Missing |
| `cuStreamBatchMemOp_v2` | Missing |
| `cuStreamBeginCapture` | Missing |
| `cuStreamBeginCaptureToCig` | Missing |
| `cuStreamBeginCaptureToGraph` | Missing |
| `cuStreamBeginCapture_ptsz` | Missing |
| `cuStreamBeginCapture_v2` | Missing |
| `cuStreamBeginRecaptureToGraph` | Missing |
| `cuStreamCopyAttributes` | Missing |
| `cuStreamCreate` | Stream::Stream(unsigned int) |
| `cuStreamCreateWithPriority` | Missing |
| `cuStreamDestroy` | Stream destructor |
| `cuStreamEndCapture` | Missing |
| `cuStreamEndCaptureToCig` | Missing |
| `cuStreamGetAttribute` | Missing |
| `cuStreamGetCaptureInfo` | Missing |
| `cuStreamGetCaptureInfo_ptsz` | Missing |
| `cuStreamGetCaptureInfo_v2` | Missing |
| `cuStreamGetCaptureInfo_v2_ptsz` | Missing |
| `cuStreamGetCaptureInfo_v3` | Missing |
| `cuStreamGetCtx` | Missing |
| `cuStreamGetCtx_v2` | Missing |
| `cuStreamGetDevResource` | Missing |
| `cuStreamGetDevice` | Missing |
| `cuStreamGetFlags` | Missing |
| `cuStreamGetGreenCtx` | Missing |
| `cuStreamGetId` | Missing |
| `cuStreamGetPriority` | Missing |
| `cuStreamIsCapturing` | Missing |
| `cuStreamQuery` | Stream::query() |
| `cuStreamSetAttribute` | Missing |
| `cuStreamSynchronize` | Stream::synchronize() |
| `cuStreamUpdateCaptureDependencies` | Missing |
| `cuStreamUpdateCaptureDependencies_ptsz` | Missing |
| `cuStreamUpdateCaptureDependencies_v2` | Missing |
| `cuStreamWaitEvent` | Stream::wait(Event &) |
| `cuStreamWaitValue32` | Stream::waitValue32() |
| `cuStreamWaitValue32_ptsz` | Missing |
| `cuStreamWaitValue32_v2` | Missing |
| `cuStreamWaitValue64` | Missing |
| `cuStreamWaitValue64_ptsz` | Missing |
| `cuStreamWaitValue64_v2` | Missing |
| `cuStreamWriteValue32` | Stream::writeValue32() |
| `cuStreamWriteValue32_ptsz` | Missing |
| `cuStreamWriteValue32_v2` | Missing |
| `cuStreamWriteValue64` | Missing |
| `cuStreamWriteValue64_ptsz` | Missing |
| `cuStreamWriteValue64_v2` | Missing |

## Event Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuEventCreate` | Event::Event(unsigned int) |
| `cuEventDestroy` | Event destructor |
| `cuEventElapsedTime` | Event::elapsedTime() |
| `cuEventQuery` | Event::query() |
| `cuEventRecord` | Event::record() / Stream::record(Event &) |
| `cuEventRecordWithFlags` | Event::record(Stream &, unsigned int) / Stream::record(Event &, unsigned int) |
| `cuEventSynchronize` | Event::synchronize() |

## External Resource Interoperability

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuDestroyExternalMemory` | Missing |
| `cuDestroyExternalSemaphore` | Missing |
| `cuExternalMemoryGetMappedBuffer` | Missing |
| `cuExternalMemoryGetMappedMipmappedArray` | Missing |
| `cuImportExternalMemory` | Missing |
| `cuImportExternalSemaphore` | Missing |

## Stream Memory Operations

> Internal APIs not counted in coverage: `cuMemcpy2DAsync_v2`, `cuMemcpy2DUnaligned_v2`, `cuMemcpy2D_v2`, `cuMemcpy3DAsync_v2`, `cuMemcpy3DBatchAsync_ptsz`, `cuMemcpy3DBatchAsync_v2`, `cuMemcpy3D_v2`, `cuMemcpyAtoA_v2`, `cuMemcpyAtoD_v2`, `cuMemcpyAtoHAsync_v2`, `cuMemcpyAtoH_v2`, `cuMemcpyBatchAsync_ptsz`, `cuMemcpyBatchAsync_v2`, `cuMemcpyDtoA_v2`, `cuMemcpyDtoDAsync_v2`, `cuMemcpyDtoD_v2`, `cuMemcpyDtoHAsync_v2`, `cuMemcpyDtoH_v2`, `cuMemcpyHtoAAsync_v2`, `cuMemcpyHtoA_v2`, `cuMemcpyHtoDAsync_v2`, `cuMemcpyHtoD_v2`, `cuMemsetD16_v2`, `cuMemsetD2D16_v2`, `cuMemsetD2D32_v2`, `cuMemsetD2D8_v2`, `cuMemsetD32_v2`, `cuMemsetD8_v2`

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuMemAdvise` | Missing |
| `cuMemcpy` | Missing |
| `cuMemcpy2D` | Missing |
| `cuMemcpy2DAsync` | Stream::memcpyHtoD2DAsync / Stream::memcpyDtoH2DAsync |
| `cuMemcpy2DAsync_v2` | Missing |
| `cuMemcpy2DUnaligned` | Missing |
| `cuMemcpy2DUnaligned_v2` | Missing |
| `cuMemcpy2D_v2` | Missing |
| `cuMemcpy3D` | Missing |
| `cuMemcpy3DAsync` | Missing |
| `cuMemcpy3DAsync_v2` | Missing |
| `cuMemcpy3DBatchAsync` | Missing |
| `cuMemcpy3DBatchAsync_ptsz` | Missing |
| `cuMemcpy3DBatchAsync_v2` | Missing |
| `cuMemcpy3DPeer` | Missing |
| `cuMemcpy3DPeerAsync` | Missing |
| `cuMemcpy3DWithAttributesAsync` | Missing |
| `cuMemcpy3D_v2` | Missing |
| `cuMemcpyAsync` | Stream::memcpyHtoHAsync(void *, const void *, size_t) / Stream::memcpyDtoDAsync(DeviceMemory &, DeviceMemory &, size_t) |
| `cuMemcpyAtoA` | Missing |
| `cuMemcpyAtoA_v2` | Missing |
| `cuMemcpyAtoD` | Missing |
| `cuMemcpyAtoD_v2` | Missing |
| `cuMemcpyAtoH` | Missing |
| `cuMemcpyAtoHAsync` | Missing |
| `cuMemcpyAtoHAsync_v2` | Missing |
| `cuMemcpyAtoH_v2` | Missing |
| `cuMemcpyBatchAsync` | Missing |
| `cuMemcpyBatchAsync_ptsz` | Missing |
| `cuMemcpyBatchAsync_v2` | Missing |
| `cuMemcpyDtoA` | Missing |
| `cuMemcpyDtoA_v2` | Missing |
| `cuMemcpyDtoD` | Missing |
| `cuMemcpyDtoDAsync` | Missing |
| `cuMemcpyDtoDAsync_v2` | Missing |
| `cuMemcpyDtoD_v2` | Missing |
| `cuMemcpyDtoH` | cu::memcpyDtoH(void *, CUdeviceptr, size_t) |
| `cuMemcpyDtoHAsync` | Stream::memcpyDtoHAsync() |
| `cuMemcpyDtoHAsync_v2` | Missing |
| `cuMemcpyDtoH_v2` | Missing |
| `cuMemcpyHtoA` | Missing |
| `cuMemcpyHtoAAsync` | Missing |
| `cuMemcpyHtoAAsync_v2` | Missing |
| `cuMemcpyHtoA_v2` | Missing |
| `cuMemcpyHtoD` | cu::memcpyHtoD(CUdeviceptr, const void *, size_t) |
| `cuMemcpyHtoDAsync` | Stream::memcpyHtoDAsync() |
| `cuMemcpyHtoDAsync_v2` | Missing |
| `cuMemcpyHtoD_v2` | Missing |
| `cuMemcpyPeer` | Missing |
| `cuMemcpyPeerAsync` | Missing |
| `cuMemcpyWithAttributesAsync` | Missing |
| `cuMemsetD16` | DeviceMemory::memset(unsigned short, size_t) |
| `cuMemsetD16Async` | Stream::memsetAsync(DeviceMemory &, unsigned short, size_t) |
| `cuMemsetD16_v2` | Missing |
| `cuMemsetD2D16` | DeviceMemory::memset2D(unsigned short, size_t, size_t, size_t) |
| `cuMemsetD2D16Async` | Stream::memset2DAsync(DeviceMemory &, unsigned short, size_t, size_t, size_t) |
| `cuMemsetD2D16_v2` | Missing |
| `cuMemsetD2D32` | DeviceMemory::memset2D(unsigned int, size_t, size_t, size_t) |
| `cuMemsetD2D32Async` | Stream::memset2DAsync(DeviceMemory &, unsigned int, size_t, size_t, size_t) |
| `cuMemsetD2D32_v2` | Missing |
| `cuMemsetD2D8` | DeviceMemory::memset2D(unsigned char, size_t, size_t, size_t) |
| `cuMemsetD2D8Async` | Stream::memset2DAsync(DeviceMemory &, unsigned char, size_t, size_t, size_t) |
| `cuMemsetD2D8_v2` | Missing |
| `cuMemsetD32` | DeviceMemory::memset(unsigned int, size_t) |
| `cuMemsetD32Async` | Stream::memsetAsync(DeviceMemory &, unsigned int, size_t) |
| `cuMemsetD32_v2` | Missing |
| `cuMemsetD8` | DeviceMemory::memset(unsigned char, size_t) |
| `cuMemsetD8Async` | Stream::memsetAsync(DeviceMemory &, unsigned char, size_t) |
| `cuMemsetD8_v2` | Missing |

## Execution Control

> Internal APIs not counted in coverage: `cuLaunchHostFunc_v2`

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuFuncGetAttribute` | Function::getAttribute() |
| `cuFuncGetModule` | Missing |
| `cuFuncGetName` | Missing |
| `cuFuncGetParamCount` | Missing |
| `cuFuncGetParamInfo` | Missing |
| `cuFuncIsLoaded` | Missing |
| `cuFuncLoad` | Missing |
| `cuFuncSetAttribute` | Function::setAttribute() |
| `cuFuncSetBlockShape` | Missing |
| `cuFuncSetCacheConfig` | Function::setCacheConfig() |
| `cuFuncSetSharedMemConfig` | Missing |
| `cuFuncSetSharedSize` | Missing |
| `cuKernelGetAttribute` | Missing |
| `cuKernelGetFunction` | Missing |
| `cuKernelGetLibrary` | Missing |
| `cuKernelGetName` | Missing |
| `cuKernelGetParamCount` | Missing |
| `cuKernelGetParamInfo` | Missing |
| `cuKernelSetAttribute` | Missing |
| `cuKernelSetCacheConfig` | Missing |
| `cuLaunchCooperativeKernel` | Stream::launchCooperativeKernel(Function &, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, const std::vector<const void *> &) |
| `cuLaunchHostFunc` | Stream::launchHostFunc() |
| `cuLaunchHostFunc_v2` | Missing |
| `cuLaunchKernel` | Stream::launchKernel(Function &, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, const std::vector<const void *> &) |
| `cuLaunchKernelEx` | Missing |
| `cuLinkAddData` | Missing |
| `cuLinkAddFile` | Missing |
| `cuLinkCreate` | Missing |

## Graph Management

> Internal APIs not counted in coverage: `cuGraphInstantiate_v2`

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuGraphAddBatchMemOpNode` | Missing |
| `cuGraphAddChildGraphNode` | Missing |
| `cuGraphAddDependencies` | Missing |
| `cuGraphAddEmptyNode` | Missing |
| `cuGraphAddEventRecordNode` | Missing |
| `cuGraphAddEventWaitNode` | Missing |
| `cuGraphAddExternalSemaphoresSignalNode` | Missing |
| `cuGraphAddExternalSemaphoresWaitNode` | Missing |
| `cuGraphAddHostNode` | Graph::addHostNode() |
| `cuGraphAddKernelNode` | Graph::addKernelNode() |
| `cuGraphAddMemAllocNode` | Graph::addMemAllocNode() |
| `cuGraphAddMemFreeNode` | Graph::addDevMemFreeNode() |
| `cuGraphAddMemcpyNode` | Graph::addMemCpyNode() |
| `cuGraphAddMemsetNode` | Missing |
| `cuGraphAddNode` | Missing |
| `cuGraphBatchMemOpNodeGetParams` | Missing |
| `cuGraphBatchMemOpNodeSetParams` | Missing |
| `cuGraphChildGraphNodeGetGraph` | Missing |
| `cuGraphClone` | Missing |
| `cuGraphConditionalHandleCreate` | Missing |
| `cuGraphCreate` | Graph::Graph(Context &, unsigned int) |
| `cuGraphDebugDotPrint` | Graph::debugDotPrint() |
| `cuGraphDestroy` | Graph destructor |
| `cuGraphDestroyNode` | Missing |
| `cuGraphEventRecordNodeGetEvent` | Missing |
| `cuGraphEventRecordNodeSetEvent` | Missing |
| `cuGraphEventWaitNodeGetEvent` | Missing |
| `cuGraphEventWaitNodeSetEvent` | Missing |
| `cuGraphExecBatchMemOpNodeSetParams` | Missing |
| `cuGraphExecChildGraphNodeSetParams` | Missing |
| `cuGraphExecDestroy` | Missing |
| `cuGraphExecEventRecordNodeSetEvent` | Missing |
| `cuGraphExecEventWaitNodeSetEvent` | Missing |
| `cuGraphExecExternalSemaphoresSignalNodeSetParams` | Missing |
| `cuGraphExecExternalSemaphoresWaitNodeSetParams` | Missing |
| `cuGraphExecGetFlags` | Missing |
| `cuGraphExecGetId` | Missing |
| `cuGraphExecHostNodeSetParams` | Missing |
| `cuGraphExecKernelNodeSetParams` | Missing |
| `cuGraphExecMemcpyNodeSetParams` | Missing |
| `cuGraphExecMemsetNodeSetParams` | Missing |
| `cuGraphExecNodeSetParams` | Missing |
| `cuGraphExecUpdate` | Missing |
| `cuGraphExternalSemaphoresSignalNodeGetParams` | Missing |
| `cuGraphExternalSemaphoresSignalNodeSetParams` | Missing |
| `cuGraphExternalSemaphoresWaitNodeGetParams` | Missing |
| `cuGraphExternalSemaphoresWaitNodeSetParams` | Missing |
| `cuGraphGetEdges` | Missing |
| `cuGraphGetId` | Missing |
| `cuGraphGetNodes` | Missing |
| `cuGraphGetRootNodes` | Missing |
| `cuGraphHostNodeGetParams` | Missing |
| `cuGraphHostNodeSetParams` | Missing |
| `cuGraphInstantiate` | Missing |
| `cuGraphInstantiateWithParams` | Missing |
| `cuGraphInstantiate_v2` | Missing |
| `cuGraphKernelNodeCopyAttributes` | Missing |
| `cuGraphKernelNodeGetAttribute` | Missing |
| `cuGraphKernelNodeGetParams` | Missing |
| `cuGraphKernelNodeSetAttribute` | Missing |
| `cuGraphKernelNodeSetParams` | Missing |
| `cuGraphLaunch` | Stream::graphLaunch() |
| `cuGraphMemAllocNodeGetParams` | Missing |
| `cuGraphMemFreeNodeGetParams` | Missing |
| `cuGraphMemcpyNodeGetParams` | Missing |
| `cuGraphMemcpyNodeSetParams` | Missing |
| `cuGraphMemsetNodeGetParams` | Missing |
| `cuGraphMemsetNodeSetParams` | Missing |
| `cuGraphNodeFindInClone` | Missing |
| `cuGraphNodeGetContainingGraph` | Missing |
| `cuGraphNodeGetDependencies` | Missing |
| `cuGraphNodeGetDependentNodes` | Missing |
| `cuGraphNodeGetEnabled` | Missing |
| `cuGraphNodeGetLocalId` | Missing |
| `cuGraphNodeGetParams` | Missing |
| `cuGraphNodeGetToolsId` | Missing |
| `cuGraphNodeGetType` | Missing |
| `cuGraphNodeSetEnabled` | Missing |
| `cuGraphNodeSetParams` | Missing |
| `cuGraphReleaseUserObject` | Missing |
| `cuGraphRemoveDependencies` | Missing |
| `cuGraphRetainUserObject` | Missing |
| `cuGraphUpload` | Missing |

## Occupancy

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuOccupancyAvailableDynamicSMemPerBlock` | Missing |
| `cuOccupancyMaxActiveBlocksPerMultiprocessor` | Function::occupancyMaxActiveBlocksPerMultiprocessor() |
| `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` | Missing |
| `cuOccupancyMaxActiveClusters` | Missing |
| `cuOccupancyMaxPotentialBlockSize` | Missing |
| `cuOccupancyMaxPotentialBlockSizeWithFlags` | Missing |
| `cuOccupancyMaxPotentialClusterSize` | Missing |

## Texture Reference Management [DEPRECATED]

> Internal APIs not counted in coverage: `cuTexRefSetAddress2D_v2`

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuTexRefCreate` | Missing |
| `cuTexRefDestroy` | Missing |
| `cuTexRefGetAddress` | Missing |
| `cuTexRefGetAddressMode` | Missing |
| `cuTexRefGetArray` | Missing |
| `cuTexRefGetBorderColor` | Missing |
| `cuTexRefGetFilterMode` | Missing |
| `cuTexRefGetFlags` | Missing |
| `cuTexRefGetFormat` | Missing |
| `cuTexRefGetMaxAnisotropy` | Missing |
| `cuTexRefGetMipmapFilterMode` | Missing |
| `cuTexRefGetMipmapLevelBias` | Missing |
| `cuTexRefGetMipmapLevelClamp` | Missing |
| `cuTexRefGetMipmappedArray` | Missing |
| `cuTexRefSetAddress` | Missing |
| `cuTexRefSetAddress2D` | Missing |
| `cuTexRefSetAddress2D_v2` | Missing |
| `cuTexRefSetAddressMode` | Missing |
| `cuTexRefSetArray` | Missing |
| `cuTexRefSetBorderColor` | Missing |
| `cuTexRefSetFilterMode` | Missing |
| `cuTexRefSetFlags` | Missing |
| `cuTexRefSetFormat` | Missing |
| `cuTexRefSetMaxAnisotropy` | Missing |
| `cuTexRefSetMipmapFilterMode` | Missing |
| `cuTexRefSetMipmapLevelBias` | Missing |
| `cuTexRefSetMipmapLevelClamp` | Missing |
| `cuTexRefSetMipmappedArray` | Missing |

## Texture Object Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuTexObjectCreate` | Missing |
| `cuTexObjectDestroy` | Missing |
| `cuTexObjectGetResourceDesc` | Missing |
| `cuTexObjectGetResourceViewDesc` | Missing |
| `cuTexObjectGetTextureDesc` | Missing |

## Surface Object Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuSurfObjectCreate` | Missing |
| `cuSurfObjectDestroy` | Missing |
| `cuSurfObjectGetResourceDesc` | Missing |

## Tensor Map Object Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuTensorMapEncodeIm2col` | Missing |
| `cuTensorMapEncodeIm2colWide` | Missing |
| `cuTensorMapEncodeTiled` | Missing |
| `cuTensorMapReplaceAddress` | Missing |

## Peer Context Memory Access

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuCtxDisablePeerAccess` | Missing |
| `cuCtxEnablePeerAccess` | Missing |
| `cuDeviceCanAccessPeer` | Missing |
| `cuDeviceGetP2PAtomicCapabilities` | Device::getP2PAtomicCapabilities() |
| `cuDeviceGetP2PAttribute` | Device::getP2PAttribute() |

## Graphics Interoperability

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuGraphicsMapResources` | Missing |
| `cuGraphicsResourceGetMappedMipmappedArray` | Missing |
| `cuGraphicsResourceGetMappedPointer` | Missing |
| `cuGraphicsResourceSetMapFlags` | Missing |
| `cuGraphicsSubResourceGetMappedArray` | Missing |
| `cuGraphicsUnmapResources` | Missing |
| `cuGraphicsUnregisterResource` | Missing |

## Driver Entry Point Access

> Internal APIs not counted in coverage: `cuGetProcAddress_v2`, `cuGetProcAddress_v2_ptsz`

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuGetExportTable` | Missing |
| `cuGetProcAddress` | Missing |
| `cuGetProcAddress_v2` | Missing |
| `cuGetProcAddress_v2_ptsz` | Missing |

## Coredump Attributes Control API

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuCoredumpDeregisterCompleteCallback` | Missing |
| `cuCoredumpDeregisterStartCallback` | Missing |
| `cuCoredumpGetAttribute` | Missing |
| `cuCoredumpGetAttributeGlobal` | Missing |
| `cuCoredumpRegisterCompleteCallback` | Missing |
| `cuCoredumpRegisterStartCallback` | Missing |
| `cuCoredumpSetAttribute` | Missing |
| `cuCoredumpSetAttributeGlobal` | Missing |

## Green Contexts

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuGreenCtxCreate` | Missing |
| `cuGreenCtxDestroy` | Missing |
| `cuGreenCtxGetDevResource` | Missing |
| `cuGreenCtxGetId` | Missing |
| `cuGreenCtxRecordEvent` | Missing |
| `cuGreenCtxStreamCreate` | Missing |
| `cuGreenCtxWaitEvent` | Missing |

## Error Log Management Functions

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuLogsCurrent` | Missing |
| `cuLogsDumpToFile` | Missing |
| `cuLogsDumpToMemory` | Missing |
| `cuLogsRegisterCallback` | Missing |
| `cuLogsUnregisterCallback` | Missing |

## CUDA Checkpointing

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuCheckpointProcessCheckpoint` | Missing |
| `cuCheckpointProcessGetRestoreThreadId` | Missing |
| `cuCheckpointProcessGetState` | Missing |
| `cuCheckpointProcessLock` | Missing |
| `cuCheckpointProcessRestore` | Missing |
| `cuCheckpointProcessUnlock` | Missing |

## Profiler Control

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuProfilerStart` | Missing |
| `cuProfilerStop` | Missing |

## Other

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuArray3DCreate` | Array::Array(unsigned width, unsigned height, unsigned depth, CUarray_format format, unsigned numChannels) |
| `cuArray3DGetDescriptor` | Missing |
| `cuArrayCreate` | Array::Array(unsigned width, unsigned height, CUarray_format format, unsigned numChannels) |
| `cuArrayDestroy` | Array destructor |
| `cuArrayGetDescriptor` | Missing |
| `cuArrayGetMemoryRequirements` | Missing |
| `cuArrayGetPlane` | Missing |
| `cuArrayGetSparseProperties` | Missing |
| `cuDevResourceGenerateDesc` | Missing |
| `cuDevSmResourceSplit` | Missing |
| `cuDevSmResourceSplitByCount` | Missing |
| `cuFlushGPUDirectRDMAWrites` | Missing |
| `cuIpcCloseMemHandle` | Missing |
| `cuIpcGetEventHandle` | Missing |
| `cuIpcGetMemHandle` | Missing |
| `cuIpcOpenEventHandle` | Missing |
| `cuIpcOpenMemHandle` | Missing |
| `cuLaunch` | Missing |
| `cuLaunchCooperativeKernelMultiDevice` | Missing |
| `cuLaunchGrid` | Missing |
| `cuLaunchGridAsync` | Missing |
| `cuMipmappedArrayCreate` | Missing |
| `cuMipmappedArrayDestroy` | Missing |
| `cuMipmappedArrayGetLevel` | Missing |
| `cuMipmappedArrayGetMemoryRequirements` | Missing |
| `cuMipmappedArrayGetSparseProperties` | Missing |
| `cuParamSetSize` | Missing |
| `cuParamSetTexRef` | Missing |
| `cuParamSetf` | Missing |
| `cuParamSeti` | Missing |
| `cuParamSetv` | Missing |
| `cuSignalExternalSemaphoresAsync` | Missing |
| `cuSurfRefGetArray` | Missing |
| `cuSurfRefSetArray` | Missing |
| `cuThreadExchangeStreamCaptureMode` | Missing |
| `cuUserObjectCreate` | Missing |
| `cuUserObjectRelease` | Missing |
| `cuUserObjectRetain` | Missing |
| `cuWaitExternalSemaphoresAsync` | Missing |
| `cudaDeviceGetTexture1DLinearMaxWidth` | Missing |
| `cudaDeviceSynchronize` | Missing |
| `cudaGetLastError` | Missing |
| `cudaGraphKernelNodeSetParam` | Missing |
| `cudaGraphKernelNodeUpdatesApply` | Missing |
| `cudaGraphLaunch` | Missing |
| `cudaGridDependencySynchronize` | Missing |
| `cudaTriggerProgrammaticLaunchCompletion` | Missing |

## Summary

- Total CUDA driver APIs: 547
- Implemented in cudawrappers: 94
- Missing implementations: 453
