# CUDA Driver API Coverage for cudawrappers

This document summarizes CUDA Driver API coverage for `cudawrappers` against CUDA 13.1. It lists the APIs in the upstream Driver API sectioning and records whether a wrapper is represented in the project.

Reference: https://docs.nvidia.com/cuda/cuda-driver-api/index.html


## Error Handling

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuGetErrorName` | `cu::getErrorName(CUresult)` |
| `cuGetErrorString` | `cu::Error::what()` |

## Initialization

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuInit` | `cu::init()` |

## Version Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuDriverGetVersion` | `cu::driverGetVersion()` |

## Device Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuDeviceGet` | `Device::Device(unsigned int)` |
| `cuDeviceGetAttribute` | `Device::getAttribute()` |
| `cuDeviceGetCount` | `Device::getCount()` |
| `cuDeviceGetDefaultMemPool` | `Device::getDefaultMemPool()` |
| `cuDeviceGetExecAffinitySupport` | `Device::getExecAffinitySupport()` |
| `cuDeviceGetHostAtomicCapabilities` | Missing |
| `cuDeviceGetLuid` | `Device::getLuid()` |
| `cuDeviceGetMemPool` | `Device::getMemPool()` |
| `cuDeviceGetName` | `Device::getName()` |
| `cuDeviceGetNvSciSyncAttributes` | `Device::getNvSciSyncAttributes()` |
| `cuDeviceGetTexture1DLinearMaxWidth` | `Device::getTexture1DLinearMaxWidth()` |
| `cuDeviceGetUuid` | `Device::getUuid()` |
| `cuDeviceSetMemPool` | `Device::setMemPool()` |
| `cuDeviceTotalMem` | `Device::totalMem()` |
| `cuFlushGPUDirectRDMAWrites` | Missing |

## Primary Context Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuDevicePrimaryCtxGetState` | Missing |
| `cuDevicePrimaryCtxRelease` | Missing |
| `cuDevicePrimaryCtxReset` | Missing |
| `cuDevicePrimaryCtxRetain` | Missing |
| `cuDevicePrimaryCtxSetFlags` | Missing |

## Context Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuCtxCreate` | `Context::Context(int flags, Device &device)` |
| `cuCtxDestroy` | `Context destructor` |
| `cuCtxGetApiVersion` | `Context::getApiVersion()` |
| `cuCtxGetCacheConfig` | `Context::getCacheConfig()` |
| `cuCtxGetCurrent` | `Context::getCurrent()` |
| `cuCtxGetDevice` | `Context::getDevice()` |
| `cuCtxGetExecAffinity` | Missing |
| `cuCtxGetFlags` | Missing |
| `cuCtxGetId` | Missing |
| `cuCtxGetLimit` | `Context::getLimit()` |
| `cuCtxGetStreamPriorityRange` | Missing |
| `cuCtxPopCurrent` | `Context::popCurrent()` |
| `cuCtxPushCurrent` | `Context::pushCurrent()` |
| `cuCtxRecordEvent` | Missing |
| `cuCtxResetPersistingL2Cache` | Missing |
| `cuCtxSetCacheConfig` | `Context::setCacheConfig()` |
| `cuCtxSetCurrent` | `Context::setCurrent()` |
| `cuCtxSetFlags` | Missing |
| `cuCtxSetLimit` | `Context::setLimit()` |
| `cuCtxSynchronize` | `Context::synchronize()` |
| `cuCtxWaitEvent` | Missing |

## Module Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuLinkAddData` | Missing |
| `cuLinkAddFile` | Missing |
| `cuLinkComplete` | Missing |
| `cuLinkCreate` | Missing |
| `cuLinkDestroy` | Missing |
| `cuModuleEnumerateFunctions` | Missing |
| `cuModuleGetFunction` | `Function::Function(const Module &, const char *)` |
| `cuModuleGetFunctionCount` | Missing |
| `cuModuleGetGlobal` | `Module::getGlobal()` |
| `cuModuleGetLoadingMode` | Missing |
| `cuModuleLoad` | `Module::Module(const char *)` |
| `cuModuleLoadData` | `Module::Module(const void *)` |
| `cuModuleLoadDataEx` | `Module::Module(const void *, Module::optionmap_t &)` |
| `cuModuleLoadFatBinary` | Missing |
| `cuModuleUnload` | `Module destructor` |

## Library Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuKernelGetAttribute` | Missing |
| `cuKernelGetFunction` | Missing |
| `cuKernelGetLibrary` | Missing |
| `cuKernelGetName` | Missing |
| `cuKernelGetParamCount` | Missing |
| `cuKernelGetParamInfo` | Missing |
| `cuKernelSetAttribute` | Missing |
| `cuKernelSetCacheConfig` | `Function::setCacheConfig(CUfunc_cache)` |
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

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuArray3DCreate` | `Array::Array(unsigned width, unsigned height, unsigned depth, CUarray_format format, unsigned numChannels)` |
| `cuArray3DGetDescriptor` | Missing |
| `cuArrayCreate` | `Array::Array(unsigned width, unsigned height, CUarray_format format, unsigned numChannels)` |
| `cuArrayDestroy` | `Array destructor` |
| `cuArrayGetDescriptor` | Missing |
| `cuArrayGetMemoryRequirements` | Missing |
| `cuArrayGetPlane` | Missing |
| `cuArrayGetSparseProperties` | Missing |
| `cuDeviceGetByPCIBusId` | `Device::getByPCIBusId()` |
| `cuDeviceGetPCIBusId` | `Device::getPCIBusId()` |
| `cuDeviceRegisterAsyncNotification` | `Device::registerAsyncNotification()` |
| `cuDeviceUnregisterAsyncNotification` | `Device::unregisterAsyncNotification()` |
| `cuIpcCloseMemHandle` | Missing |
| `cuIpcGetEventHandle` | Missing |
| `cuIpcGetMemHandle` | Missing |
| `cuIpcOpenEventHandle` | Missing |
| `cuIpcOpenMemHandle` | Missing |
| `cuMemAlloc` | `DeviceMemory::DeviceMemory(size_t, CUmemorytype, unsigned int)` |
| `cuMemAllocHost` | Missing |
| `cuMemAllocManaged` | `DeviceMemory::DeviceMemory(size_t, CUmemorytype, unsigned int)` |
| `cuMemAllocPitch` | Missing |
| `cuMemBatchDecompressAsync` | Missing |
| `cuMemFree` | `DeviceMemory destructor` |
| `cuMemFreeHost` | `HostMemory destructor` |
| `cuMemGetAddressRange` | Missing |
| `cuMemGetHandleForAddressRange` | Missing |
| `cuMemGetInfo` | `Context::getFreeMemory() / Context::getTotalMemory()` |
| `cuMemHostAlloc` | `HostMemory::HostMemory(size_t, unsigned int)` |
| `cuMemHostGetDevicePointer` | `DeviceMemory::DeviceMemory(const HostMemory &)` |
| `cuMemHostGetFlags` | Missing |
| `cuMemHostRegister` | `HostMemory::HostMemory(void *, size_t, unsigned int)` |
| `cuMemHostUnregister` | `HostMemory destructor` |
| `cuMemcpy` | Missing |
| `cuMemcpy2D` | Missing |
| `cuMemcpy2DAsync` | `Stream::memcpyHtoD2DAsync / Stream::memcpyDtoH2DAsync` |
| `cuMemcpy2DUnaligned` | Missing |
| `cuMemcpy3D` | Missing |
| `cuMemcpy3DAsync` | Missing |
| `cuMemcpy3DBatchAsync` | Missing |
| `cuMemcpy3DPeer` | Missing |
| `cuMemcpy3DPeerAsync` | Missing |
| `cuMemcpy3DWithAttributesAsync` | Missing |
| `cuMemcpyAsync` | `Stream::memcpyHtoHAsync(void *, const void *, size_t) / Stream::memcpyDtoDAsync(DeviceMemory &, DeviceMemory &, size_t)` |
| `cuMemcpyAtoA` | Missing |
| `cuMemcpyAtoD` | Missing |
| `cuMemcpyAtoH` | Missing |
| `cuMemcpyAtoHAsync` | Missing |
| `cuMemcpyBatchAsync` | Missing |
| `cuMemcpyDtoA` | Missing |
| `cuMemcpyDtoD` | Missing |
| `cuMemcpyDtoDAsync` | Missing |
| `cuMemcpyDtoH` | `cu::memcpyDtoH(void *, CUdeviceptr, size_t)` |
| `cuMemcpyDtoHAsync` | `Stream::memcpyDtoHAsync()` |
| `cuMemcpyHtoA` | Missing |
| `cuMemcpyHtoAAsync` | Missing |
| `cuMemcpyHtoD` | `cu::memcpyHtoD(CUdeviceptr, const void *, size_t)` |
| `cuMemcpyHtoDAsync` | `Stream::memcpyHtoDAsync()` |
| `cuMemcpyPeer` | Missing |
| `cuMemcpyPeerAsync` | Missing |
| `cuMemcpyWithAttributesAsync` | Missing |
| `cuMemsetD16` | `DeviceMemory::memset(unsigned short, size_t)` |
| `cuMemsetD16Async` | `Stream::memsetAsync(DeviceMemory &, unsigned short, size_t)` |
| `cuMemsetD2D16` | `DeviceMemory::memset2D(unsigned short, size_t, size_t, size_t)` |
| `cuMemsetD2D16Async` | `Stream::memset2DAsync(DeviceMemory &, unsigned short, size_t, size_t, size_t)` |
| `cuMemsetD2D32` | `DeviceMemory::memset2D(unsigned int, size_t, size_t, size_t)` |
| `cuMemsetD2D32Async` | `Stream::memset2DAsync(DeviceMemory &, unsigned int, size_t, size_t, size_t)` |
| `cuMemsetD2D8` | `DeviceMemory::memset2D(unsigned char, size_t, size_t, size_t)` |
| `cuMemsetD2D8Async` | `Stream::memset2DAsync(DeviceMemory &, unsigned char, size_t, size_t, size_t)` |
| `cuMemsetD32` | `DeviceMemory::memset(unsigned int, size_t)` |
| `cuMemsetD32Async` | `Stream::memsetAsync(DeviceMemory &, unsigned int, size_t)` |
| `cuMemsetD8` | `DeviceMemory::memset(unsigned char, size_t)` |
| `cuMemsetD8Async` | `Stream::memsetAsync(DeviceMemory &, unsigned char, size_t)` |
| `cuMipmappedArrayCreate` | Missing |
| `cuMipmappedArrayDestroy` | Missing |
| `cuMipmappedArrayGetLevel` | Missing |
| `cuMipmappedArrayGetMemoryRequirements` | Missing |
| `cuMipmappedArrayGetSparseProperties` | Missing |

## Virtual Memory Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuMemAddressFree` | Missing |
| `cuMemAddressReserve` | Missing |
| `cuMemCreate` | Missing |
| `cuMemExportToShareableHandle` | Missing |
| `cuMemGetAccess` | Missing |
| `cuMemGetAllocationGranularity` | Missing |
| `cuMemGetAllocationPropertiesFromHandle` | Missing |
| `cuMemImportFromShareableHandle` | Missing |
| `cuMemMap` | Missing |
| `cuMemMapArrayAsync` | Missing |
| `cuMemRelease` | Missing |
| `cuMemRetainAllocationHandle` | Missing |
| `cuMemSetAccess` | Missing |
| `cuMemUnmap` | Missing |

## Stream Ordered Memory Allocator

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuMemAllocAsync` | `Stream::memAllocAsync(size_t)` |
| `cuMemAllocFromPoolAsync` | Missing |
| `cuMemFreeAsync` | `Stream::memFreeAsync(DeviceMemory &)` |
| `cuMemGetDefaultMemPool` | Missing |
| `cuMemGetMemPool` | Missing |
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
| `cuMemSetMemPool` | Missing |

## Multicast Object Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuMulticastAddDevice` | Missing |
| `cuMulticastBindAddr` | Missing |
| `cuMulticastBindMem` | Missing |
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
| `cuMemAdvise` | Missing |
| `cuMemDiscardAndPrefetchBatchAsync` | Missing |
| `cuMemDiscardBatchAsync` | Missing |
| `cuMemPrefetchAsync` | `Stream::memPrefetchAsync(DeviceMemory &, size_t) / Stream::memPrefetchAsync(DeviceMemory &, size_t, Device &)` |
| `cuMemPrefetchBatchAsync` | Missing |
| `cuMemRangeGetAttribute` | Missing |
| `cuMemRangeGetAttributes` | Missing |
| `cuPointerGetAttribute` | `Implemented` |
| `cuPointerGetAttributes` | `Implemented` |
| `cuPointerSetAttribute` | `pointerSetAttribute()` |

## Stream Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuStreamAddCallback` | `Stream::addCallback(CUstreamCallback, void *, unsigned int)` |
| `cuStreamAttachMemAsync` | Missing |
| `cuStreamBeginCapture` | Missing |
| `cuStreamBeginCaptureToCig` | Missing |
| `cuStreamBeginCaptureToGraph` | Missing |
| `cuStreamBeginRecaptureToGraph` | Missing |
| `cuStreamCopyAttributes` | Missing |
| `cuStreamCreate` | `Stream::Stream(unsigned int)` |
| `cuStreamCreateWithPriority` | Missing |
| `cuStreamDestroy` | `Stream destructor` |
| `cuStreamEndCapture` | Missing |
| `cuStreamEndCaptureToCig` | Missing |
| `cuStreamGetAttribute` | Missing |
| `cuStreamGetCaptureInfo` | Missing |
| `cuStreamGetCtx` | Missing |
| `cuStreamGetDevice` | Missing |
| `cuStreamGetFlags` | Missing |
| `cuStreamGetId` | Missing |
| `cuStreamGetPriority` | Missing |
| `cuStreamIsCapturing` | Missing |
| `cuStreamQuery` | `Stream::query()` |
| `cuStreamSetAttribute` | Missing |
| `cuStreamSynchronize` | `Stream::synchronize()` |
| `cuStreamUpdateCaptureDependencies` | Missing |
| `cuStreamWaitEvent` | `Stream::wait(Event &)` |
| `cuThreadExchangeStreamCaptureMode` | Missing |

## Event Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuEventCreate` | `Event::Event(unsigned int)` |
| `cuEventDestroy` | `Event destructor` |
| `cuEventElapsedTime` | `Event::elapsedTime()` |
| `cuEventQuery` | `Event::query()` |
| `cuEventRecord` | `Event::record() / Stream::record(Event &)` |
| `cuEventRecordWithFlags` | `Event::record(Stream &, unsigned int) / Stream::record(Event &, unsigned int)` |
| `cuEventSynchronize` | `Event::synchronize()` |

## External Resource Interoperability

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuDestroyExternalMemory` | Missing |
| `cuDestroyExternalSemaphore` | Missing |
| `cuExternalMemoryGetMappedBuffer` | Missing |
| `cuExternalMemoryGetMappedMipmappedArray` | Missing |
| `cuImportExternalMemory` | Missing |
| `cuImportExternalSemaphore` | Missing |
| `cuSignalExternalSemaphoresAsync` | Missing |
| `cuWaitExternalSemaphoresAsync` | Missing |

## Stream Memory Operations

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuStreamBatchMemOp` | `Stream::batchMemOp(unsigned count, CUstreamBatchMemOpParams *, unsigned flags)` |
| `cuStreamWaitValue32` | `Stream::waitValue32()` |
| `cuStreamWaitValue64` | Missing |
| `cuStreamWriteValue32` | `Stream::writeValue32()` |
| `cuStreamWriteValue64` | Missing |

## Execution Control

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuFuncGetAttribute` | `Function::getAttribute(CUfunction_attribute/hipFunction_attribute)` |
| `cuFuncGetModule` | Missing |
| `cuFuncGetName` | Missing |
| `cuFuncGetParamCount` | Missing |
| `cuFuncGetParamInfo` | Missing |
| `cuFuncIsLoaded` | Missing |
| `cuFuncLoad` | Missing |
| `cuFuncSetAttribute` | `Function::setAttribute(CUfunction_attribute, int)` |
| `cuFuncSetCacheConfig` | `Function::setCacheConfig(CUfunc_cache)` |
| `cuLaunchCooperativeKernel` | `Stream::launchCooperativeKernel(Function &, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, const std::vector<const void *> &)` |
| `cuLaunchHostFunc` | `Stream::launchHostFunc()` |
| `cuLaunchKernel` | `Stream::launchKernel(Function &, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, const std::vector<const void *> &)` |
| `cuLaunchKernelEx` | Missing |

## Graph Management

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuDeviceGetGraphMemAttribute` | `Device::getGraphMemAttribute()` |
| `cuDeviceGraphMemTrim` | `Device::graphMemTrim()` |
| `cuDeviceSetGraphMemAttribute` | `Device::setGraphMemAttribute()` |
| `cuGraphAddBatchMemOpNode` | Missing |
| `cuGraphAddChildGraphNode` | Missing |
| `cuGraphAddDependencies` | Missing |
| `cuGraphAddEmptyNode` | Missing |
| `cuGraphAddEventRecordNode` | Missing |
| `cuGraphAddEventWaitNode` | Missing |
| `cuGraphAddExternalSemaphoresSignalNode` | Missing |
| `cuGraphAddExternalSemaphoresWaitNode` | Missing |
| `cuGraphAddHostNode` | `Graph::addHostNode()` |
| `cuGraphAddKernelNode` | `Graph::addKernelNode()` |
| `cuGraphAddMemAllocNode` | `Graph::addMemAllocNode()` |
| `cuGraphAddMemFreeNode` | `Graph::addDevMemFreeNode()` |
| `cuGraphAddMemcpyNode` | `Graph::addMemCpyNode()` |
| `cuGraphAddMemsetNode` | Missing |
| `cuGraphAddNode` | Missing |
| `cuGraphBatchMemOpNodeGetParams` | Missing |
| `cuGraphBatchMemOpNodeSetParams` | Missing |
| `cuGraphChildGraphNodeGetGraph` | Missing |
| `cuGraphClone` | Missing |
| `cuGraphConditionalHandleCreate` | Missing |
| `cuGraphCreate` | `Graph::Graph(Context &, unsigned int)` |
| `cuGraphDebugDotPrint` | `Graph::debugDotPrint()` |
| `cuGraphDestroy` | `Graph destructor` |
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
| `cuGraphKernelNodeCopyAttributes` | Missing |
| `cuGraphKernelNodeGetAttribute` | Missing |
| `cuGraphKernelNodeGetParams` | Missing |
| `cuGraphKernelNodeSetAttribute` | Missing |
| `cuGraphKernelNodeSetParams` | Missing |
| `cuGraphLaunch` | `Stream::graphLaunch(GraphExec &)` |
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
| `cuUserObjectCreate` | Missing |
| `cuUserObjectRelease` | Missing |
| `cuUserObjectRetain` | Missing |

## Occupancy

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuOccupancyAvailableDynamicSMemPerBlock` | Missing |
| `cuOccupancyMaxActiveBlocksPerMultiprocessor` | `Function::occupancyMaxActiveBlocksPerMultiprocessor(int, size_t)` |
| `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` | Missing |
| `cuOccupancyMaxActiveClusters` | Missing |
| `cuOccupancyMaxPotentialBlockSize` | Missing |
| `cuOccupancyMaxPotentialBlockSizeWithFlags` | Missing |
| `cuOccupancyMaxPotentialClusterSize` | Missing |

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
| `cuCtxDisablePeerAccess` | `Context::disablePeerAccess(Context &)` |
| `cuCtxEnablePeerAccess` | `Context::enablePeerAccess(Context &, unsigned int)` |
| `cuDeviceCanAccessPeer` | `Device::canAccessPeer(const Device &, const Device &)` |
| `cuDeviceGetP2PAtomicCapabilities` | `Device::getP2PAtomicCapabilities()` |
| `cuDeviceGetP2PAttribute` | `Device::getP2PAttribute()` |

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

| CUDA Driver API | cudawrappers interface |
|---|---|
| `cuGetProcAddress` | Missing |

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
| `cuCtxFromGreenCtx` | `Implemented` |
| `cuCtxGetDevResource` | Missing |
| `cuDevResourceGenerateDesc` | Missing |
| `cuDevSmResourceSplit` | Missing |
| `cuDevSmResourceSplitByCount` | Missing |
| `cuDeviceGetDevResource` | `Device::getDevResource()` |
| `cuGreenCtxCreate` | `GreenContext::GreenContext(CUdevResourceDesc, Device &, unsigned int)` |
| `cuGreenCtxDestroy` | `GreenContext destructor` |
| `cuGreenCtxGetDevResource` | `GreenContext::getDevResource(CUdevResource &, CUdevResourceType)` |
| `cuGreenCtxGetId` | `GreenContext::getId()` |
| `cuGreenCtxRecordEvent` | `GreenContext::recordEvent(Event &)` |
| `cuGreenCtxStreamCreate` | `GreenContext::createStream(unsigned int, int)` |
| `cuGreenCtxWaitEvent` | `GreenContext::waitEvent(Event &)` |
| `cuStreamGetDevResource` | Missing |
| `cuStreamGetGreenCtx` | `Stream::getGreenCtx()` |

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
