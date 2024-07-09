#if !defined MACROS_H
#ifdef __HIP_PLATFORM_AMD__
#define CUDA_ARRAY3D_CUBEMAP hipArrayCubemap
#define CUDA_ARRAY3D_DESCRIPTOR HIP_ARRAY3D_DESCRIPTOR
#define CUDA_ARRAY3D_DESCRIPTOR_st HIP_ARRAY3D_DESCRIPTOR
#define CUDA_ARRAY3D_LAYERED hipArrayLayered
#define CUDA_ARRAY3D_SURFACE_LDST hipArraySurfaceLoadStore
#define CUDA_ARRAY_DESCRIPTOR HIP_ARRAY_DESCRIPTOR
#define CUDA_ARRAY_DESCRIPTOR_st HIP_ARRAY_DESCRIPTOR
#define CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC   hipCooperativeLaunchMultiDeviceNoPostSync
#define CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC   hipCooperativeLaunchMultiDeviceNoPreSync
#define CUDA_C_16BF HIP_C_16BF
#define CUDA_C_16F HIP_C_16F
#define CUDA_C_32F HIP_C_32F
#define CUDA_C_32I HIP_C_32I
#define CUDA_C_32U HIP_C_32U
#define CUDA_C_64F HIP_C_64F
#define CUDA_C_8I HIP_C_8I
#define CUDA_C_8U HIP_C_8U
#define CUDA_ERROR_ALREADY_ACQUIRED hipErrorAlreadyAcquired
#define CUDA_ERROR_ALREADY_MAPPED hipErrorAlreadyMapped
#define CUDA_ERROR_ARRAY_IS_MAPPED hipErrorArrayIsMapped
#define CUDA_ERROR_ASSERT hipErrorAssert
#define CUDA_ERROR_CAPTURED_EVENT hipErrorCapturedEvent
#define CUDA_ERROR_CONTEXT_ALREADY_CURRENT hipErrorContextAlreadyCurrent
#define CUDA_ERROR_CONTEXT_ALREADY_IN_USE hipErrorContextAlreadyInUse
#define CUDA_ERROR_CONTEXT_IS_DESTROYED hipErrorContextIsDestroyed
#define CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE   hipErrorCooperativeLaunchTooLarge
#define CUDA_ERROR_DEINITIALIZED hipErrorDeinitialized
#define CUDA_ERROR_ECC_UNCORRECTABLE hipErrorECCNotCorrectable
#define CUDA_ERROR_FILE_NOT_FOUND hipErrorFileNotFound
#define CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE hipErrorGraphExecUpdateFailure
#define CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED   hipErrorHostMemoryAlreadyRegistered
#define CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED hipErrorHostMemoryNotRegistered
#define CUDA_ERROR_ILLEGAL_ADDRESS hipErrorIllegalAddress
#define CUDA_ERROR_ILLEGAL_STATE hipErrorIllegalState
#define CUDA_ERROR_INVALID_CONTEXT hipErrorInvalidContext
#define CUDA_ERROR_INVALID_DEVICE hipErrorInvalidDevice
#define CUDA_ERROR_INVALID_GRAPHICS_CONTEXT hipErrorInvalidGraphicsContext
#define CUDA_ERROR_INVALID_HANDLE hipErrorInvalidHandle
#define CUDA_ERROR_INVALID_IMAGE hipErrorInvalidImage
#define CUDA_ERROR_INVALID_PTX hipErrorInvalidKernelFile
#define CUDA_ERROR_INVALID_SOURCE hipErrorInvalidSource
#define CUDA_ERROR_INVALID_VALUE hipErrorInvalidValue
#define CUDA_ERROR_LAUNCH_FAILED hipErrorLaunchFailure
#define CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES hipErrorLaunchOutOfResources
#define CUDA_ERROR_LAUNCH_TIMEOUT hipErrorLaunchTimeOut
#define CUDA_ERROR_MAP_FAILED hipErrorMapFailed
#define CUDA_ERROR_NOT_FOUND hipErrorNotFound
#define CUDA_ERROR_NOT_INITIALIZED hipErrorNotInitialized
#define CUDA_ERROR_NOT_MAPPED hipErrorNotMapped
#define CUDA_ERROR_NOT_MAPPED_AS_ARRAY hipErrorNotMappedAsArray
#define CUDA_ERROR_NOT_MAPPED_AS_POINTER hipErrorNotMappedAsPointer
#define CUDA_ERROR_NOT_READY hipErrorNotReady
#define CUDA_ERROR_NOT_SUPPORTED hipErrorNotSupported
#define CUDA_ERROR_NO_BINARY_FOR_GPU hipErrorNoBinaryForGpu
#define CUDA_ERROR_NO_DEVICE hipErrorNoDevice
#define CUDA_ERROR_OPERATING_SYSTEM hipErrorOperatingSystem
#define CUDA_ERROR_OUT_OF_MEMORY hipErrorOutOfMemory
#define CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED hipErrorPeerAccessAlreadyEnabled
#define CUDA_ERROR_PEER_ACCESS_NOT_ENABLED hipErrorPeerAccessNotEnabled
#define CUDA_ERROR_PEER_ACCESS_UNSUPPORTED hipErrorPeerAccessUnsupported
#define CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE hipErrorSetOnActiveProcess
#define CUDA_ERROR_PROFILER_ALREADY_STARTED hipErrorProfilerAlreadyStarted
#define CUDA_ERROR_PROFILER_ALREADY_STOPPED hipErrorProfilerAlreadyStopped
#define CUDA_ERROR_PROFILER_DISABLED hipErrorProfilerDisabled
#define CUDA_ERROR_PROFILER_NOT_INITIALIZED hipErrorProfilerNotInitialized
#define CUDA_ERROR_SHARED_OBJECT_INIT_FAILED hipErrorSharedObjectInitFailed
#define CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND   hipErrorSharedObjectSymbolNotFound
#define CUDA_ERROR_STREAM_CAPTURE_IMPLICIT hipErrorStreamCaptureImplicit
#define CUDA_ERROR_STREAM_CAPTURE_INVALIDATED hipErrorStreamCaptureInvalidated
#define CUDA_ERROR_STREAM_CAPTURE_ISOLATION hipErrorStreamCaptureIsolation
#define CUDA_ERROR_STREAM_CAPTURE_MERGE hipErrorStreamCaptureMerge
#define CUDA_ERROR_STREAM_CAPTURE_UNJOINED hipErrorStreamCaptureUnjoined
#define CUDA_ERROR_STREAM_CAPTURE_UNMATCHED hipErrorStreamCaptureUnmatched
#define CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED hipErrorStreamCaptureUnsupported
#define CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD hipErrorStreamCaptureWrongThread
#define CUDA_ERROR_UNKNOWN hipErrorUnknown
#define CUDA_ERROR_UNMAP_FAILED hipErrorUnmapFailed
#define CUDA_ERROR_UNSUPPORTED_LIMIT hipErrorUnsupportedLimit
#define CUDA_EXTERNAL_MEMORY_DEDICATED hipExternalMemoryDedicated
#define CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH   hipGraphInstantiateFlagAutoFreeOnLaunch
#define CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH   hipGraphInstantiateFlagDeviceLaunch
#define CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD hipGraphInstantiateFlagUpload
#define CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY   hipGraphInstantiateFlagUseNodePriority
#define CUDA_IPC_HANDLE_SIZE HIP_IPC_HANDLE_SIZE
#define CUDA_R_16BF HIP_R_16BF
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_32F HIP_R_32F
#define CUDA_R_32I HIP_R_32I
#define CUDA_R_32U HIP_R_32U
#define CUDA_R_64F HIP_R_64F
#define CUDA_R_8I HIP_R_8I
#define CUDA_R_8U HIP_R_8U
#define CUDA_SUCCESS hipSuccess
#define CUFFT_ALLOC_FAILED HIPFFT_ALLOC_FAILED
#define CUFFT_C2C HIPFFT_C2C
#define CUFFT_C2R HIPFFT_C2R
#define CUFFT_CB_LD_COMPLEX HIPFFT_CB_LD_COMPLEX
#define CUFFT_CB_LD_COMPLEX_DOUBLE HIPFFT_CB_LD_COMPLEX_DOUBLE
#define CUFFT_CB_LD_REAL HIPFFT_CB_LD_REAL
#define CUFFT_CB_LD_REAL_DOUBLE HIPFFT_CB_LD_REAL_DOUBLE
#define CUFFT_CB_ST_COMPLEX HIPFFT_CB_ST_COMPLEX
#define CUFFT_CB_ST_COMPLEX_DOUBLE HIPFFT_CB_ST_COMPLEX_DOUBLE
#define CUFFT_CB_ST_REAL HIPFFT_CB_ST_REAL
#define CUFFT_CB_ST_REAL_DOUBLE HIPFFT_CB_ST_REAL_DOUBLE
#define CUFFT_CB_UNDEFINED HIPFFT_CB_UNDEFINED
#define CUFFT_D2Z HIPFFT_D2Z
#define CUFFT_EXEC_FAILED HIPFFT_EXEC_FAILED
#define CUFFT_FORWARD HIPFFT_FORWARD
#define CUFFT_INCOMPLETE_PARAMETER_LIST HIPFFT_INCOMPLETE_PARAMETER_LIST
#define CUFFT_INTERNAL_ERROR HIPFFT_INTERNAL_ERROR
#define CUFFT_INVALID_DEVICE HIPFFT_INVALID_DEVICE
#define CUFFT_INVALID_PLAN HIPFFT_INVALID_PLAN
#define CUFFT_INVALID_SIZE HIPFFT_INVALID_SIZE
#define CUFFT_INVALID_TYPE HIPFFT_INVALID_TYPE
#define CUFFT_INVALID_VALUE HIPFFT_INVALID_VALUE
#define CUFFT_INVERSE HIPFFT_BACKWARD
#define CUFFT_NOT_IMPLEMENTED HIPFFT_NOT_IMPLEMENTED
#define CUFFT_NOT_SUPPORTED HIPFFT_NOT_SUPPORTED
#define CUFFT_NO_WORKSPACE HIPFFT_NO_WORKSPACE
#define CUFFT_PARSE_ERROR HIPFFT_PARSE_ERROR
#define CUFFT_R2C HIPFFT_R2C
#define CUFFT_SETUP_FAILED HIPFFT_SETUP_FAILED
#define CUFFT_SUCCESS HIPFFT_SUCCESS
#define CUFFT_UNALIGNED_DATA HIPFFT_UNALIGNED_DATA
#define CUFFT_Z2D HIPFFT_Z2D
#define CUFFT_Z2Z HIPFFT_Z2Z
#define CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL   hipArraySparseSubresourceTypeMiptail
#define CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL   hipArraySparseSubresourceTypeSparseLevel
#define CU_COMPUTEMODE_DEFAULT hipComputeModeDefault
#define CU_COMPUTEMODE_EXCLUSIVE hipComputeModeExclusive
#define CU_COMPUTEMODE_EXCLUSIVE_PROCESS hipComputeModeExclusiveProcess
#define CU_COMPUTEMODE_PROHIBITED hipComputeModeProhibited
#define CU_CTX_BLOCKING_SYNC hipDeviceScheduleBlockingSync
#define CU_CTX_LMEM_RESIZE_TO_MAX hipDeviceLmemResizeToMax
#define CU_CTX_MAP_HOST hipDeviceMapHost
#define CU_CTX_SCHED_AUTO hipDeviceScheduleAuto
#define CU_CTX_SCHED_BLOCKING_SYNC hipDeviceScheduleBlockingSync
#define CU_CTX_SCHED_MASK hipDeviceScheduleMask
#define CU_CTX_SCHED_SPIN hipDeviceScheduleSpin
#define CU_CTX_SCHED_YIELD hipDeviceScheduleYield
#define CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT   hipDeviceAttributeAsyncEngineCount
#define CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY   hipDeviceAttributeCanMapHostMemory
#define CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM   hipDeviceAttributeCanUseHostPointerForRegisteredMem
#define CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR   hipDeviceAttributeCanUseStreamWaitValue
#define CU_DEVICE_ATTRIBUTE_CLOCK_RATE hipDeviceAttributeClockRate
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR   hipDeviceAttributeComputeCapabilityMajor
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR   hipDeviceAttributeComputeCapabilityMinor
#define CU_DEVICE_ATTRIBUTE_COMPUTE_MODE hipDeviceAttributeComputeMode
#define CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED   hipDeviceAttributeComputePreemptionSupported
#define CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS   hipDeviceAttributeConcurrentKernels
#define CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS   hipDeviceAttributeConcurrentManagedAccess
#define CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH   hipDeviceAttributeCooperativeLaunch
#define CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH   hipDeviceAttributeCooperativeMultiDeviceLaunch
#define CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST   hipDeviceAttributeDirectManagedMemAccessFromHost
#define CU_DEVICE_ATTRIBUTE_ECC_ENABLED hipDeviceAttributeEccEnabled
#define CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED   hipDeviceAttributeGlobalL1CacheSupported
#define CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH   hipDeviceAttributeMemoryBusWidth
#define CU_DEVICE_ATTRIBUTE_GPU_OVERLAP hipDeviceAttributeAsyncEngineCount
#define CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED   hipDeviceAttributeHostNativeAtomicSupported
#define CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED   hipDeviceAttributeHostRegisterSupported
#define CU_DEVICE_ATTRIBUTE_INTEGRATED hipDeviceAttributeIntegrated
#define CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT   hipDeviceAttributeKernelExecTimeout
#define CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE hipDeviceAttributeL2CacheSize
#define CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED   hipDeviceAttributeLocalL1CacheSupported
#define CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY hipDeviceAttributeManagedMemory
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR   hipDeviceAttributeMaxBlocksPerMultiprocessor
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X hipDeviceAttributeMaxBlockDimX
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y hipDeviceAttributeMaxBlockDimY
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z hipDeviceAttributeMaxBlockDimZ
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X hipDeviceAttributeMaxGridDimX
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y hipDeviceAttributeMaxGridDimY
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z hipDeviceAttributeMaxGridDimZ
#define CU_DEVICE_ATTRIBUTE_MAX_PITCH hipDeviceAttributeMaxPitch
#define CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK   hipDeviceAttributeMaxRegistersPerBlock
#define CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR   hipDeviceAttributeMaxRegistersPerMultiprocessor
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK   hipDeviceAttributeMaxSharedMemoryPerBlock
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN   hipDeviceAttributeSharedMemPerBlockOptin
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR   hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK   hipDeviceAttributeMaxThreadsPerBlock
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR   hipDeviceAttributeMaxThreadsPerMultiProcessor
#define CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE hipDeviceAttributeMemoryClockRate
#define CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED   hipDeviceAttributeMemoryPoolsSupported
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT   hipDeviceAttributeMultiprocessorCount
#define CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD hipDeviceAttributeIsMultiGpuBoard
#define CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID   hipDeviceAttributeMultiGpuBoardGroupId
#define CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS   hipDeviceAttributePageableMemoryAccess
#define CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES   hipDeviceAttributePageableMemoryAccessUsesHostPageTables
#define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID hipDeviceAttributePciBusId
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID hipDeviceAttributePciDeviceId
#define CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID hipDeviceAttributePciDomainID
#define CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK   hipDeviceAttributeMaxRegistersPerBlock
#define CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK   hipDeviceAttributeMaxSharedMemoryPerBlock
#define CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO   hipDeviceAttributeSingleToDoublePrecisionPerfRatio
#define CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED   hipDeviceAttributeStreamPrioritiesSupported
#define CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT hipDeviceAttributeSurfaceAlignment
#define CU_DEVICE_ATTRIBUTE_TCC_DRIVER hipDeviceAttributeTccDriver
#define CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY   hipDeviceAttributeTotalConstantMemory
#define CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING   hipDeviceAttributeUnifiedAddressing
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED   hipDeviceAttributeVirtualMemoryManagementSupported
#define CU_DEVICE_ATTRIBUTE_WARP_SIZE hipDeviceAttributeWarpSize
#define CU_DEVICE_CPU hipCpuDeviceId
#define CU_DEVICE_INVALID hipInvalidDeviceId
#define CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED   hipDevP2PAttrHipArrayAccessSupported
#define CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED hipDevP2PAttrAccessSupported
#define CU_DEVICE_P2P_ATTRIBUTE_ARRAY_ACCESS_ACCESS_SUPPORTED   hipDevP2PAttrHipArrayAccessSupported
#define CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED   hipDevP2PAttrHipArrayAccessSupported
#define CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED   hipDevP2PAttrNativeAtomicSupported
#define CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK hipDevP2PAttrPerformanceRank
#define CU_EVENT_BLOCKING_SYNC hipEventBlockingSync
#define CU_EVENT_DEFAULT hipEventDefault
#define CU_EVENT_DISABLE_TIMING hipEventDisableTiming
#define CU_EVENT_INTERPROCESS hipEventInterprocess
#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE   hipExternalMemoryHandleTypeD3D11Resource
#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT   hipExternalMemoryHandleTypeD3D11ResourceKmt
#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP   hipExternalMemoryHandleTypeD3D12Heap
#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE   hipExternalMemoryHandleTypeD3D12Resource
#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD   hipExternalMemoryHandleTypeOpaqueFd
#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32   hipExternalMemoryHandleTypeOpaqueWin32
#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT   hipExternalMemoryHandleTypeOpaqueWin32Kmt
#define CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE   hipExternalSemaphoreHandleTypeD3D12Fence
#define CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD   hipExternalSemaphoreHandleTypeOpaqueFd
#define CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32   hipExternalSemaphoreHandleTypeOpaqueWin32
#define CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT   hipExternalSemaphoreHandleTypeOpaqueWin32Kmt
#define CU_FUNC_ATTRIBUTE_BINARY_VERSION HIP_FUNC_ATTRIBUTE_BINARY_VERSION
#define CU_FUNC_ATTRIBUTE_CACHE_MODE_CA HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA
#define CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
#define CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
#define CU_FUNC_ATTRIBUTE_MAX HIP_FUNC_ATTRIBUTE_MAX
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES   HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK   HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define CU_FUNC_ATTRIBUTE_NUM_REGS HIP_FUNC_ATTRIBUTE_NUM_REGS
#define CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT   HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
#define CU_FUNC_ATTRIBUTE_PTX_VERSION HIP_FUNC_ATTRIBUTE_PTX_VERSION
#define CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#define CU_FUNC_CACHE_PREFER_EQUAL hipFuncCachePreferEqual
#define CU_FUNC_CACHE_PREFER_L1 hipFuncCachePreferL1
#define CU_FUNC_CACHE_PREFER_NONE hipFuncCachePreferNone
#define CU_FUNC_CACHE_PREFER_SHARED hipFuncCachePreferShared
#define CU_IPC_HANDLE_SIZE HIP_IPC_HANDLE_SIZE
#define CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS hipIpcMemLazyEnablePeerAccess
#define CU_JIT_CACHE_MODE HIPRTC_JIT_CACHE_MODE
#define CU_JIT_ERROR_LOG_BUFFER HIPRTC_JIT_ERROR_LOG_BUFFER
#define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES   HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
#define CU_JIT_FALLBACK_STRATEGY HIPRTC_JIT_FALLBACK_STRATEGY
#define CU_JIT_FAST_COMPILE HIPRTC_JIT_FAST_COMPILE
#define CU_JIT_GENERATE_DEBUG_INFO HIPRTC_JIT_GENERATE_DEBUG_INFO
#define CU_JIT_GENERATE_LINE_INFO HIPRTC_JIT_GENERATE_LINE_INFO
#define CU_JIT_INFO_LOG_BUFFER HIPRTC_JIT_INFO_LOG_BUFFER
#define CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES
#define CU_JIT_INPUT_CUBIN HIPRTC_JIT_INPUT_CUBIN
#define CU_JIT_INPUT_FATBINARY HIPRTC_JIT_INPUT_FATBINARY
#define CU_JIT_INPUT_LIBRARY HIPRTC_JIT_INPUT_LIBRARY
#define CU_JIT_INPUT_NVVM HIPRTC_JIT_INPUT_NVVM
#define CU_JIT_INPUT_OBJECT HIPRTC_JIT_INPUT_OBJECT
#define CU_JIT_INPUT_PTX HIPRTC_JIT_INPUT_PTX
#define CU_JIT_LOG_VERBOSE HIPRTC_JIT_LOG_VERBOSE
#define CU_JIT_MAX_REGISTERS HIPRTC_JIT_MAX_REGISTERS
#define CU_JIT_NEW_SM3X_OPT HIPRTC_JIT_NEW_SM3X_OPT
#define CU_JIT_NUM_INPUT_TYPES HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES
#define CU_JIT_NUM_OPTIONS HIPRTC_JIT_NUM_OPTIONS
#define CU_JIT_OPTIMIZATION_LEVEL HIPRTC_JIT_OPTIMIZATION_LEVEL
#define CU_JIT_TARGET HIPRTC_JIT_TARGET
#define CU_JIT_TARGET_FROM_CUCONTEXT HIPRTC_JIT_TARGET_FROM_HIPCONTEXT
#define CU_JIT_THREADS_PER_BLOCK HIPRTC_JIT_THREADS_PER_BLOCK
#define CU_JIT_WALL_TIME HIPRTC_JIT_WALL_TIME
#define CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW   hipKernelNodeAttributeAccessPolicyWindow
#define CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE hipKernelNodeAttributeCooperative
#define CU_LAUNCH_PARAM_BUFFER_POINTER HIP_LAUNCH_PARAM_BUFFER_POINTER
#define CU_LAUNCH_PARAM_BUFFER_SIZE HIP_LAUNCH_PARAM_BUFFER_SIZE
#define CU_LAUNCH_PARAM_END HIP_LAUNCH_PARAM_END
#define CU_LIMIT_MALLOC_HEAP_SIZE hipLimitMallocHeapSize
#define CU_LIMIT_PRINTF_FIFO_SIZE hipLimitPrintfFifoSize
#define CU_LIMIT_STACK_SIZE hipLimitStackSize
#define CU_MEMHOSTALLOC_DEVICEMAP hipHostMallocMapped
#define CU_MEMHOSTALLOC_PORTABLE hipHostMallocPortable
#define CU_MEMHOSTALLOC_WRITECOMBINED hipHostMallocWriteCombined
#define CU_MEMHOSTREGISTER_DEVICEMAP hipHostRegisterMapped
#define CU_MEMHOSTREGISTER_IOMEMORY hipHostRegisterIoMemory
#define CU_MEMHOSTREGISTER_PORTABLE hipHostRegisterPortable
#define CU_MEMHOSTREGISTER_READ_ONLY hipHostRegisterReadOnly
#define CU_MEMORYTYPE_ARRAY hipMemoryTypeArray
#define CU_MEMORYTYPE_DEVICE hipMemoryTypeDevice
#define CU_MEMORYTYPE_HOST hipMemoryTypeHost
#define CU_MEMORYTYPE_UNIFIED hipMemoryTypeUnified
#define CU_MEMPOOL_ATTR_RELEASE_THRESHOLD hipMemPoolAttrReleaseThreshold
#define CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT hipMemPoolAttrReservedMemCurrent
#define CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH hipMemPoolAttrReservedMemHigh
#define CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES   hipMemPoolReuseAllowInternalDependencies
#define CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC   hipMemPoolReuseAllowOpportunistic
#define CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES   hipMemPoolReuseFollowEventDependencies
#define CU_MEMPOOL_ATTR_USED_MEM_CURRENT hipMemPoolAttrUsedMemCurrent
#define CU_MEMPOOL_ATTR_USED_MEM_HIGH hipMemPoolAttrUsedMemHigh
#define CU_MEM_ACCESS_FLAGS_PROT_NONE hipMemAccessFlagsProtNone
#define CU_MEM_ACCESS_FLAGS_PROT_READ hipMemAccessFlagsProtRead
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#define CU_MEM_ADVISE_SET_ACCESSED_BY hipMemAdviseSetAccessedBy
#define CU_MEM_ADVISE_SET_PREFERRED_LOCATION hipMemAdviseSetPreferredLocation
#define CU_MEM_ADVISE_SET_READ_MOSTLY hipMemAdviseSetReadMostly
#define CU_MEM_ADVISE_UNSET_ACCESSED_BY hipMemAdviseUnsetAccessedBy
#define CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION   hipMemAdviseUnsetPreferredLocation
#define CU_MEM_ADVISE_UNSET_READ_MOSTLY hipMemAdviseUnsetReadMostly
#define CU_MEM_ALLOCATION_TYPE_INVALID hipMemAllocationTypeInvalid
#define CU_MEM_ALLOCATION_TYPE_MAX hipMemAllocationTypeMax
#define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
#define CU_MEM_ALLOC_GRANULARITY_MINIMUM hipMemAllocationGranularityMinimum
#define CU_MEM_ALLOC_GRANULARITY_RECOMMENDED   hipMemAllocationGranularityRecommended
#define CU_MEM_ATTACH_GLOBAL hipMemAttachGlobal
#define CU_MEM_ATTACH_HOST hipMemAttachHost
#define CU_MEM_ATTACH_SINGLE hipMemAttachSingle
#define CU_MEM_HANDLE_TYPE_GENERIC hipMemHandleTypeGeneric
#define CU_MEM_HANDLE_TYPE_NONE hipMemHandleTypeNone
#define CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR   hipMemHandleTypePosixFileDescriptor
#define CU_MEM_HANDLE_TYPE_WIN32 hipMemHandleTypeWin32
#define CU_MEM_HANDLE_TYPE_WIN32_KMT hipMemHandleTypeWin32Kmt
#define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#define CU_MEM_LOCATION_TYPE_INVALID hipMemLocationTypeInvalid
#define CU_MEM_OPERATION_TYPE_MAP hipMemOperationTypeMap
#define CU_MEM_OPERATION_TYPE_UNMAP hipMemOperationTypeUnmap
#define CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY hipMemRangeAttributeAccessedBy
#define CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION   hipMemRangeAttributeLastPrefetchLocation
#define CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION   hipMemRangeAttributePreferredLocation
#define CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY hipMemRangeAttributeReadMostly
#define CU_OCCUPANCY_DEFAULT hipOccupancyDefault
#define CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE hipOccupancyDisableCachingOverride
#define CU_POINTER_ATTRIBUTE_ACCESS_FLAGS HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS
#define CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES   HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
#define CU_POINTER_ATTRIBUTE_BUFFER_ID HIP_POINTER_ATTRIBUTE_BUFFER_ID
#define CU_POINTER_ATTRIBUTE_CONTEXT HIP_POINTER_ATTRIBUTE_CONTEXT
#define CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
#define CU_POINTER_ATTRIBUTE_DEVICE_POINTER HIP_POINTER_ATTRIBUTE_DEVICE_POINTER
#define CU_POINTER_ATTRIBUTE_HOST_POINTER HIP_POINTER_ATTRIBUTE_HOST_POINTER
#define CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE   HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
#define CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE   HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE
#define CU_POINTER_ATTRIBUTE_IS_MANAGED HIP_POINTER_ATTRIBUTE_IS_MANAGED
#define CU_POINTER_ATTRIBUTE_MAPPED HIP_POINTER_ATTRIBUTE_MAPPED
#define CU_POINTER_ATTRIBUTE_MEMORY_TYPE HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
#define CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
#define CU_POINTER_ATTRIBUTE_P2P_TOKENS HIP_POINTER_ATTRIBUTE_P2P_TOKENS
#define CU_POINTER_ATTRIBUTE_RANGE_SIZE HIP_POINTER_ATTRIBUTE_RANGE_SIZE
#define CU_POINTER_ATTRIBUTE_RANGE_START_ADDR   HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
#define CU_POINTER_ATTRIBUTE_SYNC_MEMOPS HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS
#define CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE hipSharedMemBankSizeDefault
#define CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE hipSharedMemBankSizeEightByte
#define CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE hipSharedMemBankSizeFourByte
#define CU_STREAM_ADD_CAPTURE_DEPENDENCIES hipStreamAddCaptureDependencies
#define CU_STREAM_CAPTURE_MODE_GLOBAL hipStreamCaptureModeGlobal
#define CU_STREAM_CAPTURE_MODE_RELAXED hipStreamCaptureModeRelaxed
#define CU_STREAM_CAPTURE_MODE_THREAD_LOCAL hipStreamCaptureModeThreadLocal
#define CU_STREAM_CAPTURE_STATUS_ACTIVE hipStreamCaptureStatusActive
#define CU_STREAM_CAPTURE_STATUS_INVALIDATED hipStreamCaptureStatusInvalidated
#define CU_STREAM_CAPTURE_STATUS_NONE hipStreamCaptureStatusNone
#define CU_STREAM_DEFAULT hipStreamDefault
#define CU_STREAM_NON_BLOCKING hipStreamNonBlocking
#define CU_STREAM_PER_THREAD hipStreamPerThread
#define CU_STREAM_SET_CAPTURE_DEPENDENCIES hipStreamSetCaptureDependencies
#define CU_STREAM_WAIT_VALUE_AND hipStreamWaitValueAnd
#define CU_STREAM_WAIT_VALUE_EQ hipStreamWaitValueEq
#define CU_STREAM_WAIT_VALUE_GEQ hipStreamWaitValueGte
#define CU_STREAM_WAIT_VALUE_NOR hipStreamWaitValueNor
#define CUaccessPolicyWindow hipAccessPolicyWindow
#define CUaccessPolicyWindow_st hipAccessPolicyWindow
#define CUaccessProperty hipAccessProperty
#define CUaccessProperty_enum hipAccessProperty
#define CUaddress_mode HIPaddress_mode
#define CUaddress_mode_enum HIPaddress_mode_enum
#define CUarray hipArray_t
#define CUarrayMapInfo hipArrayMapInfo
#define CUarrayMapInfo_st hipArrayMapInfo
#define CUarraySparseSubresourceType hipArraySparseSubresourceType
#define CUarraySparseSubresourceType_enum hipArraySparseSubresourceType
#define CUarray_format hipArray_Format
#define CUarray_format_enum hipArray_Format
#define CUarray_st hipArray
#define CUcomputemode hipComputeMode
#define CUcomputemode_enum hipComputeMode
#define CUcontext hipCtx_t
#define CUctx_st ihipCtx_t
#define CUdevice hipDevice_t
#define CUdevice_P2PAttribute hipDeviceP2PAttr
#define CUdevice_P2PAttribute_enum hipDeviceP2PAttr
#define CUdevice_attribute hipDeviceAttribute_t
#define CUdevice_attribute_enum hipDeviceAttribute_t
#define CUdeviceptr hipDeviceptr_t
#define CUevent hipEvent_t
#define CUevent_st ihipEvent_t
#define CUexternalMemory hipExternalMemory_t
#define CUexternalMemoryHandleType hipExternalMemoryHandleType
#define CUexternalMemoryHandleType_enum hipExternalMemoryHandleType_enum
#define CUexternalSemaphore hipExternalSemaphore_t
#define CUexternalSemaphoreHandleType hipExternalSemaphoreHandleType
#define CUexternalSemaphoreHandleType_enum hipExternalSemaphoreHandleType_enum
#define CUfilter_mode HIPfilter_mode
#define CUfilter_mode_enum HIPfilter_mode_enum
#define CUfunc_cache hipFuncCache_t
#define CUfunc_cache_enum hipFuncCache_t
#define CUfunc_st hipModuleSymbol_t
#define CUfunction hipFunction_t
#define CUfunction_attribute hipFuncAttribute
#define CUfunction_attribute_enum hipFuncAttribute
#define CUhostFn hipHostFn_t
#define CUipcEventHandle hipIpcEventHandle_t
#define CUipcEventHandle_st hipIpcEventHandle_st
#define CUipcMemHandle hipIpcMemHandle_t
#define CUipcMemHandle_st hipIpcMemHandle_st
#define CUjitInputType hiprtcJITInputType
#define CUjitInputType_enum hiprtcJITInputType
#define CUjit_option hipJitOption
#define CUjit_option_enum hipJitOption
#define CUkernelNodeAttrID hipKernelNodeAttrID
#define CUkernelNodeAttrID_enum hipKernelNodeAttrID
#define CUkernelNodeAttrValue hipKernelNodeAttrValue
#define CUkernelNodeAttrValue_union hipKernelNodeAttrValue
#define CUlimit hipLimit_t
#define CUlimit_enum hipLimit_t
#define CUlinkState hiprtcLinkState
#define CUlinkState_st ihiprtcLinkState
#define CUmemAccessDesc hipMemAccessDesc
#define CUmemAccessDesc_st hipMemAccessDesc
#define CUmemAccess_flags hipMemAccessFlags
#define CUmemAccess_flags_enum hipMemAccessFlags
#define CUmemAllocationGranularity_flags hipMemAllocationGranularity_flags
#define CUmemAllocationGranularity_flags_enum hipMemAllocationGranularity_flags
#define CUmemAllocationHandleType hipMemAllocationHandleType
#define CUmemAllocationHandleType_enum hipMemAllocationHandleType
#define CUmemAllocationProp hipMemAllocationProp
#define CUmemAllocationProp_st hipMemAllocationProp
#define CUmemAllocationType hipMemAllocationType
#define CUmemAllocationType_enum hipMemAllocationType
#define CUmemGenericAllocationHandle hipMemGenericAllocationHandle_t
#define CUmemHandleType hipMemHandleType
#define CUmemHandleType_enum hipMemHandleType
#define CUmemLocation hipMemLocation
#define CUmemLocationType hipMemLocationType
#define CUmemLocationType_enum hipMemLocationType
#define CUmemLocation_st hipMemLocation
#define CUmemOperationType hipMemOperationType
#define CUmemOperationType_enum hipMemOperationType
#define CUmemPoolHandle_st ihipMemPoolHandle_t
#define CUmemPoolProps hipMemPoolProps
#define CUmemPoolProps_st hipMemPoolProps
#define CUmemPoolPtrExportData hipMemPoolPtrExportData
#define CUmemPoolPtrExportData_st hipMemPoolPtrExportData
#define CUmemPool_attribute hipMemPoolAttr
#define CUmemPool_attribute_enum hipMemPoolAttr
#define CUmem_advise hipMemoryAdvise
#define CUmem_advise_enum hipMemoryAdvise
#define CUmem_range_attribute hipMemRangeAttribute
#define CUmem_range_attribute_enum hipMemRangeAttribute
#define CUmemoryPool hipMemPool_t
#define CUmemorytype hipMemoryType
#define CUmemorytype_enum hipMemoryType
#define CUmipmappedArray hipMipmappedArray_t
#define CUmipmappedArray_st hipMipmappedArray
#define CUmod_st ihipModule_t
#define CUmodule hipModule_t
#define CUoccupancyB2DSize void*
#define CUpointer_attribute hipPointer_attribute
#define CUpointer_attribute_enum hipPointer_attribute
#define CUresourceViewFormat HIPresourceViewFormat
#define CUresourceViewFormat_enum HIPresourceViewFormat_enum
#define CUresourcetype HIPresourcetype
#define CUresourcetype_enum HIPresourcetype_enum
#define CUresult hipError_t
#define CUsharedconfig hipSharedMemConfig
#define CUsharedconfig_enum hipSharedMemConfig
#define CUstream hipStream_t
#define CUstreamCallback hipStreamCallback_t
#define CUstreamCaptureMode hipStreamCaptureMode
#define CUstreamCaptureMode_enum hipStreamCaptureMode
#define CUstreamCaptureStatus hipStreamCaptureStatus
#define CUstreamCaptureStatus_enum hipStreamCaptureStatus
#define CUstreamUpdateCaptureDependencies_flags   hipStreamUpdateCaptureDependenciesFlags
#define CUstreamUpdateCaptureDependencies_flags_enum   hipStreamUpdateCaptureDependenciesFlags
#define CUstream_st ihipStream_t
#define CUsurfObject hipSurfaceObject_t
#define CUtexObject hipTextureObject_t
#define CUuserObject hipUserObject_t
#define CUuserObjectRetain_flags hipUserObjectRetainFlags
#define CUuserObjectRetain_flags_enum hipUserObjectRetainFlags
#define CUuserObject_flags hipUserObjectFlags
#define CUuserObject_flags_enum hipUserObjectFlags
#define CUuserObject_st hipUserObject
#define CUuuid hipUUID
#define CUuuid_st hipUUID_t
#define MACROS_H
#define NVRTC_ERROR_BUILTIN_OPERATION_FAILURE   HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE
#define NVRTC_ERROR_COMPILATION HIPRTC_ERROR_COMPILATION
#define NVRTC_ERROR_INTERNAL_ERROR HIPRTC_ERROR_INTERNAL_ERROR
#define NVRTC_ERROR_INVALID_INPUT HIPRTC_ERROR_INVALID_INPUT
#define NVRTC_ERROR_INVALID_OPTION HIPRTC_ERROR_INVALID_OPTION
#define NVRTC_ERROR_INVALID_PROGRAM HIPRTC_ERROR_INVALID_PROGRAM
#define NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID   HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
#define NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION   HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
#define NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION   HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
#define NVRTC_ERROR_OUT_OF_MEMORY HIPRTC_ERROR_OUT_OF_MEMORY
#define NVRTC_ERROR_PROGRAM_CREATION_FAILURE   HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
#define NVRTC_SUCCESS HIPRTC_SUCCESS
#define cuArray3DCreate hipArray3DCreate
#define cuArray3DGetDescriptor hipArray3DGetDescriptor
#define cuArrayCreate hipArrayCreate
#define cuArrayDestroy hipArrayDestroy
#define cuArrayGetDescriptor hipArrayGetDescriptor
#define cuCtxCreate hipCtxCreate
#define cuCtxDestroy hipCtxDestroy
#define cuCtxGetApiVersion hipCtxGetApiVersion
#define cuCtxGetCacheConfig hipCtxGetCacheConfig
#define cuCtxGetCurrent hipCtxGetCurrent
#define cuCtxGetDevice hipCtxGetDevice
#define cuCtxGetFlags hipCtxGetFlags
#define cuCtxGetLimit hipDeviceGetLimit
#define cuCtxGetSharedMemConfig hipCtxGetSharedMemConfig
#define cuCtxGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cuCtxPopCurrent hipCtxPopCurrent
#define cuCtxPushCurrent hipCtxPushCurrent
#define cuCtxSetCacheConfig hipCtxSetCacheConfig
#define cuCtxSetCurrent hipCtxSetCurrent
#define cuCtxSetLimit hipDeviceSetLimit
#define cuCtxSetSharedMemConfig hipCtxSetSharedMemConfig
#define cuCtxSynchronize hipCtxSynchronize
#define cuDeviceComputeCapability hipDeviceComputeCapability
#define cuDeviceGet hipDeviceGet
#define cuDeviceGetAttribute hipDeviceGetAttribute
#define cuDeviceGetByPCIBusId hipDeviceGetByPCIBusId
#define cuDeviceGetCount hipGetDeviceCount
#define cuDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
#define cuDeviceGetMemPool hipDeviceGetMemPool
#define cuDeviceGetName hipDeviceGetName
#define cuDeviceGetPCIBusId hipDeviceGetPCIBusId
#define cuDeviceGetUuid hipDeviceGetUuid
#define cuDevicePrimaryCtxGetState hipDevicePrimaryCtxGetState
#define cuDevicePrimaryCtxRelease hipDevicePrimaryCtxRelease
#define cuDevicePrimaryCtxReset hipDevicePrimaryCtxReset
#define cuDevicePrimaryCtxRetain hipDevicePrimaryCtxRetain
#define cuDevicePrimaryCtxSetFlags hipDevicePrimaryCtxSetFlags
#define cuDeviceSetMemPool hipDeviceSetMemPool
#define cuDeviceTotalMem hipDeviceTotalMem
#define cuDriverGetVersion hipDriverGetVersion
#define cuEventCreate hipEventCreateWithFlags
#define cuEventDestroy hipEventDestroy
#define cuEventElapsedTime hipEventElapsedTime
#define cuEventQuery hipEventQuery
#define cuEventRecord hipEventRecord
#define cuEventSynchronize hipEventSynchronize
#define cuFuncGetAttribute hipFuncGetAttribute
#define cuFuncSetAttribute hipFuncSetAttribute
#define cuFuncSetCacheConfig hipFuncSetCacheConfig
#define cuGetErrorName hipDrvGetErrorName
#define cuGetErrorString hipDrvGetErrorString
#define cuGetProcAddress hipGetProcAddress
#define cuInit hipInit
#define cuIpcCloseMemHandle hipIpcCloseMemHandle
#define cuIpcGetEventHandle hipIpcGetEventHandle
#define cuIpcGetMemHandle hipIpcGetMemHandle
#define cuIpcOpenEventHandle hipIpcOpenEventHandle
#define cuIpcOpenMemHandle hipIpcOpenMemHandle
#define cuLaunchCooperativeKernel hipModuleLaunchCooperativeKernel
#define cuLaunchCooperativeKernelMultiDevice   hipModuleLaunchCooperativeKernelMultiDevice
#define cuLaunchHostFunc hipLaunchHostFunc
#define cuLaunchKernel hipModuleLaunchKernel
#define cuLinkAddData hiprtcLinkAddData
#define cuLinkAddFile hiprtcLinkAddFile
#define cuLinkComplete hiprtcLinkComplete
#define cuLinkCreate hiprtcLinkCreate
#define cuLinkDestroy hiprtcLinkDestroy
#define cuMemAlloc hipMalloc
#define cuMemAlloc hipMalloc
#define cuMemAllocAsync hipMallocAsync
#define cuMemAllocFromPoolAsync hipMallocFromPoolAsync
#define cuMemAllocHost hipMemAllocHost
#define cuMemAllocHost hipMemAllocHost
#define cuMemAllocManaged hipMallocManaged
#define cuMemAllocManaged hipMallocManaged
#define cuMemAllocPitch hipMemAllocPitch
#define cuMemAllocPitch hipMemAllocPitch
#define cuMemFree hipFree
#define cuMemFree hipFree
#define cuMemFreeAsync hipFreeAsync
#define cuMemFreeHost hipHostFree
#define cuMemFreeHost hipHostFree
#define cuMemGetAddressRange hipMemGetAddressRange
#define cuMemGetAddressRange hipMemGetAddressRange
#define cuMemGetInfo hipMemGetInfo
#define cuMemGetInfo hipMemGetInfo
#define cuMemHostAlloc hipHostMalloc
#define cuMemHostGetDevicePointer hipHostGetDevicePointer
#define cuMemHostGetDevicePointer hipHostGetDevicePointer
#define cuMemHostGetFlags hipHostGetFlags
#define cuMemHostGetFlags hipHostGetFlags
#define cuMemHostRegister hipHostRegister
#define cuMemHostRegister hipHostRegister
#define cuMemHostUnregister hipHostUnregister
#define cuMemHostUnregister hipHostUnregister
#define cuMemPrefetchAsync hipMemPrefetchAsync
#define cuMemcpy2D hipMemcpyParam2D
#define cuMemcpy2D hipMemcpyParam2D
#define cuMemcpy2DAsync hipMemcpyParam2DAsync
#define cuMemcpy2DAsync hipMemcpyParam2DAsync
#define cuMemcpy2DUnaligned hipDrvMemcpy2DUnaligned
#define cuMemcpy2DUnaligned hipDrvMemcpy2DUnaligned
#define cuMemcpy3D hipDrvMemcpy3D
#define cuMemcpy3D hipDrvMemcpy3D
#define cuMemcpy3DAsync hipDrvMemcpy3DAsync
#define cuMemcpy3DAsync hipDrvMemcpy3DAsync
#define cuMemcpyAtoH hipMemcpyAtoH
#define cuMemcpyAtoH hipMemcpyAtoH
#define cuMemcpyDtoD hipMemcpyDtoD
#define cuMemcpyDtoD hipMemcpyDtoD
#define cuMemcpyDtoDAsync hipMemcpyDtoDAsync
#define cuMemcpyDtoDAsync hipMemcpyDtoDAsync
#define cuMemcpyDtoH hipMemcpyDtoH
#define cuMemcpyDtoH hipMemcpyDtoH
#define cuMemcpyDtoHAsync hipMemcpyDtoHAsync
#define cuMemcpyDtoHAsync hipMemcpyDtoHAsync
#define cuMemcpyHtoA hipMemcpyHtoA
#define cuMemcpyHtoA hipMemcpyHtoA
#define cuMemcpyHtoD hipMemcpyHtoD
#define cuMemcpyHtoD hipMemcpyHtoD
#define cuMemcpyHtoDAsync hipMemcpyHtoDAsync
#define cuMemcpyHtoDAsync hipMemcpyHtoDAsync
#define cuMemsetD16 hipMemsetD16
#define cuMemsetD16 hipMemsetD16
#define cuMemsetD16Async hipMemsetD16Async
#define cuMemsetD16Async hipMemsetD16Async
#define cuMemsetD32 hipMemsetD32
#define cuMemsetD32 hipMemsetD32
#define cuMemsetD32Async hipMemsetD32Async
#define cuMemsetD32Async hipMemsetD32Async
#define cuMemsetD8 hipMemsetD8
#define cuMemsetD8 hipMemsetD8
#define cuMemsetD8Async hipMemsetD8Async
#define cuMemsetD8Async hipMemsetD8Async
#define cuMipmappedArrayCreate hipMipmappedArrayCreate
#define cuMipmappedArrayDestroy hipMipmappedArrayDestroy
#define cuMipmappedArrayGetLevel hipMipmappedArrayGetLevel
#define cuModuleGetFunction hipModuleGetFunction
#define cuModuleGetGlobal hipModuleGetGlobal
#define cuModuleGetTexRef hipModuleGetTexRef
#define cuModuleLoad hipModuleLoad
#define cuModuleLoadData hipModuleLoadData
#define cuModuleLoadDataEx hipModuleLoadDataEx
#define cuModuleUnload hipModuleUnload
#define cuOccupancyMaxActiveBlocksPerMultiprocessor   hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
#define cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags   hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
#define cuOccupancyMaxPotentialBlockSize hipModuleOccupancyMaxPotentialBlockSize
#define cuOccupancyMaxPotentialBlockSizeWithFlags   hipModuleOccupancyMaxPotentialBlockSizeWithFlags
#define cuPointerGetAttribute hipPointerGetAttribute
#define cuPointerGetAttributes hipDrvPointerGetAttributes
#define cuPointerSetAttribute hipPointerSetAttribute
#define cuStreamAddCallback hipStreamAddCallback
#define cuStreamAttachMemAsync hipStreamAttachMemAsync
#define cuStreamBeginCapture hipStreamBeginCapture
#define cuStreamCreate hipStreamCreateWithFlags
#define cuStreamCreateWithPriority hipStreamCreateWithPriority
#define cuStreamDestroy hipStreamDestroy
#define cuStreamEndCapture hipStreamEndCapture
#define cuStreamGetCaptureInfo hipStreamGetCaptureInfo
#define cuStreamGetFlags hipStreamGetFlags
#define cuStreamGetPriority hipStreamGetPriority
#define cuStreamIsCapturing hipStreamIsCapturing
#define cuStreamQuery hipStreamQuery
#define cuStreamSynchronize hipStreamSynchronize
#define cuStreamUpdateCaptureDependencies hipStreamUpdateCaptureDependencies
#define cuStreamWaitEvent hipStreamWaitEvent
#define cuStreamWaitValue32 hipStreamWaitValue32
#define cuStreamWaitValue64 hipStreamWaitValue64
#define cuStreamWriteValue32 hipStreamWriteValue32
#define cuStreamWriteValue64 hipStreamWriteValue64
#define cuThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode
#define cudaFuncAttribute hipFuncAttribute
#define cudaFuncAttributes hipFuncAttributes
#endif  // __HIP_PLATFORM_AMD__
#endif  // MACROS_H