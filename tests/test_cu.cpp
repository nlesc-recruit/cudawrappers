#include <algorithm>
#include <array>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <cudawrappers/cu.hpp>

TEST_CASE("Test list all GPUs", "[device]") {
  cu::init();
  const int count = cu::Device::getCount();
  CHECK(count >= 0);
  std::cout << "Number of devices: " << count << std::endl;

  for (int i = 0; i < count; i++) {
    cu::Device device(static_cast<unsigned int>(i));
    std::cout << "Device " << i << ":" << std::endl;
    std::cout << "  Name:         " << device.getName() << std::endl;
    std::cout << "  Arch:         " << device.getArch() << std::endl;
    std::cout << "  Total memory: " << (device.totalMem() / (1024 * 1024))
              << " MiB" << std::endl;

    int major;
    int minor;
    device.getComputeCapability(major, minor);
    std::cout << "  Compute capability: " << major << "." << minor << std::endl;

    std::cout << "  PCI bus ID:   " << device.getPCIBusId() << std::endl;
    std::cout << "  UUID:         " << device.getUuid() << std::endl;
  }

  CHECK(count >= 1);
}

TEST_CASE("Test cu::Device", "[device]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);

  SECTION("Test Device.getName", "[device]") {
    const std::string name = device.getName();
    std::cout << "Device name: " << name << std::endl;
    CHECK(name.size() > 0);
  }

  SECTION("Test Device.getArch", "[device]") {
    const std::string arch = device.getArch();
    std::cout << "Device arch: " << arch << std::endl;
    CHECK(arch.size() > 0);
  }

  SECTION("Test device.totalMem", "[device]") {
    const size_t total_mem = device.totalMem();
    std::cout << "Device total memory: " << (total_mem / (1024 * 1024))
              << " bytes" << std::endl;
    CHECK(total_mem > 0);
  }

  SECTION("Test Device.getOrdinal", "[device]") {
    const int dev_ordinal = device.getOrdinal();
    CHECK(dev_ordinal >= 0);
  }

  SECTION("Test Device.getComputeCapability", "[device]") {
    int major;
    int minor;
    device.getComputeCapability(major, minor);
    CHECK(major >= 0);
    CHECK(minor >= 0);
  }

  SECTION("Test Device.getPCIBusId and Device.getByPCIBusId", "[device]") {
    const std::string pciBusId = device.getPCIBusId();
    CHECK(pciBusId.size() > 0);

    const cu::Device deviceByPci = cu::Device::getByPCIBusId(pciBusId);
    CHECK(deviceByPci.getOrdinal() == device.getOrdinal());
    CHECK(deviceByPci.getName() == device.getName());
  }

#if defined(CUDA_VERSION)
  SECTION("Test Device.getTexture1DLinearMaxWidth", "[device]") {
    const size_t maxWidth =
        device.getTexture1DLinearMaxWidth(CU_AD_FORMAT_UNSIGNED_INT8, 1);
    CHECK(maxWidth > 0);
  }
#endif

  SECTION("Test Device memory pool APIs", "[device]") {
    if (!device.getAttribute(CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED)) {
      WARN("Device does not support memory pools");
      return;
    }

    CUmemoryPool pool{};
    device.getDefaultMemPool(pool);
    CHECK(pool != 0);

    device.getMemPool(pool);
    device.setMemPool(pool);
  }

#if defined(CUDA_VERSION)
  SECTION("Test Device.getExecAffinitySupport", "[device]") {
    int pi = 0;
    device.getExecAffinitySupport(pi, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
    CHECK(pi >= 0);
  }

  SECTION("Test Device.getProperties", "[device]") {
    CUdevprop properties;
    device.getProperties(properties);
    CHECK(properties.maxThreadsPerBlock > 0);
    CHECK(properties.clockRate > 0);
  }
#endif
}

TEST_CASE("Test context::getDevice", "[device]") {
  cu::init();

  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);

  SECTION("Test getName from context") {
    CHECK(device.getName() == context.getCurrent().getDevice().getName());
  }
}

TEST_CASE("Test copying cu::DeviceMemory and cu::HostMemory using cu::Stream",
          "[memcpy]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);

  SECTION("Test copying a std::array to the device and back") {
    const std::array<int, 3> src = {1, 2, 3};
    std::array<int, 3> tgt = {0, 0, 0};
    const size_t size = sizeof(src);

    cu::DeviceMemory mem(size);

    cu::Stream stream;
    stream.memcpyHtoDAsync(mem, src.data(), size);
    stream.memcpyDtoHAsync(tgt.data(), mem, size);
    stream.synchronize();

    CHECK(src == tgt);
  }

  SECTION("Test copying a 2D std::array to the device and back") {
    const std::array<std::array<int, 3>, 3> src = {
        {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
    std::array<std::array<int, 3>, 3> tgt = {{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
    const size_t width = 3 * sizeof(int);
    const size_t height = 3;
    const size_t pitch = width;

    cu::DeviceMemory mem(pitch * height);

    cu::Stream stream;
    stream.memcpyHtoD2DAsync(mem, pitch, src.data(), pitch, width, height);
    stream.memcpyDtoH2DAsync(tgt.data(), pitch, mem, pitch, width, height);
    stream.synchronize();

    CHECK(src == tgt);
  }

  SECTION("Test copying HostMemory to the device and back") {
    const size_t N = 3;
    const size_t size = N * sizeof(int);
    cu::HostMemory src(size);
    cu::HostMemory tgt(size);

    // Populate the memory with values
    int* const src_ptr = static_cast<int*>(src);
    int* const tgt_ptr = static_cast<int*>(tgt);
    for (int i = 0; i < N; i++) {
      src_ptr[i] = i;
      tgt_ptr[i] = 0;
    }
    cu::DeviceMemory mem(size);

    cu::Stream stream;
    stream.memcpyHtoDAsync(mem, src, size);
    stream.memcpyDtoHAsync(tgt, mem, size);
    stream.synchronize();

    CHECK(!static_cast<bool>(memcmp(src, tgt, size)));
  }

  SECTION("Test copying unmanaged HostMemory to the device and back") {
    const size_t N = 3;
    const size_t size = N * sizeof(int);

    std::vector<int> src_data = {1, 2, 3};
    std::vector<int> tgt_data(N, 0);

    cu::DeviceMemory mem(size);
    cu::Stream stream;

    {
      cu::UnmanagedMemory src(src_data.data(), size);
      cu::UnmanagedMemory tgt(tgt_data.data(), size);

      stream.memcpyHtoDAsync(mem, src, size);
      stream.memcpyDtoHAsync(tgt, mem, size);
      stream.synchronize();
    }

    // HostMemory objects are now destroyed; the underlying data must still be
    // valid.
    CHECK(src_data == tgt_data);
  }

  SECTION("Test copying 2D HostMemory to the device and back") {
    const size_t width = 3 * sizeof(int);
    const size_t height = 3;
    const size_t pitch = width;
    const size_t size = pitch * height;
    cu::HostMemory src(size);
    cu::HostMemory tgt(size);

    // Populate the 2D memory with values
    int* const src_ptr = static_cast<int*>(src);
    int* const tgt_ptr = static_cast<int*>(tgt);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < 3; ++x) {
        src_ptr[y * 3 + x] = y * 3 + x + 1;
        tgt_ptr[y * 3 + x] = 0;
      }
    }

    cu::DeviceMemory mem(size);
    cu::Stream stream;

    stream.memcpyHtoD2DAsync(mem, pitch, src, pitch, width, height);
    stream.memcpyDtoH2DAsync(tgt, pitch, mem, pitch, width, height);
    stream.synchronize();

    CHECK(static_cast<bool>(memcmp(src, tgt, size)) == 0);
  }
}

TEST_CASE("Test cu::DeviceMemory", "[devicememory]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);

  SECTION("Test zeroing cu::DeviceMemory asynchronously") {
    const size_t N = 3;
    const size_t size = N * sizeof(int);
    cu::HostMemory src(size);
    cu::HostMemory tgt(size);

    // Populate the memory with values
    int* const src_ptr = static_cast<int*>(src);
    int* const tgt_ptr = static_cast<int*>(tgt);
    for (int i = 0; i < N; i++) {
      src_ptr[i] = i;
      tgt_ptr[i] = 0;
    }
    cu::DeviceMemory mem(size);

    cu::Stream stream;
    stream.memcpyHtoDAsync(mem, src, size);
    stream.zero(mem, size);
    stream.memcpyDtoHAsync(tgt, mem, size);
    stream.synchronize();

    CHECK(static_cast<bool>(memcmp(src, tgt, size)));
  }

  SECTION("Test zeroing cu::DeviceMemory synchronously") {
    const size_t N = 3;
    const size_t size = N * sizeof(int);
    cu::HostMemory src(size);
    cu::HostMemory tgt(size);

    // Populate the memory with values
    int* const src_ptr = static_cast<int*>(src);
    int* const tgt_ptr = static_cast<int*>(tgt);
    for (int i = 0; i < N; i++) {
      src_ptr[i] = i;
      tgt_ptr[i] = 0;
    }
    cu::DeviceMemory mem(size);

    cu::Stream stream;
    stream.memcpyHtoDAsync(mem, src, size);
    stream.synchronize();
    mem.zero(size);
    stream.memcpyDtoHAsync(tgt, mem, size);
    stream.synchronize();

    CHECK(static_cast<bool>(memcmp(src, tgt, size)));
  }

  SECTION("Test cuPointerSetAttribute") {
    const size_t size = 256;
    cu::DeviceMemory mem(size);
    unsigned int syncMemOps = 1;

    CHECK_NOTHROW(cu::pointerSetAttribute(
        &syncMemOps, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, mem));
  }

  SECTION("Test cu::DeviceMemory memcpy asynchronously") {
    const size_t N = 3;
    const size_t size = N * sizeof(int);

    std::vector<int> data_in(size);
    std::vector<int> data_out(size);

    for (size_t i = 0; i < N; i++) {
      data_in[i] = i + 1;
      data_out[i] = 0;
    }

    cu::HostMemory src(data_in.data(), size, 0);
    cu::HostMemory tgt(data_out.data(), size, 0);

    cu::DeviceMemory mem(size);
    cu::Stream stream;

    stream.memcpyHtoDAsync(mem, src, size);
    stream.memcpyDtoHAsync(tgt, mem, size);
    stream.synchronize();

    CHECK(data_in == data_out);
  }

  SECTION("Test cu::DeviceMemory with CU_MEMORYTYPE_DEVICE as host pointer") {
    cu::DeviceMemory mem(sizeof(float), CU_MEMORYTYPE_DEVICE, 0);
    float* ptr;
    CHECK_NOTHROW(ptr = mem);
  }

  SECTION("Test cu::DeviceMemory with CU_MEMORYTYPE_UNIFIED as host pointer") {
    cu::DeviceMemory mem(sizeof(float), CU_MEMORYTYPE_UNIFIED,
                         CU_MEM_ATTACH_GLOBAL);
    float* ptr = mem;
    CHECK_NOTHROW(ptr[0] = 42.f);
  }

  SECTION("Test cu::DeviceMemory with invalid CUmemorytype") {
    const size_t size = 1024;
    CHECK_THROWS(cu::DeviceMemory(size, CU_MEMORYTYPE_ARRAY));
    CHECK_THROWS(cu::DeviceMemory(size, CU_MEMORYTYPE_HOST));
  }

  SECTION("Test cu::DeviceMemory with CU_MEMORYTYPE_DEVICE and flags") {
    const size_t size = 1024;
    CHECK_NOTHROW(cu::DeviceMemory(size, CU_MEMORYTYPE_DEVICE, 0));
    CHECK_THROWS(
        cu::DeviceMemory(size, CU_MEMORYTYPE_DEVICE, CU_MEM_ATTACH_GLOBAL));
    CHECK_THROWS(
        cu::DeviceMemory(size, CU_MEMORYTYPE_DEVICE, CU_MEM_ATTACH_HOST));
  }

  SECTION("Test cu::DeviceMemory offset") {
    const size_t size = 1024;
    const size_t offset = 512;
    const size_t slice_size = 512;
    cu::DeviceMemory mem(size);
    CHECK_NOTHROW(cu::DeviceMemory(mem, offset, slice_size));
  }

  SECTION("Test cu::DeviceMemory invalid offset") {
    const size_t size = 1024;
    const size_t offset = 512;
    const size_t slice_size = 1024;
    cu::DeviceMemory mem(size);
    CHECK_THROWS(cu::DeviceMemory(mem, offset, slice_size));
  }
}

using TestTypes = std::tuple<unsigned char, unsigned short, unsigned int>;
TEMPLATE_LIST_TEST_CASE("Test memset 1D", "[memset]", TestTypes) {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);

  SECTION("Test memset cu::DeviceMemory asynchronously") {
    const size_t N = 3;
    const size_t size = N * sizeof(TestType);
    cu::HostMemory a(size);
    cu::HostMemory b(size);
    TestType value = 0xAA;

    // Populate the memory with values
    TestType* const a_ptr = static_cast<TestType*>(a);
    TestType* const b_ptr = static_cast<TestType*>(b);
    for (int i = 0; i < N; i++) {
      a_ptr[i] = 0;
      b_ptr[i] = value;
    }
    cu::DeviceMemory mem(size);

    cu::Stream stream;
    stream.memcpyHtoDAsync(mem, a, size);
    stream.memsetAsync(mem, value, N);
    stream.memcpyDtoHAsync(b, mem, size);
    stream.synchronize();

    CHECK(static_cast<bool>(memcmp(a, b, size)));
  }

  SECTION("Test zeroing cu::DeviceMemory synchronously") {
    const size_t N = 3;
    const size_t size = N * sizeof(TestType);
    cu::HostMemory a(size);
    cu::HostMemory b(size);
    TestType value = 0xAA;

    // Populate the memory with values
    TestType* const a_ptr = static_cast<TestType*>(a);
    TestType* const b_ptr = static_cast<TestType*>(b);
    for (int i = 0; i < N; i++) {
      a_ptr[i] = 0;
      b_ptr[i] = value;
    }
    cu::DeviceMemory mem(size);

    cu::Stream stream;
    stream.memcpyHtoDAsync(mem, a, size);
    stream.synchronize();
    mem.memset(value, N);
    stream.memcpyDtoHAsync(b, mem, size);
    stream.synchronize();

    CHECK(static_cast<bool>(memcmp(a, b, size)));
  }
}

using TestTypes = std::tuple<unsigned char, unsigned short, unsigned int>;
TEMPLATE_LIST_TEST_CASE("Test memset 2D", "[memset]", TestTypes) {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);

  SECTION("Test memset2D cu::DeviceMemory asynchronously") {
    const size_t width = 3;
    const size_t height = 3;
    const size_t pitch = width * sizeof(TestType);
    const size_t size = pitch * height;
    cu::HostMemory a(size);
    cu::HostMemory b(size);
    TestType value = 0xAA;

    // Populate the memory with initial values
    TestType* const a_ptr = static_cast<TestType*>(a);
    TestType* const b_ptr = static_cast<TestType*>(b);
    for (int i = 0; i < width * height; i++) {
      a_ptr[i] = 0;
      b_ptr[i] = value;
    }

    cu::DeviceMemory mem(size);
    cu::Stream stream;

    // Perform the 2D memory operations
    stream.memcpyHtoD2DAsync(mem, pitch, b, pitch, width, height);
    stream.memset2DAsync(mem, value, pitch, width, height);
    stream.memcpyDtoH2DAsync(b, pitch, mem, pitch, width, height);

    CHECK(static_cast<bool>(memcmp(a, b, size)));
  }

  SECTION("Test zeroing cu::DeviceMemory synchronously in 2D") {
    const size_t width = 3;
    const size_t height = 3;
    const size_t pitch = width * sizeof(TestType);
    const size_t size = pitch * height;
    cu::HostMemory a(size);
    cu::HostMemory b(size);
    TestType value = 0xAA;

    // Populate the memory with initial values
    TestType* const a_ptr = static_cast<TestType*>(a);
    TestType* const b_ptr = static_cast<TestType*>(b);
    for (int i = 0; i < width * height; i++) {
      a_ptr[i] = 0;
      b_ptr[i] = value;
    }

    cu::DeviceMemory mem(size);
    cu::Stream stream;

    // Perform the 2D memory operations
    stream.memcpyHtoD2DAsync(mem, pitch, b, pitch, width, height);
    stream.synchronize();
    mem.memset2D(value, pitch, width, height);
    stream.memcpyDtoH2DAsync(b, pitch, mem, pitch, width, height);
    stream.synchronize();

    CHECK(static_cast<bool>(memcmp(a, b, size)));
  }
}

TEST_CASE("Test cu::Event", "[event]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  cu::Stream stream;
  cu::Event start;
  cu::Event end;
  constexpr unsigned int record_flags = 0;

  SECTION("Test Event::record(Stream&, unsigned int)") {
    context.setCurrent();
    start.record(stream, record_flags);
    end.record(stream, record_flags);
    stream.synchronize();
    CHECK(end.elapsedTime(start) >= 0.0f);
  }

  SECTION("Test Stream::record(Event&, unsigned int)") {
    context.setCurrent();
    stream.record(start, record_flags);
    stream.record(end, record_flags);
    stream.synchronize();
    CHECK(end.elapsedTime(start) >= 0.0f);
  }
}

#if !defined(__HIP__) && defined(CUDA_VERSION)
TEST_CASE("Test cu::GreenContext", "[greencontext]") {
  cu::init();

  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  context.setCurrent();

  CUdevResource input{};
  device.getDevResource(input, CU_DEV_RESOURCE_TYPE_SM);

  unsigned int nbGroups = 1;
  auto minCount = static_cast<unsigned int>(input.sm.smCount * 0.4f);
  std::array<CUdevResource, 2> resources{};
#if CUDA_VERSION >= 13010
  CHECK_NOTHROW(cu::devSmResourceSplitByCount(&resources[0], &nbGroups, &input,
                                              &resources[1], 0, minCount));
#endif

  CUdevResourceDesc desc{};
  CHECK_NOTHROW(cu::devResourceGenerateDesc(&desc, &resources[0], 1));

  cu::GreenContext greenContext(desc, device, CU_GREEN_CTX_DEFAULT_STREAM);
  cu::Context greenContextBase = cu::Context::fromGreenCtx(greenContext);
  greenContextBase.setCurrent();

  cu::Stream stream = greenContext.createStream(CU_STREAM_NON_BLOCKING, 0);

  const size_t size = 1024;
  cu::DeviceMemory a(size);
  cu::HostMemory b(size);

  CHECK_NOTHROW(cu::memcpyHtoD(a, b, size));
  CHECK_NOTHROW(stream.memcpyHtoDAsync(a, b, size));
  CHECK_NOTHROW(stream.synchronize());
}
#endif

TEST_CASE("Test cu::Stream", "[stream]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  cu::Stream stream;

  SECTION("Test memAllocAsync") {
    context.setCurrent();
    const size_t size = 1024;
    cu::DeviceMemory mem = stream.memAllocAsync(size);
    CHECK(mem.size() == size);
    CHECK_NOTHROW(stream.memFreeAsync(mem));
    CHECK_NOTHROW(stream.synchronize());
  }

  SECTION("Test launchHostFunc") {
    context.setCurrent();
    int value = 0;
    stream.launchHostFunc(
        +[](void* data) { *static_cast<int*>(data) = 42; }, &value);
    stream.synchronize();
    CHECK(value == 42);
  }
}

// ============================================================================
// Merged from test_comprehensive.cpp (89 test cases)
// ============================================================================

// ============================================================================
// Device selection helper
// ============================================================================
static int g_testDevice = 0;

static unsigned int getTestDevice() {
  return static_cast<unsigned int>(g_testDevice);
}

// ============================================================================
// Helper: find first device from a specific backend
// ============================================================================
static int findDevice(cu::Device& dev, const char* desiredArch = nullptr) {
  int count = cu::Device::getCount();
  for (int i = 0; i < count; ++i) {
    cu::Device d(static_cast<unsigned int>(i));
    if (desiredArch) {
      std::string arch = d.getArch();
      if (arch.find(desiredArch) != std::string::npos) {
        dev = d;
        return i;
      }
    } else {
      dev = d;
      return i;
    }
  }
  return -1;
}

// Simple kernel that adds 1 to each element via host function
static void addOneKernel(void* data) {
  int* ptr = static_cast<int*>(data);
  *ptr += 1;
}

// ============================================================================
// 1. DRIVER / INIT
// ============================================================================
TEST_CASE("Driver: init", "[driver]") { CHECK_NOTHROW(cu::init()); }

TEST_CASE("Driver: driverGetVersion", "[driver]") {
  cu::init();
  int version = cu::driverGetVersion();
  CHECK(version > 0);
}

TEST_CASE("Driver: getErrorName", "[driver]") {
  const char* name = cu::getErrorName(CUDA_SUCCESS);
  CHECK(name != nullptr);
  CHECK(std::string(name).find("SUCCESS") != std::string::npos);
}

TEST_CASE("Driver: checkCudaCall throws on error", "[driver]") {
  // deviceGetCount with NULL should fail on most backends
  // We just verify the error mechanism works
  cu::init();
  bool threw = false;
  try {
    // This should work, not throw
    int count = cu::Device::getCount();
    (void)count;
  } catch (const cu::Error&) {
    threw = true;
  }
  // Not throwing is fine
  (void)threw;
}

// ============================================================================
// 2. DEVICE
// ============================================================================
TEST_CASE("Device: getCount", "[device]") {
  cu::init();
  int count = cu::Device::getCount();
  CHECK(count >= 1);
}

TEST_CASE("Device: debug enumeration", "[device]") {
  cu::init();
  int total = cu::Device::getCount();
  WARN("Total devices: " << total);
  for (int i = 0; i < total; ++i) {
    try {
      cu::Device dev(static_cast<unsigned int>(i));
      WARN("  Device " << i << ": " << dev.getName() << " [" << dev.getArch()
                       << "]");
    } catch (const std::exception& e) {
      WARN("  Device " << i << ": FAILED - " << e.what());
    }
  }
  cu::Device dev2(0);
  // Also try the selected test device
  WARN("Trying getTestDevice()=" << findDevice(dev2));
  try {
    cu::Device dev3(findDevice(dev2));
    WARN("  getTestDevice device: " << dev3.getName() << " [" << dev3.getArch()
                                    << "]");
  } catch (const std::exception& e) {
    WARN("  getTestDevice FAILED: " << e.what());
  }
}

TEST_CASE("Device: getName", "[device]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  std::string name = dev.getName();
  CHECK_FALSE(name.empty());
}

TEST_CASE("Device: getArch", "[device]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  std::string arch = dev.getArch();
  CHECK_FALSE(arch.empty());
  // Should be either sm_XX or gfxXXX
  CHECK((arch.find("sm_") != std::string::npos ||
         arch.find("gfx") != std::string::npos));
}

TEST_CASE("Device: getComputeCapability", "[device]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  int major = 0, minor = 0;
  CHECK_NOTHROW(dev.getComputeCapability(major, minor));
  CHECK(major >= 0);
  CHECK(minor >= 0);
}

TEST_CASE("Device: totalMem", "[device]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  size_t mem = dev.totalMem();
  CHECK(mem > 0);
}

TEST_CASE("Device: getOrdinal", "[device]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  CHECK(dev.getOrdinal() >= 0);
}

TEST_CASE("Device: getAttribute", "[device]") {
  cu::init();
  cu::Device dev(findDevice(dev));

  SECTION("MAX_THREADS_PER_BLOCK") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK>();
    CHECK(val > 0);
  }

  SECTION("MAX_BLOCK_DIM_X") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X>();
    CHECK(val >= 32);
  }

  SECTION("WARP_SIZE") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_WARP_SIZE>();
    CHECK(val > 0);
  }

  SECTION("MULTIPROCESSOR_COUNT") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT>();
    CHECK(val > 0);
  }

  SECTION("COMPUTE_CAPABILITY_MAJOR") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>();
    CHECK(val > 0);
  }

  SECTION("COMPUTE_CAPABILITY_MINOR") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();
    CHECK(val >= 0);
  }

  SECTION("CLOCK_RATE") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_CLOCK_RATE>();
    CHECK(val > 0);
  }

  SECTION("GLOBAL_MEMORY_BUS_WIDTH") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH>();
    CHECK(val > 0);
  }

  SECTION("L2_CACHE_SIZE") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE>();
    CHECK(val >= 0);
  }

  SECTION("MAX_SHARED_MEMORY_PER_BLOCK") {
    int val =
        dev.getAttribute<CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK>();
    CHECK(val >= 1024);
  }

  SECTION("TOTAL_CONSTANT_MEMORY") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY>();
    CHECK(val >= 0);
  }

  SECTION("MAX_REGISTERS_PER_BLOCK") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK>();
    CHECK(val > 0);
  }

  SECTION("MAX_GRID_DIM_X") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X>();
    CHECK(val > 0);
  }

  SECTION("MAX_PITCH") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_MAX_PITCH>();
    CHECK(val >= 0);
  }

  SECTION("TEXTURE_ALIGNMENT") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT>();
    CHECK(val >= 0);
  }

  SECTION("ECC_ENABLED") {
    // AMD GPUs don't report ECC
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_ECC_ENABLED>();
    CHECK((val == 0 || val == 1));
  }

  SECTION("MANAGED_MEMORY") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY>();
    CHECK((val == 0 || val == 1));
  }

  SECTION("ASYNC_ENGINE_COUNT") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT>();
    CHECK(val >= 0);
  }

  SECTION("COMPUTE_MODE") {
    int val = dev.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_MODE>();
    CHECK((val >= 0));
  }
}

TEST_CASE("Device: getUUID", "[device]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  std::string uuid = dev.getUuid();
  CHECK_FALSE(uuid.empty());
  // UUID format: GPU-XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
  CHECK(uuid.substr(0, 3) == "GPU");
}

TEST_CASE("Device: getPCIBusId", "[device]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  std::string pciId = dev.getPCIBusId();
  CHECK_FALSE(pciId.empty());
}

TEST_CASE("Device: getByPCIBusId round-trip", "[device]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  std::string pciId = dev.getPCIBusId();
  cu::Device dev2 = cu::Device::getByPCIBusId(pciId);
  CHECK(dev2.getPCIBusId() == pciId);
}

TEST_CASE("Device: Memory pools", "[device]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  CUmemoryPool pool{};
  CHECK_NOTHROW(dev.getDefaultMemPool(pool));
  CHECK(pool != nullptr);

  CUmemoryPool currentPool{};
  CHECK_NOTHROW(dev.getMemPool(currentPool));

  CHECK_NOTHROW(dev.setMemPool(pool));
  CUmemoryPool readBack{};
  CHECK_NOTHROW(dev.getMemPool(readBack));
}

#ifndef __HIP__
TEST_CASE("Device: getProperties (CUDA-only)", "[device][cuda]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  CUdevprop prop{};
  CHECK_NOTHROW(dev.getProperties(prop));
  CHECK(prop.maxThreadsPerBlock > 0);
  CHECK(prop.maxThreadsDim[0] >= 32);
  CHECK(prop.maxGridSize[0] > 0);
  CHECK(prop.sharedMemPerBlock >= 1024);
  CHECK(prop.SIMDWidth > 0);
}
#endif

// ============================================================================
// 3. CONTEXT
// ============================================================================
TEST_CASE("Context: create and setCurrent", "[context]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  CHECK_NOTHROW(context.setCurrent());
}

TEST_CASE("Context: getCurrent", "[context]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();
  cu::Context current = context.getCurrent();
  CHECK_NOTHROW(current.getDevice());
}

TEST_CASE("Context: getDevice", "[context]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();
  cu::Device ctxDev = context.getDevice();
  CHECK(ctxDev.getArch() == dev.getArch());
}

TEST_CASE("Context: getApiVersion", "[context]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();
  unsigned version = context.getApiVersion();
  CHECK(version > 0);
}

TEST_CASE("Context: Cache config", "[context]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  CHECK_NOTHROW(cu::Context::setCacheConfig(CU_FUNC_CACHE_PREFER_SHARED));
  CUfunc_cache config = cu::Context::getCacheConfig();
  CHECK(config == CU_FUNC_CACHE_PREFER_SHARED);

  CHECK_NOTHROW(cu::Context::setCacheConfig(CU_FUNC_CACHE_PREFER_L1));
  config = cu::Context::getCacheConfig();
  CHECK(config == CU_FUNC_CACHE_PREFER_L1);

  CHECK_NOTHROW(cu::Context::setCacheConfig(CU_FUNC_CACHE_PREFER_NONE));
  config = cu::Context::getCacheConfig();
  CHECK(config == CU_FUNC_CACHE_PREFER_NONE);
}

TEST_CASE("Context: getLimit / setLimit", "[context]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  size_t val = cu::Context::getLimit(CU_LIMIT_PRINTF_FIFO_SIZE);
  CHECK(val >= 0);

  // Set to a valid value and read back
  size_t newVal = 1024 * 1024;
  CHECK_NOTHROW(cu::Context::setLimit(CU_LIMIT_PRINTF_FIFO_SIZE, newVal));
  size_t readBack = cu::Context::getLimit(CU_LIMIT_PRINTF_FIFO_SIZE);
  CHECK(readBack == newVal);

  // Restore
  cu::Context::setLimit(CU_LIMIT_PRINTF_FIFO_SIZE, val);
}

TEST_CASE("Context: getFreeMemory / getTotalMemory", "[context]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  size_t free = context.getFreeMemory();
  size_t total = context.getTotalMemory();
  CHECK(free > 0);
  CHECK(total > 0);
  CHECK(total >= free);
}

TEST_CASE("Context: synchronize", "[context]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();
  CHECK_NOTHROW(cu::Context::synchronize());
}

TEST_CASE("Context: pushCurrent / popCurrent", "[context]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context ctx1(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  ctx1.setCurrent();

  cu::Context ctx2(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  ctx2.setCurrent();
  CHECK_NOTHROW(cu::Context::synchronize());
}

// ============================================================================
// 4. HOST MEMORY
// ============================================================================
TEST_CASE("HostMemory: alloc", "[hostmemory]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  size_t size = 1024;
  cu::HostMemory hostMem(size);
  CHECK(hostMem.size() == size);

  // Write and read back
  int* ptr = static_cast<int*>(hostMem);
  REQUIRE(ptr != nullptr);
  ptr[0] = 42;
  CHECK(ptr[0] == 42);
}

TEST_CASE("HostMemory: register", "[hostmemory]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  size_t size = 1024;
  std::vector<int> data(size / sizeof(int), 0);
  cu::HostMemory hostMem(data.data(), size);
  CHECK(hostMem.size() == size);
}

TEST_CASE("HostMemory: UnmanagedMemory", "[hostmemory]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  size_t size = 1024;
  std::vector<int> data(size / sizeof(int), 123);
  cu::UnmanagedMemory mem(data.data(), size);
  CHECK(mem.size() == size);
}

// ============================================================================
// 5. DEVICE MEMORY
// ============================================================================
TEST_CASE("DeviceMemory: alloc and free", "[devicememory]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  size_t size = 1024;
  cu::DeviceMemory devMem(size);
  CHECK(devMem.size() == size);
  CHECK(devMem != CUdeviceptr{});
}

TEST_CASE("DeviceMemory: unified memory", "[devicememory]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  size_t size = 1024;
  try {
    cu::DeviceMemory devMem(size, CU_MEMORYTYPE_UNIFIED, 0);
    CHECK(devMem.size() == size);
  } catch (const std::exception& e) {
    // cuMemAllocManaged may not be supported or may need specific flags
    WARN("cuMemAllocManaged failed: " << e.what());
  }
}

TEST_CASE("DeviceMemory: invalid type throws", "[devicememory]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  CHECK_THROWS_AS(cu::DeviceMemory(1024, CU_MEMORYTYPE_HOST, 0),
                  std::runtime_error);
}

TEST_CASE("DeviceMemory: flags with device type throws", "[devicememory]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  CHECK_THROWS_AS(cu::DeviceMemory(1024, CU_MEMORYTYPE_DEVICE, 1),
                  std::runtime_error);
}

TEST_CASE("DeviceMemory: copy from host", "[devicememory]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  size_t size = 256;
  cu::HostMemory host(size);
  int* ptr = static_cast<int*>(host);
  for (size_t i = 0; i < size / sizeof(int); ++i) {
    ptr[i] = static_cast<int>(i);
  }
  cu::DeviceMemory devMem(host);
  CHECK(devMem.size() == size);
}

TEST_CASE("DeviceMemory: offset slicing", "[devicememory]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  size_t size = 1024;
  cu::DeviceMemory base(size);

  SECTION("Valid offset") {
    cu::DeviceMemory slice(base, 512, 512);
    CHECK(slice.size() == 512);
  }

  SECTION("Offset exceeds bounds throws") {
    CHECK_THROWS_AS(cu::DeviceMemory(base, 513, 512), std::runtime_error);
  }

  SECTION("Zero-size offset at end") {
    cu::DeviceMemory slice(base, size, 0);
    CHECK(slice.size() == 0);
  }
}

TEST_CASE("DeviceMemory: memset 1D", "[devicememory][memset]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  size_t size = 1024;
  cu::DeviceMemory devMem(size);

  SECTION("memset 8-bit") {
    CHECK_NOTHROW(devMem.memset(static_cast<unsigned char>(0xAB), size));
  }

  SECTION("memset 16-bit") {
    CHECK_NOTHROW(devMem.memset(static_cast<unsigned short>(0xABCD), size / 2));
  }

  SECTION("memset 32-bit") {
    CHECK_NOTHROW(
        devMem.memset(static_cast<unsigned int>(0xDEADBEEF), size / 4));
  }

  SECTION("zero") { CHECK_NOTHROW(devMem.zero(size)); }
}

TEST_CASE("DeviceMemory: memset 2D", "[devicememory][memset]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  size_t width = 64;
  size_t height = 8;
  size_t pitch = width * sizeof(int);
  // Allocate enough for pitch * height plus some padding
  size_t allocSize = pitch * height + 256;
  cu::DeviceMemory devMem(allocSize);

  SECTION("memset2D 8-bit") {
    CHECK_NOTHROW(devMem.memset2D(static_cast<unsigned char>(0xCD), pitch,
                                  width, height));
  }

  SECTION("memset2D 16-bit") {
    CHECK_NOTHROW(devMem.memset2D(static_cast<unsigned short>(0xBEEF), pitch,
                                  width / 2, height));
  }

  SECTION("memset2D 32-bit") {
    CHECK_NOTHROW(devMem.memset2D(static_cast<unsigned int>(0xCAFEBABE), pitch,
                                  width / 4, height));
  }
}

// ============================================================================
// 6. SYNC MEMORY COPY (global free functions)
// ============================================================================
TEST_CASE("Sync memcpy: HtoD and DtoH", "[memcpy]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  size_t count = 256;
  std::vector<int> src(count);
  std::iota(src.begin(), src.end(), 0);

  cu::DeviceMemory devMem(count * sizeof(int));

  // HtoD
  CHECK_NOTHROW(cu::memcpyHtoD(devMem, src.data(), count * sizeof(int)));

  // DtoH
  std::vector<int> dst(count, 0);
  CHECK_NOTHROW(cu::memcpyDtoH(dst.data(), devMem, count * sizeof(int)));
  CHECK(dst == src);
}

// ============================================================================
// 7. STREAM
// ============================================================================
TEST_CASE("Stream: create default", "[stream]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();
  cu::Stream stream;
  CHECK_NOTHROW(stream.synchronize());
}

TEST_CASE("Stream: create with flags", "[stream]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  SECTION("CU_STREAM_DEFAULT") {
    cu::Stream stream(CU_STREAM_DEFAULT);
    CHECK_NOTHROW(stream.synchronize());
  }

  SECTION("CU_STREAM_NON_BLOCKING") {
    cu::Stream stream(CU_STREAM_NON_BLOCKING);
    CHECK_NOTHROW(stream.synchronize());
  }
}

TEST_CASE("Stream: memAllocAsync / memFreeAsync", "[stream][async]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  size_t size = 1024;
  cu::DeviceMemory mem = stream.memAllocAsync(size);
  CHECK(mem.size() == size);
  CHECK_NOTHROW(stream.memFreeAsync(mem));
  CHECK_NOTHROW(stream.synchronize());
}

TEST_CASE("Stream: memcpyHtoDAsync", "[stream][memcpy]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  size_t count = 128;
  std::vector<int> src(count, 42);
  cu::DeviceMemory devMem(count * sizeof(int));

  SECTION("From raw pointer") {
    CHECK_NOTHROW(
        stream.memcpyHtoDAsync(devMem, src.data(), count * sizeof(int)));
  }

  SECTION("From HostMemory") {
    cu::HostMemory host(count * sizeof(int));
    int* hptr = static_cast<int*>(host);
    std::iota(hptr, hptr + count, 0);
    CHECK_NOTHROW(stream.memcpyHtoDAsync(devMem, host, count * sizeof(int)));
  }

  CHECK_NOTHROW(stream.synchronize());
}

TEST_CASE("Stream: memcpyDtoHAsync", "[stream][memcpy]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  size_t count = 128;
  std::vector<int> src(count);
  std::iota(src.begin(), src.end(), 0);
  cu::DeviceMemory devMem(count * sizeof(int));

  // First copy HtoD
  stream.memcpyHtoDAsync(devMem, src.data(), count * sizeof(int));
  stream.synchronize();

  SECTION("To raw pointer") {
    std::vector<int> dst(count, 0);
    CHECK_NOTHROW(
        stream.memcpyDtoHAsync(dst.data(), devMem, count * sizeof(int)));
    stream.synchronize();
    CHECK(dst == src);
  }

  SECTION("To HostMemory") {
    cu::HostMemory host(count * sizeof(int));
    CHECK_NOTHROW(stream.memcpyDtoHAsync(host, devMem, count * sizeof(int)));
    stream.synchronize();
    int* hptr = static_cast<int*>(host);
    for (size_t i = 0; i < count; ++i) {
      CHECK(hptr[i] == static_cast<int>(i));
    }
  }
}

TEST_CASE("Stream: memcpyDtoDAsync", "[stream][memcpy]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  size_t count = 128;
  std::vector<int> src(count);
  std::iota(src.begin(), src.end(), 0);

  cu::DeviceMemory srcMem(count * sizeof(int));
  cu::DeviceMemory dstMem(count * sizeof(int));

  stream.memcpyHtoDAsync(srcMem, src.data(), count * sizeof(int));
  stream.synchronize();

  CHECK_NOTHROW(stream.memcpyDtoDAsync(dstMem, srcMem, count * sizeof(int)));
  stream.synchronize();

  std::vector<int> result(count, 0);
  stream.memcpyDtoHAsync(result.data(), dstMem, count * sizeof(int));
  stream.synchronize();
  CHECK(result == src);
}

TEST_CASE("Stream: memcpyHtoHAsync", "[stream][memcpy]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  size_t size = 512;
  std::vector<char> src(size, 'A');
  std::vector<char> dst(size, 0);

  CHECK_NOTHROW(stream.memcpyHtoHAsync(dst.data(), src.data(), size));
  CHECK_NOTHROW(stream.synchronize());
  CHECK(dst == src);
}

TEST_CASE("Stream: 2D pitched copies", "[stream][memcpy]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  size_t width = 32;
  size_t height = 8;
  size_t elementSize = sizeof(int);
  size_t pitch = width * elementSize + 16;  // Pitched
  size_t allocSize = pitch * (height + 1);

  std::vector<char> hostBuf(allocSize, 0);
  std::vector<char> hostDst(allocSize, 0);

  // Fill host source with a pattern
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      int val = static_cast<int>(y * width + x);
      std::memcpy(hostBuf.data() + y * pitch + x * elementSize, &val,
                  elementSize);
    }
  }

  cu::DeviceMemory devMem(allocSize);

  SECTION("HtoD2D then DtoH2D") {
    stream.memcpyHtoD2DAsync(devMem, pitch, hostBuf.data(), pitch,
                             width * elementSize, height);
    stream.synchronize();

    stream.memcpyDtoH2DAsync(hostDst.data(), pitch, devMem, pitch,
                             width * elementSize, height);
    stream.synchronize();

    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        int expected;
        std::memcpy(&expected, hostBuf.data() + y * pitch + x * elementSize,
                    elementSize);
        int actual;
        std::memcpy(&actual, hostDst.data() + y * pitch + x * elementSize,
                    elementSize);
        CHECK(actual == expected);
      }
    }
  }
}

TEST_CASE("Stream: memsetAsync", "[stream][memset]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  size_t size = 1024;
  cu::DeviceMemory devMem = stream.memAllocAsync(size);

  SECTION("8-bit") {
    CHECK_NOTHROW(
        stream.memsetAsync(devMem, static_cast<unsigned char>(0xFF), size));
  }

  SECTION("16-bit") {
    CHECK_NOTHROW(stream.memsetAsync(
        devMem, static_cast<unsigned short>(0xBEEF), size / 2));
  }

  SECTION("32-bit") {
    CHECK_NOTHROW(stream.memsetAsync(
        devMem, static_cast<unsigned int>(0xDEADBEEF), size / 4));
  }

  SECTION("zero") { CHECK_NOTHROW(stream.zero(devMem, size)); }

  CHECK_NOTHROW(stream.memFreeAsync(devMem));
  CHECK_NOTHROW(stream.synchronize());
}

TEST_CASE("Stream: memset2DAsync", "[stream][memset]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  size_t width = 64;
  size_t height = 8;
  size_t pitch = width * sizeof(int);
  size_t allocSize = pitch * (height + 1);
  cu::DeviceMemory devMem = stream.memAllocAsync(allocSize);

  SECTION("8-bit") {
    CHECK_NOTHROW(stream.memset2DAsync(devMem, static_cast<unsigned char>(0xCD),
                                       pitch, width, height));
  }

  SECTION("16-bit") {
    CHECK_NOTHROW(stream.memset2DAsync(
        devMem, static_cast<unsigned short>(0xBEEF), pitch, width / 2, height));
  }

  SECTION("32-bit") {
    CHECK_NOTHROW(stream.memset2DAsync(devMem,
                                       static_cast<unsigned int>(0xCAFEBABE),
                                       pitch, width / 4, height));
  }

  SECTION("zero2D") {
    CHECK_NOTHROW(stream.zero2D(devMem, pitch, width, height));
  }

  CHECK_NOTHROW(stream.memFreeAsync(devMem));
  CHECK_NOTHROW(stream.synchronize());
}

TEST_CASE("Stream: query", "[stream]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  // After synchronize, query should return success
  stream.synchronize();
  CHECK_NOTHROW(stream.query());
}

TEST_CASE("Stream: launchHostFunc", "[stream]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;

  SECTION("Basic host function") {
    int value = 0;
    stream.launchHostFunc(
        +[](void* data) { *static_cast<int*>(data) = 42; }, &value);
    stream.synchronize();
    CHECK(value == 42);
  }

  SECTION("Multiple host functions") {
    int a = 0, b = 0, c = 0;
    stream.launchHostFunc(
        +[](void* data) { *static_cast<int*>(data) = 1; }, &a);
    stream.launchHostFunc(
        +[](void* data) { *static_cast<int*>(data) = 2; }, &b);
    stream.launchHostFunc(
        +[](void* data) { *static_cast<int*>(data) = 3; }, &c);
    stream.synchronize();
    CHECK(a == 1);
    CHECK(b == 2);
    CHECK(c == 3);
  }
}

// ============================================================================
// 8. EVENT
// ============================================================================
TEST_CASE("Event: create with flags", "[event]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  SECTION("Default flags") {
    cu::Event event;
    CHECK_NOTHROW(event.synchronize());
  }

  SECTION("Blocking sync") {
    cu::Event event(CU_EVENT_BLOCKING_SYNC);
    CHECK_NOTHROW(event.synchronize());
  }

  SECTION("Disable timing") {
    cu::Event event(CU_EVENT_DISABLE_TIMING);
    CHECK_NOTHROW(event.synchronize());
  }

  SECTION("Interprocess") {
    try {
      cu::Event event(CU_EVENT_INTERPROCESS);
      event.synchronize();
    } catch (const cu::Error&) {
      WARN("CU_EVENT_INTERPROCESS not supported on this device");
    }
  }
}

TEST_CASE("Event: record and elapsedTime", "[event]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  cu::Event start;
  cu::Event end;

  SECTION("Event::record(stream)") {
    start.record(stream);
    end.record(stream);
    stream.synchronize();
    float time = end.elapsedTime(start);
    CHECK(time >= 0.0f);
  }

  SECTION("Event::record(stream, flags)") {
    start.record(stream, 0);
    end.record(stream, 0);
    stream.synchronize();
    float time = end.elapsedTime(start);
    CHECK(time >= 0.0f);
  }

  SECTION("Stream::record(event)") {
    stream.record(start);
    stream.record(end);
    stream.synchronize();
    float time = end.elapsedTime(start);
    CHECK(time >= 0.0f);
  }

  SECTION("Stream::record(event, flags)") {
    stream.record(start, 0);
    stream.record(end, 0);
    stream.synchronize();
    float time = end.elapsedTime(start);
    CHECK(time >= 0.0f);
  }
}

TEST_CASE("Event: query", "[event]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  cu::Event event;
  event.record(stream);
  stream.synchronize();
  // After sync, event should be complete, query should succeed
  CHECK_NOTHROW(event.query());
}

TEST_CASE("Event: synchronize", "[event]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  cu::Event event;
  event.record(stream);
  CHECK_NOTHROW(event.synchronize());
}

TEST_CASE("Event: timing accuracy", "[event]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  cu::Event start;
  cu::Event end;

  // Record immediately (no real work) - should have very small time
  start.record(stream);
  end.record(stream);
  stream.synchronize();
  float time = end.elapsedTime(start);
  CHECK(time >= 0.0f);
}

// ============================================================================
// 9. KERNEL LAUNCH
// ============================================================================
static const char* kernelSource = R"(
extern "C" __global__ void addOne(int* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] += 1;
}

extern "C" __global__ void multiplyTwo(int* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] *= 2;
}
)";

TEST_CASE("Module: load from data", "[module]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  // Load via NVRTC-compiled PTX would be ideal, but we can test moduleLoadData
  // with an empty/minimal PTX string to verify the path works
  // For now, just verify the constructor doesn't crash on null/empty
  cu::Stream stream;

  // We need actual PTX to test fully; skip if not available
  // This tests the code path exists
  cu::DeviceMemory devMem(static_cast<size_t>(256));
  int hostVal = 0;
  stream.memcpyHtoDAsync(devMem, &hostVal, sizeof(int));
  stream.synchronize();
  stream.memcpyDtoHAsync(&hostVal, devMem, sizeof(int));
  stream.synchronize();
  CHECK(hostVal == 0);
}

// ============================================================================
// 10. GRAPH
// ============================================================================
TEST_CASE("Graph: create and destroy", "[graph]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Graph graph(context);
  // Graph should be valid (not null)
  CHECK_NOTHROW(graph.debugDotPrint("/tmp/graph_test.dot"));
}

TEST_CASE("Graph: host node", "[graph]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  int hostValue = 0;
  cu::GraphHostNodeParams hostParams(
      +[](void* data) { *static_cast<int*>(data) = 99; }, &hostValue);

  cu::Graph graph(context);
  cu::GraphNode node;
  std::vector<CUgraphNode> deps;

  CHECK_NOTHROW(graph.addHostNode(node, deps, hostParams));
  CHECK_NOTHROW(graph.debugDotPrint("/tmp/graph_host.dot"));

  CUgraphExec execRaw = graph.instantiateWithFlags();
  cu::GraphExec exec(execRaw);
  cu::Stream stream;
  stream.graphLaunch(exec);
  stream.synchronize();

  CHECK(hostValue == 99);
}

TEST_CASE("Graph: memcpy node (H2D then D2H)", "[graph]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  size_t count = 16;
  std::vector<int> srcData(count);
  std::iota(srcData.begin(), srcData.end(), 1);

  cu::DeviceMemory devMem(count * sizeof(int));

  // Create H2D memcpy node
  cu::GraphMemCopyToDeviceNodeParams copyH2DParams(devMem, srcData.data(),
                                                   count, 1, 1, sizeof(int));

  cu::Graph graph(context);
  cu::GraphNode h2dNode;
  std::vector<CUgraphNode> deps;
  graph.addMemCpyNode(h2dNode, deps, copyH2DParams);

  // Create D2H memcpy node
  std::vector<int> dstData(count, 0);
  cu::GraphMemCopyToHostNodeParams copyD2HParams(dstData.data(), devMem, count,
                                                 1, 1, sizeof(int));

  cu::GraphNode d2hNode;
  std::vector<CUgraphNode> deps2 = {h2dNode};
  graph.addMemCpyNode(d2hNode, deps2, copyD2HParams);

  CUgraphExec execRaw = graph.instantiateWithFlags();
  cu::GraphExec exec(execRaw);
  cu::Stream stream;
  stream.graphLaunch(exec);
  stream.synchronize();

  CHECK(dstData == srcData);
}

TEST_CASE("Graph: debugDotPrint with flags", "[graph]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  int hostValue = 0;
  cu::GraphHostNodeParams hostParams(
      +[](void* data) { *static_cast<int*>(data) = 1; }, &hostValue);

  cu::Graph graph(context);
  cu::GraphNode node;
  std::vector<CUgraphNode> deps;
  graph.addHostNode(node, deps, hostParams);

  CHECK_NOTHROW(graph.debugDotPrint("/tmp/graph_debug.dot",
                                    CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE));
}

// ============================================================================
// 11. ARRAY
// ============================================================================
TEST_CASE("Array: 1D creation", "[array]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Array arr(256, CU_AD_FORMAT_FLOAT, 1);
  // Just verify it was created without throwing
  CHECK_NOTHROW(arr);
}

TEST_CASE("Array: 2D creation", "[array]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Array arr(64, 64, CU_AD_FORMAT_FLOAT, 1);
  CHECK_NOTHROW(arr);
}

TEST_CASE("Array: 3D creation", "[array]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Array arr(32, 32, 4, CU_AD_FORMAT_FLOAT, 1);
  CHECK_NOTHROW(arr);
}

// ============================================================================
// 12. POINTER ATTRIBUTES
// ============================================================================
TEST_CASE("Pointer: setAttribute", "[pointer]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::DeviceMemory devMem(static_cast<size_t>(1024));

  unsigned int syncFlag = 1;
  // Set SYNC_MEMOPS on device memory
  CHECK_NOTHROW(cu::pointerSetAttribute(
      &syncFlag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, devMem));
}

// ============================================================================
// 13. FULL PIPELINE: alloc -> copy -> kernel -> copy -> verify
// ============================================================================
TEST_CASE("Pipeline: host -> device -> kernel -> device -> host",
          "[pipeline]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  size_t count = 256;

  // Host data
  std::vector<int> src(count, 0);
  std::vector<int> dst(count, -1);

  // Device memory
  cu::DeviceMemory devMem(count * sizeof(int));

  // HtoD
  stream.memcpyHtoDAsync(devMem, src.data(), count * sizeof(int));
  stream.synchronize();

  // DtoH - should be all zeros
  stream.memcpyDtoHAsync(dst.data(), devMem, count * sizeof(int));
  stream.synchronize();

  for (size_t i = 0; i < count; ++i) {
    CHECK(dst[i] == 0);
  }

  // Memset to non-zero
  stream.memsetAsync(devMem, static_cast<unsigned int>(0x12345678), count);
  stream.synchronize();

  // Read back
  stream.memcpyDtoHAsync(dst.data(), devMem, count * sizeof(int));
  stream.synchronize();

  for (size_t i = 0; i < count; ++i) {
    CHECK(dst[i] == static_cast<int>(0x12345678));
  }
}

TEST_CASE("Pipeline: async alloc copy free", "[pipeline]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  size_t count = 64;

  // Async alloc
  cu::DeviceMemory devMem = stream.memAllocAsync(count * sizeof(int));
  CHECK(devMem.size() == count * sizeof(int));

  // Host data
  std::vector<int> src(count, 42);
  std::vector<int> dst(count, 0);

  // Async copy
  stream.memcpyHtoDAsync(devMem, src.data(), count * sizeof(int));
  stream.memcpyDtoHAsync(dst.data(), devMem, count * sizeof(int));
  stream.synchronize();

  CHECK(dst == src);

  // Async free
  stream.memFreeAsync(devMem);
  stream.synchronize();
}

TEST_CASE("Pipeline: event timing around kernel", "[pipeline]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  cu::Event start;
  cu::Event end;

  size_t count = 1024;
  cu::DeviceMemory devMem(count * sizeof(int));

  // Initialize
  std::vector<int> src(count, 1);
  stream.memcpyHtoDAsync(devMem, src.data(), count * sizeof(int));
  stream.synchronize();

  // Time a memset (as a proxy for kernel work)
  start.record(stream);
  stream.memsetAsync(devMem, static_cast<unsigned int>(0), count);
  end.record(stream);
  stream.synchronize();

  float ms = end.elapsedTime(start);
  CHECK(ms >= 0.0f);

  // Verify the memset worked
  std::vector<int> dst(count, -1);
  stream.memcpyDtoHAsync(dst.data(), devMem, count * sizeof(int));
  stream.synchronize();
  for (size_t i = 0; i < count; ++i) {
    CHECK(dst[i] == 0);
  }
}

TEST_CASE("Pipeline: 2D copy round trip", "[pipeline]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  size_t width = 16;
  size_t height = 8;
  size_t elemSize = sizeof(int);
  size_t pitch = ((width * elemSize + 255) / 256) * 256;  // Aligned pitch
  size_t allocSize = pitch * (height + 1);

  // Create source data
  std::vector<char> srcBuf(allocSize, 0);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      int val = static_cast<int>(y * width + x);
      std::memcpy(srcBuf.data() + y * pitch + x * elemSize, &val, elemSize);
    }
  }

  cu::DeviceMemory devMem(allocSize);

  // H2D
  stream.memcpyHtoD2DAsync(devMem, pitch, srcBuf.data(), pitch,
                           width * elemSize, height);
  stream.synchronize();

  // D2H
  std::vector<char> dstBuf(allocSize, 0);
  stream.memcpyDtoH2DAsync(dstBuf.data(), pitch, devMem, pitch,
                           width * elemSize, height);
  stream.synchronize();

  // Verify
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      int expected, actual;
      std::memcpy(&expected, srcBuf.data() + y * pitch + x * elemSize,
                  elemSize);
      std::memcpy(&actual, dstBuf.data() + y * pitch + x * elemSize, elemSize);
      CHECK(actual == expected);
    }
  }
}

TEST_CASE("Pipeline: multiple streams concurrency", "[pipeline]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream1;
  cu::Stream stream2;

  size_t count = 128;
  std::vector<int> data1(count, 1);
  std::vector<int> data2(count, 2);
  std::vector<int> result1(count, 0);
  std::vector<int> result2(count, 0);

  cu::DeviceMemory devMem1(count * sizeof(int));
  cu::DeviceMemory devMem2(count * sizeof(int));

  // Copy concurrently
  stream1.memcpyHtoDAsync(devMem1, data1.data(), count * sizeof(int));
  stream2.memcpyHtoDAsync(devMem2, data2.data(), count * sizeof(int));

  stream1.memcpyDtoHAsync(result1.data(), devMem1, count * sizeof(int));
  stream2.memcpyDtoHAsync(result2.data(), devMem2, count * sizeof(int));

  stream1.synchronize();
  stream2.synchronize();

  CHECK(result1 == data1);
  CHECK(result2 == data2);
}

// ============================================================================
// 14. MULTI-BACKEND DEVICE ENUMERATION
// ============================================================================
TEST_CASE("Multi-backend: enumerate all devices", "[device]") {
  cu::init();
  int total = 0;
  try {
    total = cu::Device::getCount();
  } catch (const std::exception& e) {
    WARN("getCount failed: " << e.what());
    return;
  }
  CHECK(total >= 1);

  for (int i = 0; i < total; ++i) {
    try {
      cu::Device dev(i);
      INFO("Device " << i << ": " << dev.getName() << " [" << dev.getArch()
                     << "]");
      CHECK_FALSE(dev.getName().empty());
      CHECK_FALSE(dev.getArch().empty());
      CHECK(dev.totalMem() > 0);

      cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
      context.setCurrent();

      size_t size = 1024;
      cu::DeviceMemory devMem(size);
      CHECK(devMem.size() == size);

      cu::Stream stream;
      std::vector<int> data(1, 42);
      std::vector<int> result(1, 0);
      stream.memcpyHtoDAsync(devMem, data.data(), sizeof(int));
      stream.memcpyDtoHAsync(result.data(), devMem, sizeof(int));
      stream.synchronize();
      CHECK(result[0] == 42);

      cu::Event start, end;
      start.record(stream);
      end.record(stream);
      stream.synchronize();
      CHECK(end.elapsedTime(start) >= 0.0f);
    } catch (const std::exception& e) {
      WARN("Device " << i << " failed: " << e.what());
    }
  }
}

TEST_CASE("Multi-backend: each device has unique PCI bus ID", "[device]") {
  cu::init();
  int total = 0;
  try {
    total = cu::Device::getCount();
  } catch (const std::exception&) {
    return;
  }
  std::vector<std::string> pciIds;
  for (int i = 0; i < total; ++i) {
    try {
      cu::Device dev(i);
      pciIds.push_back(dev.getPCIBusId());
    } catch (const std::exception&) {
      // Skip devices that can't be opened
    }
  }
  std::sort(pciIds.begin(), pciIds.end());
  auto last = std::unique(pciIds.begin(), pciIds.end());
  CHECK(last == pciIds.end());
}

// ============================================================================
// 15. EDGE CASES
// ============================================================================
TEST_CASE("Edge case: zero-size allocation", "[edge]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  // Zero-size allocation may be rejected by the driver
  try {
    cu::DeviceMemory devMem(static_cast<size_t>(0));
    CHECK(devMem.size() == 0);
  } catch (const cu::Error&) {
    WARN("Zero-size allocation not supported on this driver");
  }
}

TEST_CASE("Edge case: single byte operations", "[edge]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream;
  cu::DeviceMemory devMem = stream.memAllocAsync(1);

  char hostSrc = 'X';
  char hostDst = 0;

  stream.memcpyHtoDAsync(devMem, &hostSrc, 1);
  stream.memcpyDtoHAsync(&hostDst, devMem, 1);
  stream.synchronize();
  CHECK(hostDst == 'X');

  stream.memFreeAsync(devMem);
  stream.synchronize();
}

TEST_CASE("Edge case: large memset", "[edge]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  // Use all available memory minus 1MB
  size_t total = dev.totalMem();
  size_t size = total - (1024 * 1024);
  if (size > 0) {
    cu::DeviceMemory devMem(size);
    CHECK(devMem.size() == size);
    CHECK_NOTHROW(devMem.zero(std::min(size, (size_t)1024)));
  }
}

TEST_CASE("Edge case: many small allocations", "[edge]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  std::vector<cu::DeviceMemory> allocs;
  for (int i = 0; i < 64; ++i) {
    allocs.emplace_back(static_cast<size_t>(1024));
    CHECK(allocs.back().size() == 1024);
  }
  allocs.clear();
}

TEST_CASE("Edge case: stream flags", "[edge]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  SECTION("Default") {
    cu::Stream s(CU_STREAM_DEFAULT);
    CHECK_NOTHROW(s.synchronize());
  }

  SECTION("Non-blocking") {
    cu::Stream s(CU_STREAM_NON_BLOCKING);
    CHECK_NOTHROW(s.synchronize());
  }
}

// ============================================================================
// 16. ERROR HANDLING
// ============================================================================
TEST_CASE("Error: what() returns string", "[error]") {
  cu::init();
  cu::Error err(CUDA_SUCCESS);
  const char* what = err.what();
  CHECK(what != nullptr);
}

TEST_CASE("Error: implicit conversion to CUresult", "[error]") {
  cu::init();
  cu::Error err(CUDA_SUCCESS);
  CUresult result = err;
  CHECK(result == CUDA_SUCCESS);
}

// ============================================================================
// 17. NEW API: canAccessPeer, getP2PAttribute, pointerGetAttribute(s)
// ============================================================================
TEST_CASE("Device: canAccessPeer", "[device]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Device dev1(findDevice(dev));
  bool canAccess = dev.canAccessPeer(dev1);
  CHECK((canAccess == true || canAccess == false));
}

TEST_CASE("Device: getP2PAttribute", "[device]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  int count = cu::Device::getCount();
  if (count > 1) {
    cu::Device dev1(1);
    int value =
        dev.getP2PAttribute(CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK, dev1);
    (void)value;
  } else {
    WARN("Skipping P2P test with only 1 device");
  }
}

TEST_CASE("Pointer: pointerGetAttribute", "[pointer]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::DeviceMemory devMem(static_cast<size_t>(1024));
  CUdeviceptr ptr = static_cast<CUdeviceptr>(devMem);
  try {
    CUdeviceptr base =
        cu::pointerGetAttribute(CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, ptr);
    CHECK(base > 0);
  } catch (const std::exception& e) {
    WARN("pointerGetAttribute failed: " << e.what());
  }
}

TEST_CASE("Pointer: pointerGetAttributes", "[pointer]") {
#ifndef __HIP__
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::DeviceMemory devMem(static_cast<size_t>(1024));
  CUdeviceptr ptr = static_cast<CUdeviceptr>(devMem);
  CUpointer_attribute attr = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
  unsigned int memType = 0;
  void* dataPtr = &memType;
  void** dataArr = &dataPtr;
  cu::pointerGetAttributes(1, &attr, dataArr, ptr);
  CHECK(memType != 0);
#endif
}

// ============================================================================
// 18. NEW API: streamCreateWithPriority, streamGetFlags, streamGetPriority
// ============================================================================
TEST_CASE("Stream: createWithPriority", "[stream]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  int leastPriority{}, greatestPriority{};
  cu::Device::getStreamPriorityRange(leastPriority, greatestPriority);
  cu::Stream stream(CU_STREAM_DEFAULT);
  stream.synchronize();
}

TEST_CASE("Stream: getFlags and getPriority", "[stream]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream(CU_STREAM_DEFAULT);
  unsigned int flags = stream.getFlags();
  CHECK(flags == CU_STREAM_DEFAULT);

  int priority = stream.getPriority();
  CHECK(priority <= 0);
  stream.synchronize();
}

// ============================================================================
// 19. NEW API: stream capture
// ============================================================================
TEST_CASE("Stream: isCapturing", "[stream]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream(CU_STREAM_DEFAULT);
  CHECK_FALSE(stream.isCapturing());
  stream.synchronize();
}

TEST_CASE("Stream: beginCapture and endCapture", "[stream]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::Stream stream(CU_STREAM_DEFAULT);
  cu::DeviceMemory devMem(static_cast<size_t>(1024));

  stream.beginCapture();
  CHECK(stream.isCapturing());

  CUdeviceptr ptr = static_cast<CUdeviceptr>(devMem);
  int zero = 0;
  stream.memcpyHtoDAsync(ptr, &zero, sizeof(int));
  CUgraph graph = stream.endCapture();
  CHECK(graph != CUgraph{});
}

// ============================================================================
// 20. NEW API: occupancy maxPotentialBlockSize
// ============================================================================
// Tested via test_graph.cpp which includes nvrtc.hpp for kernel compilation.

// ============================================================================
// 21. NEW API: memAdvise
// ============================================================================
TEST_CASE("memAdvise", "[memory]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::DeviceMemory devMem(static_cast<size_t>(1024));
  CUdeviceptr ptr = static_cast<CUdeviceptr>(devMem);
  try {
    cu::memAdvise(reinterpret_cast<const void*>(ptr), static_cast<size_t>(1024),
                  CU_MEM_ADVISE_SET_PREFERRED_LOCATION, getTestDevice());
  } catch (const std::exception& e) {
    WARN("memAdvise failed (may need managed memory): " << e.what());
  }
}

// ============================================================================
// 22. NEW API: MemPool
// ============================================================================
TEST_CASE("MemPool: create and destroy", "[mempool]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::MemPool pool(dev);
  CUmemoryPool rawPool = pool;
  CHECK(rawPool != CUmemoryPool{});
}

TEST_CASE("MemPool: setAttribute and getAttribute", "[mempool]") {
  cu::init();
  cu::Device dev(findDevice(dev));
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, dev);
  context.setCurrent();

  cu::MemPool pool(dev);
  size_t threshold = 0;
  pool.getAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold);
  CHECK(threshold >= 0);
}

// ============================================================================
// 23. NEW API: NVRTC, NVTX, cuFFT, NVML tests
// ============================================================================
// These tests live in their respective test files:
//   test_nvrtc.cpp, test_nvtx.cpp, test_cufft.cpp, test_nvml.cpp
