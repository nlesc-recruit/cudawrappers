#include <array>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <iostream>
#include <string>

#include <cudawrappers/cu.hpp>

TEST_CASE("Test cu::Device", "[device]") {
  cu::init();
  cu::Device device(0);

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
}

TEST_CASE("Test copying cu::DeviceMemory and cu::HostMemory using cu::Stream",
          "[memcpy]") {
  cu::init();
  cu::Device device(0);

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

TEST_CASE("Test cu::Stream", "[stream]") {
  cu::init();
  cu::Device device(0);
  cu::Stream stream;

  SECTION("Test memAllocAsync") {
    const size_t size = 1024;
    cu::DeviceMemory mem = stream.memAllocAsync(size);
    CHECK(mem.size() == size);
    CHECK_NOTHROW(stream.memFreeAsync(mem));
    CHECK_NOTHROW(stream.synchronize());
  }
}
