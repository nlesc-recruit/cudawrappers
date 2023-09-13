#include <array>
#include <cstring>
#include <iostream>
#include <string>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <cuda.h>
#include <cudawrappers/cu.hpp>

TEST_CASE("Test cu::Device", "[device]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);

  SECTION("Test Device.getName") {
    const std::string name = device.getName();
    std::cout << "Device name: " << name << std::endl;
    CHECK(name.size() > 0);
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
}

TEST_CASE("Test zeroing cu::DeviceMemory", "[zero]") {
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
    mem.zero(size, stream);
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
}
