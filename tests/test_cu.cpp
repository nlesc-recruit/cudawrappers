#include <catch2/catch.hpp>
#include <cstring>
#include <iostream>
#include <string>

#include "cu.hpp"

TEST_CASE("Test cu::Device", "[cu::Device]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);

  SECTION("Test Device.getName") {
    const std::string name = device.getName();
    std::cout << "Device name: " << name << std::endl;
    REQUIRE(name.size() > 0);
  }

  SECTION("Test copying a std::array to the device and back") {
    const std::array<int, 3> src = {1, 2, 3};
    std::array<int, 3> tgt = {0, 0, 0};
    const size_t size = sizeof(src);

    cu::DeviceMemory mem(size);

    cu::Stream stream;
    stream.memcpyHtoDAsync(mem, src.data(), size);
    stream.memcpyDtoHAsync(tgt.data(), mem, size);
    stream.synchronize();

    REQUIRE(src == tgt);
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

    REQUIRE(!static_cast<bool>(memcmp(src, tgt, size)));
  }
}
