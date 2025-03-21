#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvml.hpp>

TEST_CASE("Test nvml::Context", "[context]") { nvml::Context context; }

TEST_CASE("Test nvml::Device with device number", "[device]") {
  nvml::Context context;
  nvml::Device device(0);
}

TEST_CASE("Test nvml::Device::getClock", "[device]") {
  nvml::Context context;
  nvml::Device device(0);
  const unsigned int clockMHz =
      device.getClock(NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT);
  REQUIRE(clockMHz > 0);
}

TEST_CASE("Test nvml::Device::getPower", "[device]") {
  nvml::Context context;
  nvml::Device device(0);
  const unsigned int power = device.getPower();
  REQUIRE(power > 0);
}

TEST_CASE("Test nvml::Device with device", "[device]") {
  cu::init();
  cu::Device cu_device(0);
  nvml::Context nvml_context;
  nvml::Device nvml_device(cu_device);
}
