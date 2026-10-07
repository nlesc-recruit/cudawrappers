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

TEST_CASE("Test nvml::Device::getTemperature", "[device]") {
  nvml::Context context;
  nvml::Device device(0);
  const unsigned int temp = device.getTemperature(NVML_TEMPERATURE_GPU);
  CHECK(temp > 0);
  CHECK(temp < 150);
}

TEST_CASE("Test nvml::Device::getUtilizationRates", "[device]") {
  nvml::Context context;
  nvml::Device device(0);
  nvmlUtilization_t util = device.getUtilizationRates();
  CHECK(util.gpu >= 0);
  CHECK(util.gpu <= 100);
}

TEST_CASE("Test nvml::Device::getMemoryInfo", "[device]") {
  nvml::Context context;
  nvml::Device device(0);
  nvmlMemory_t mem = device.getMemoryInfo();
  CHECK(mem.total > 0);
  CHECK(mem.free <= mem.total);
}

TEST_CASE("Test nvml::Device::getDriverVersion", "[device]") {
  nvml::Context context;
  std::string version = nvml::Device::getDriverVersion();
  CHECK_FALSE(version.empty());
}

TEST_CASE("Test nvml::Device::getNvmlVersion", "[device]") {
  nvml::Context context;
  std::string version = nvml::Device::getNvmlVersion();
  CHECK_FALSE(version.empty());
}
