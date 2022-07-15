#include <catch2/catch.hpp>
#include <iostream>
#include <string>

#include "cu.hpp"

TEST_CASE("Test cu::Device", "[cu::Device]") {
  bool success{true};
  try {
    cu::init();
  } catch (cu::Error &error) {
    std::cerr << "cu::Error: " << error.what() << std::endl;
    success = false;
  }
  REQUIRE(success);
  cu::Device device(0);

  SECTION("Test Device.getName") {
    const std::string name = device.getName();
    std::cout << "Device name: " << name << std::endl;
    REQUIRE(name.size() > 0);
  }
}
