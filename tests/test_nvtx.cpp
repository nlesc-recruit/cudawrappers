#include <catch2/catch_test_macros.hpp>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvtx.hpp>

TEST_CASE("Test nvtx Marker creation", "[marker]") {
  nvtx::Marker marker("message");
}

TEST_CASE("Test nvtx Marker use", "[marker-use]") {
  nvtx::Marker marker("message", nvtx::Marker::red);
  marker.start();
  cu::init();
  marker.end();
}
