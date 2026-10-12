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

TEST_CASE("Test nvtx mark", "[mark]") {
  CHECK_NOTHROW(nvtx::mark("test_mark"));
}

TEST_CASE("Test nvtx ThreadRange", "[thread-range]") {
  {
    nvtx::ThreadRange range("test_thread_range");
  }
}
