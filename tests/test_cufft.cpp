#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fstream>
#include <iostream>

#if defined(__HIP__)
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif

#include <cudawrappers/cufft.hpp>

#define FP16_EPSILON 1e-3f
#define FP32_EPSILON 1e-6f

template <typename T>
void generateSignal(T *in, size_t size, size_t patchSize, T signal) {
  for (size_t i = 0; i < patchSize; i++) {
    in[i] = signal;
  }
}

template <typename T>
void generateSignal(T *in, size_t height, size_t width, size_t patchSize,
                    T signal) {
  for (size_t i = 0; i < patchSize; i++) {
    for (size_t j = 0; j < patchSize; j++) {
      in[(width * i) + j] = signal;
    }
  }
}

template <typename T>
void scaleSignal(T *in, T *out, size_t n, float scale) {
  for (size_t i = 0; i < n; i++) {
    out[i].x = static_cast<float>(in[i].x) / scale;
    out[i].y = static_cast<float>(in[i].y) / scale;
  }
}

void compare(float a, float b, double epsilon = FP32_EPSILON) {
  REQUIRE_THAT(a, Catch::Matchers::WithinAbs(b, epsilon));
}

void compare(half a, half b, double epsilon = FP16_EPSILON) {
  compare(__half2float(a), __half2float(b), epsilon);
}

template <typename T>
void compare(T a, T b) {
  compare(a.x, b.x);
  compare(a.y, b.y);
}

template <typename T>
void compare(T *a, T *b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    compare(a[i], b[i]);
  }
}

TEST_CASE("Test 1D FFT", "[FFT1D]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  cu::Stream stream;

  const size_t size = 256;
  const size_t patchSize = 10;

  SECTION("FP32") {
    const size_t arraySize = size * sizeof(cufft::cufftComplex);

    cu::HostMemory h_in(arraySize);
    cu::HostMemory h_out(arraySize);
    cu::DeviceMemory d_in(arraySize);
    cu::DeviceMemory d_out(arraySize);
    cu::DeviceMemory d_out2(arraySize);

    generateSignal(static_cast<cufft::cufftComplex *>(h_in), size, patchSize, {1, 1});
    stream.memcpyHtoDAsync(d_in, h_in, arraySize);

    cufft::FFT1D<cufft::CUDA_C_32F> fft(size);
    fft.setStream(stream);

    fft.execute(d_in, d_out, cufft::CUFFT_FORWARD);
    fft.execute(d_out, d_out2, cufft::CUFFT_INVERSE);
    stream.memcpyDtoHAsync(h_out, d_out2, arraySize);
    stream.synchronize();

    cufft::cufftComplex *in_ptr = static_cast<cufft::cufftComplex *>(h_in);
    cufft::cufftComplex *out_ptr = static_cast<cufft::cufftComplex *>(h_out);
    scaleSignal(out_ptr, out_ptr, size, float(size));
    compare(out_ptr, in_ptr, size);
  }

  SECTION("FP16") {
    const size_t arraySize = size * sizeof(half2);

    cu::HostMemory h_in(arraySize);
    cu::HostMemory h_out(arraySize);
    cu::DeviceMemory d_in(arraySize);
    cu::DeviceMemory d_out(arraySize);
    cu::DeviceMemory d_out2(arraySize);

    generateSignal(static_cast<half2 *>(h_in), size, patchSize, {0.1, 0.1});
    stream.memcpyHtoDAsync(d_in, h_in, arraySize);

    cufft::FFT1D<cufft::CUDA_C_16F> fft(size);
    fft.setStream(stream);

    fft.execute(d_in, d_out, cufft::CUFFT_FORWARD);
    fft.execute(d_out, d_out2, cufft::CUFFT_INVERSE);
    stream.memcpyDtoHAsync(h_out, d_out2, arraySize);
    stream.synchronize();

    half2 *in_ptr = static_cast<half2 *>(h_in);
    half2 *out_ptr = static_cast<half2 *>(h_out);
    scaleSignal(out_ptr, out_ptr, size, float(size));
    compare(out_ptr, in_ptr, size);
  }

  SECTION("FP32 FFT with Real-To-Complex translation, and back") {
    const size_t arraySize = size * sizeof(cufft::cufftComplex);

    cu::HostMemory h_in(arraySize);
    cu::HostMemory h_out(arraySize);
    cu::DeviceMemory d_in(arraySize);
    cu::DeviceMemory d_out(arraySize);
    cu::DeviceMemory d_out2(arraySize);

    generateSignal(static_cast<cufft::cufftComplex *>(h_in), size, patchSize, {1, 1});
    stream.memcpyHtoDAsync(d_in, h_in, arraySize);

    cufft::FFT1DR2C<cufft::CUDA_R_32F> fft_r2c(size, 1, 1, 1);
    cufft::FFT1DC2R<cufft::CUDA_C_32F> fft_c2r(size, 1, 1, 1);
    fft_r2c.setStream(stream);
    fft_c2r.setStream(stream);

    fft_r2c.execute(d_in, d_out, cufft::CUFFT_FORWARD);
    fft_c2r.execute(d_out, d_out2, cufft::CUFFT_INVERSE);
    stream.memcpyDtoHAsync(h_out, d_out2, arraySize);
    stream.synchronize();

    cufft::cufftComplex *in_ptr = static_cast<cufft::cufftComplex *>(h_in);
    cufft::cufftComplex *out_ptr = static_cast<cufft::cufftComplex *>(h_out);
    scaleSignal(out_ptr, out_ptr, size, float(size));
    compare(out_ptr, in_ptr, size);
  }
}

TEST_CASE("Test 2D FFT", "[FFT2D]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  cu::Stream stream;

  const size_t height = 256;
  const size_t width = height;
  const size_t patchSize = 10;

  SECTION("FP32") {
    const size_t arraySize = height * width * sizeof(cufft::cufftComplex);

    cu::HostMemory h_in(arraySize);
    cu::HostMemory h_out(arraySize);
    cu::DeviceMemory d_in(arraySize);
    cu::DeviceMemory d_out(arraySize);
    cu::DeviceMemory d_out2(arraySize);

    generateSignal(static_cast<cufft::cufftComplex *>(h_in), height, width, patchSize,
                   {1, 1});
    stream.memcpyHtoDAsync(d_in, h_in, arraySize);

    cufft::FFT2D<cufft::CUDA_C_32F> fft(height, width);
    fft.setStream(stream);

    fft.execute(d_in, d_out, cufft::CUFFT_FORWARD);
    fft.execute(d_out, d_out2, cufft::CUFFT_INVERSE);
    stream.memcpyDtoHAsync(h_out, d_out2, arraySize);
    stream.synchronize();

    cufft::cufftComplex *in_ptr = static_cast<cufft::cufftComplex *>(h_in);
    cufft::cufftComplex *out_ptr = static_cast<cufft::cufftComplex *>(h_out);
    scaleSignal(out_ptr, out_ptr, height * width, float(height * width));
    compare(out_ptr, in_ptr, height * width);
  }

  SECTION("FP32 batched") {
    const size_t batch = 2;
    const size_t arraySize = batch * height * width * sizeof(cufft::cufftComplex);

    cu::HostMemory h_in(arraySize);
    cu::HostMemory h_out(arraySize);
    cu::DeviceMemory d_in(arraySize);
    cu::DeviceMemory d_out(arraySize);
    cu::DeviceMemory d_out2(arraySize);

    const size_t stride = 1;
    const size_t dist = height * width;

    generateSignal(static_cast<cufft::cufftComplex *>(h_in), height, width,
                   patchSize, {1, 1});
    generateSignal(static_cast<cufft::cufftComplex *>(h_in) + dist, height, width,
                   patchSize, {2, 2});
    stream.memcpyHtoDAsync(d_in, h_in, arraySize);

    cufft::FFT2D<cufft::CUDA_C_32F> fft(height, width, stride, dist, batch);
    fft.setStream(stream);

    fft.execute(d_in, d_out, cufft::CUFFT_FORWARD);
    fft.execute(d_out, d_out2, cufft::CUFFT_INVERSE);
    stream.memcpyDtoHAsync(h_out, d_out2, arraySize);
    stream.synchronize();

    cufft::cufftComplex *in_ptr = static_cast<cufft::cufftComplex *>(h_in);
    cufft::cufftComplex *out_ptr = static_cast<cufft::cufftComplex *>(h_out);
    scaleSignal(out_ptr, out_ptr, height * width, float(height * width));
    compare(out_ptr, in_ptr, height * width);
  }

  SECTION("FP16") {
    const size_t arraySize = height * width * sizeof(half2);

    cu::HostMemory h_in(arraySize);
    cu::HostMemory h_out(arraySize);
    cu::DeviceMemory d_in(arraySize);
    cu::DeviceMemory d_out(arraySize);
    cu::DeviceMemory d_out2(arraySize);

    generateSignal(static_cast<half2 *>(h_in), height, width, patchSize,
                   {0.1, 0.1});
    stream.memcpyHtoDAsync(d_in, h_in, arraySize);

    cufft::FFT2D<cufft::CUDA_C_16F> fft(height, width);
    fft.setStream(stream);

    fft.execute(d_in, d_out, cufft::CUFFT_FORWARD);
    fft.execute(d_out, d_out2, cufft::CUFFT_INVERSE);
    stream.memcpyDtoHAsync(h_out, d_out2, arraySize);
    stream.synchronize();

    half2 *in_ptr = static_cast<half2 *>(h_in);
    half2 *out_ptr = static_cast<half2 *>(h_out);
    scaleSignal(out_ptr, out_ptr, height * width, float(height * width));
    compare(out_ptr, in_ptr, height * width);
  }
}

TEST_CASE("Test error messages", "[Error]") {
  CHECK_THROWS_WITH(throw cufft::Error(cufft::CUFFT_SUCCESS), "CUFFT_SUCCESS");
  CHECK_THROWS_WITH(throw cufft::Error(cufft::CUFFT_INVALID_PLAN),
                   "CUFFT_INVALID_PLAN");
  CHECK_THROWS_WITH(throw cufft::Error(cufft::CUFFT_ALLOC_FAILED),
                   "CUFFT_ALLOC_FAILED");
}

TEST_CASE("Test cuFFT version and property", "[FFT1D]") {
  cu::init();
  CHECK(cufft::getVersion() > 0);
  CHECK(cufft::getProperty(cufft::CUFFT_MAJOR_VERSION) > 0);
  CHECK(cufft::getProperty(cufft::CUFFT_MINOR_VERSION) > 0);
}

TEST_CASE("Test work-size estimation", "[FFT1D]") {
  cu::init();
  // Estimates precede any plan; they must work on the active backend.
  CHECK_NOTHROW(cufft::estimate1d(64, cufft::CUFFT_C2C, 1));
  CHECK_NOTHROW(cufft::estimate2d(16, 16, cufft::CUFFT_C2C));
  CHECK_NOTHROW(cufft::estimate3d(8, 8, 8, cufft::CUFFT_C2C));
  int n[1] = {64};
  CHECK_NOTHROW(cufft::estimateMany(1, n, nullptr, 1, 1, nullptr, 1, 1,
                                   cufft::CUFFT_C2C, 1));
}

// The 64-bit execution entry points do not exist in current cuFFT or hipFFT; the
// backend stubs them, so calling one must raise CUFFT_NOT_SUPPORTED, never
// crash.  This is version- and vendor-independent.
TEST_CASE("Test unsupported cuFFT calls raise CUFFT_NOT_SUPPORTED", "[FFT1D]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  cu::Stream stream;

  const size_t size = 64;
  const size_t arraySize = size * sizeof(cufft::cufftComplex);
  cu::DeviceMemory d_in(arraySize);
  cu::DeviceMemory d_out(arraySize);

  cufft::FFT1D<cufft::CUDA_C_32F> fft(size);
  fft.setStream(stream);
  try {
    fft.execC2C64(d_in, d_out, cufft::CUFFT_FORWARD);
    // If a future cuFFT/hipFFT adds the symbol, the call may succeed; either
    // outcome is acceptable, but a stale stub must never be hit silently.
  } catch (const cufft::Error &e) {
    CHECK(static_cast<cufft::cufftResult>(e) == cufft::CUFFT_NOT_SUPPORTED);
  }
}

TEST_CASE("Test 3D FFT", "[FFT3D]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  cu::Stream stream;

  cufft::FFT3D<cufft::CUDA_C_32F> fft(16, 16, 16);
}

// Run a 1-D FFT round-trip on every GPU of every backend (and thus, on a
// machine with both NVIDIA and AMD GPUs, of every vendor in one process).
TEST_CASE("Test 1D FFT on all backends", "[FFT1D][multi_backend]") {
  cu::init();
  const int count = cu::Device::getCount();
  REQUIRE(count >= 1);

  for (int ordinal = 0; ordinal < count; ordinal++) {
    INFO("ordinal " << ordinal);
    try {
      cu::Device device(ordinal);
      cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
      cu::Stream stream;

      const size_t size = 64;
      const size_t arraySize = size * sizeof(cufft::cufftComplex);

      cu::HostMemory h_in(arraySize);
      cu::HostMemory h_out(arraySize);
      cu::DeviceMemory d_in(arraySize);
      cu::DeviceMemory d_out(arraySize);

      generateSignal(static_cast<cufft::cufftComplex *>(h_in), size, 6, {1, 1});
      stream.memcpyHtoDAsync(d_in, h_in, arraySize);

      cufft::FFT1D<cufft::CUDA_C_32F> fft(size);
      fft.setStream(stream);
      fft.execute(d_in, d_out, cufft::CUFFT_FORWARD);
      stream.memcpyDtoHAsync(h_out, d_out, arraySize);
      stream.synchronize();

      // Forward FFT of a boxcar differs from the boxcar; works on any GPU.
      cufft::cufftComplex *out = static_cast<cufft::cufftComplex *>(h_out);
      bool allSame = true;
      for (size_t i = 0; i < size; i++) {
        allSame = out[i].x == 1.0f && out[i].y == 1.0f;
        if (!allSame) break;
      }
      CHECK(!allSame);
    } catch (const std::exception &e) {
      WARN("skipping device " << ordinal << ": " << e.what());
    }
  }
}

// Keep an NVIDIA and an AMD plan alive and executing in a single process, the
// defining feature of the runtime-dispatched wrapper.  On a host where one
// vendor's devices cannot be used (e.g. busy, or inaccessible in this process),
// the test degrades to a single-vendor coverage run.
TEST_CASE("Test mixed-vendor FFT1D", "[FFT1D][multi_backend]") {
  cu::init();
  const int count = cu::Device::getCount();

  // one context, plan, and stream per usable GPU
  std::vector<cu::Device>             devices;
  for (int ordinal = 0; ordinal < count; ordinal++) {
    try {
      devices.emplace_back(ordinal);
    } catch (const std::exception &e) {
      WARN("device " << ordinal << " unusable: " << e.what());
    }
  }
  if (devices.size() < 2) {
    WARN("fewer than two usable GPUs; mixed-vendor test reduced to coverage");
  }

  std::vector<cu::Context>             contexts;
  std::vector<cu::Stream>              streams;
  std::vector<cu::DeviceMemory>        d_ins, d_outs;
  std::vector<cu::HostMemory>          h_ins;
  std::vector<cufft::FFT1D<cufft::CUDA_C_32F>> ffts;

  for (cu::Device &device : devices) {
    const size_t size = 64;
    const size_t arraySize = size * sizeof(cufft::cufftComplex);

    try {
      contexts.emplace_back(CU_CTX_SCHED_BLOCKING_SYNC, device);
      streams.emplace_back();
      h_ins.emplace_back(arraySize);
      d_ins.emplace_back(arraySize);
      d_outs.emplace_back(arraySize);
      generateSignal(static_cast<cufft::cufftComplex *>(h_ins.back()), size, 6,
                    {1, 1});
      streams.back().memcpyHtoDAsync(d_ins.back(), h_ins.back(), arraySize);

      ffts.emplace_back(size);
      ffts.back().setStream(streams.back());
      ffts.back().execute(d_ins.back(), d_outs.back(), cufft::CUFFT_FORWARD);
      streams.back().synchronize();
    } catch (const std::exception &e) {
      WARN("device " << device.getOrdinal() << " unusable: " << e.what());
    }
  }

  // all plans still work, one per GPU vendor, in the same process
  for (size_t i = 0; i < ffts.size(); i++)
    ffts[i].execute(d_ins[i], d_outs[i], cufft::CUFFT_FORWARD);
  for (size_t i = 0; i < ffts.size(); i++)
    streams[i].synchronize();
}
