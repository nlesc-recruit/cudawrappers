
#include <fstream>
#include <iostream>

#include <catch2/catch.hpp>
#include <cudawrappers/cufft.hpp>

void generateSignal(cufftComplex *in, size_t height, size_t width,
                    size_t patchSize) {
  for (size_t i = 0; i < patchSize; i++) {
    for (size_t j = 0; j < patchSize; j++)
      in[(width * i) + j] = cufftComplex{1, 1};
  }
}

void scaleSignal(cufftComplex *in, cufftComplex *out, size_t n) {
  for (size_t i = 0; i < n; i++) {
    out[i].x = in[i].x / float(n);
    out[i].y = in[i].y / float(n);
  }
}

void compare(float a, float b) {
  REQUIRE_THAT(a, Catch::Matchers::WithinRel(b, 1e-3f) ||
                      Catch::Matchers::WithinAbs(0, 1e-6f));
}

void compare(cuFloatComplex a, cuFloatComplex b) {
  compare(a.x, b.x);
  compare(a.y, b.y);
}

void compare(cufftComplex *a, cufftComplex *b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    const float a_real = a[i].x;
    const float a_imag = a[i].x;
    const float b_real = b[i].x;
    const float b_imag = b[i].x;
    compare(a_real, b_real);
    compare(a_imag, b_imag);
    compare(a[i], b[i]);
  }
}

TEST_CASE("Test FFT is correct: 2D", "[correctness]") {
  const size_t height = 256;
  const size_t width = height;
  const size_t arraySize = height * width * sizeof(cufftComplex);
  const size_t patchSize = 10;

  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  cu::Stream stream;
  cu::HostMemory h_in(arraySize);
  cu::HostMemory h_out(arraySize);
  cu::DeviceMemory d_in(arraySize);
  cu::DeviceMemory d_out(arraySize);
  cu::DeviceMemory d_out2(arraySize);

  generateSignal(static_cast<cufftComplex *>(h_in), height, width, patchSize);
  stream.memcpyHtoDAsync(d_in, h_in, arraySize);

  cufft::FFT<cufftComplex, cufftComplex, 2> fft{height, width};
  fft.setStream(stream);

  fft.execute(d_in, d_out, CUFFT_FORWARD);
  fft.execute(d_out, d_out2, CUFFT_INVERSE);
  stream.memcpyDtoHAsync(h_out, d_out2, arraySize);
  stream.synchronize();
  scaleSignal(h_out, h_out, height * width);
  compare(h_out, h_in, height * width);
}
