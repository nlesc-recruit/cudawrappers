
#include <fstream>
#include <iostream>

#include <catch2/catch.hpp>
#include <cudawrappers/cufft.hpp>

const float DEFAULT_FLOAT_TOLERANCE = 1.e-6f;

void generateSignal(cufftComplex *signal, unsigned signalSize,
                    unsigned patchSize = 100) {
  for (int i = 0; i < patchSize; i++) {
    for (int j = 0; j < patchSize; j++)
      signal[(signalSize * i) + j] = cufftComplex{1, 1};
  }
}

void rescaleFFT(cufftComplex *signal, cufftComplex *output,
                unsigned signalSize) {
  const unsigned totalElements = signalSize * signalSize;
  const float rescale = float(totalElements);
  for (int i = 0; i < totalElements; i++) {
    output[i].x = signal[i].x / rescale;
    output[i].y = signal[i].y / rescale;
  }
}

void compare(float a, float b) {
  REQUIRE_THAT(a, Catch::Matchers::WithinRel(b, DEFAULT_FLOAT_TOLERANCE) ||
                      Catch::Matchers::WithinAbs(0, DEFAULT_FLOAT_TOLERANCE));
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
  const int fftSize = 1024;
  const size_t arraySize = fftSize * fftSize * sizeof(cufftComplex);

  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  cu::DeviceMemory in_dev(arraySize), out_dev(arraySize),
      out_test_dev(arraySize);
  cu::HostMemory in_host(arraySize), out_host(arraySize), out_test(arraySize);
  cufftComplex *in = in_host;
  cu::Stream stream;
  generateSignal(in, fftSize);

  cufft::FFT<cufftComplex, cufftComplex, 2> fft{fftSize, fftSize};
  fft.setStream(stream);
  const float percentageOfExpectedDifference = 0.9;

  stream.memcpyHtoDAsync(in_dev, in_host, arraySize);

  SECTION("Test forward fft") {
    fft.execute(in_dev, out_dev, CUFFT_FORWARD);
    stream.memcpyDtoHAsync(out_host, out_dev, arraySize);
    stream.synchronize();
    compare(out_test, in, fftSize);
  }

  SECTION("Test inverse fft") {
    fft.execute(in_dev, out_dev, CUFFT_FORWARD);
    fft.execute(out_dev, out_test_dev, CUFFT_INVERSE);
    stream.memcpyDtoHAsync(out_test, out_test_dev, arraySize);
    stream.synchronize();
    rescaleFFT(out_test, out_test, fftSize);
    compare(out_test, in, fftSize);
  }
}
