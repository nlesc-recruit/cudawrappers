#include <cudawrappers/cu.hpp>
#include <cudawrappers/cufft.hpp>

#include "fstream"
#include "iostream"


int main(int argc, char *argv[]) {
  const unsigned fftSize = 1024u;
  const size_t arraySize = sizeof(cufftComplex) * fftSize * fftSize;



  my_stream.memcpyHtoDAsync(in_dev, in, arraySize);

  cufft::FFT<cufftComplex, cufftComplex, 2> fft{fftSize, fftSize};
  fft.setStream(my_stream);
  fft.execute(in_dev, out_dev, CUFFT_FORWARD);
  my_stream.memcpyDtoHAsync(out_host, out_dev, arraySize);
  my_stream.synchronize();

  fft.execute(out_dev, out_test_dev, CUFFT_INVERSE);
  my_stream.memcpyDtoHAsync(out_test, out_test_dev, arraySize);
  my_stream.synchronize();

  rescaleFFT(out_test, out_test, fftSize);

  ArrayComparisonStats result = compareResults(out_test, in, fftSize, 1.e-6);

  std::cout << result << std::endl;
}