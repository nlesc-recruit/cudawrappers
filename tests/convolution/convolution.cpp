#include "cu.hpp"
#include "cufft.hpp"
#include "fstream"
#include "iostream"

void store_array_to_file(cufftComplex *array, std::string fname,
                         unsigned lSize) {
  std::ofstream ofile(fname, std::ios::binary);

  if (ofile.is_open()) {
    for (size_t k = 0; k < (lSize * lSize); k++) {
      ofile.write((char *)&array[k].x, sizeof(float));
      ofile.write((char *)&array[k].y, sizeof(float));
    }
    ofile.close();
    std::cout << "Completed" << std::endl;
  } else {
    std::cerr << "Cannot open file " << fname << std::endl;
    exit(1);
  }
}

int main(int argc, char *argv[]) {
  const unsigned fftSize = 1024u;
  const size_t arraySize = sizeof(cufftComplex) * fftSize * fftSize;

  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  cu::DeviceMemory in_dev(arraySize), out_dev(arraySize);
  cu::HostMemory in_host(arraySize), out_host(arraySize);

  cu::Stream my_stream;
  cufftComplex *in = in_host;
  cufftComplex *out = out_host;
  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < 100; j++) in[(fftSize * i) + j] = cufftComplex{1, 1};
  }
  my_stream.memcpyHtoDAsync(in_dev, in, arraySize);
  cufft::FFT<cufftComplex, cufftComplex, 2> fft{fftSize, fftSize};
  fft.setStream(my_stream);
  fft.execute(in_dev, out_dev, CUFFT_FORWARD);
  my_stream.memcpyDtoHAsync(out_host, out_dev, arraySize);
  my_stream.synchronize();
  store_array_to_file(out, "output.dat", fftSize);
}