#include <cudawrappers/cu.hpp>
#include <cudawrappers/cufft.hpp>

#include "fstream"
#include "iostream"

const float DEFAULT_FLOAT_TOLERANCE = 1.e-6;
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

void generateSignal(cufftComplex *signal, unsigned signalSize) {
  const unsigned patchSize = 100;
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

struct ArrayComparisonStats {
  float totalDifference;
  unsigned exceedingCount;

  float meanDifference;
  float exceedingFraction;

  static struct ArrayComparisonStats computeStats(float totalDifference,
                                                  unsigned exceedingCount,
                                                  unsigned totalCount) {
    return ArrayComparisonStats{totalDifference, exceedingCount,
                                totalDifference / float(totalCount),
                                float(exceedingCount) / float(totalCount)};
  }
};
using ArrayComparisonStats = struct ArrayComparisonStats;

inline float norm(cufftComplex a) { return sqrt((a.x * a.x) + (a.y * a.y)); }

inline float absoluteDistance(cufftComplex a, cufftComplex b) {
  return norm(cufftComplex{a.x - b.x, b.y - b.y});
}

ArrayComparisonStats compareResults(
    cufftComplex *signal, cufftComplex *expected, unsigned linearSize,
    float tolerance = DEFAULT_FLOAT_TOLERANCE,
    float absolute_tolerance = DEFAULT_FLOAT_TOLERANCE) {
  float totalDiff = 0.;
  unsigned exceedingCount = 0;
  const unsigned totalElements = linearSize * linearSize;
  for (int i = 0; i < totalElements; i++) {
    const float adistance = absoluteDistance(signal[i], expected[i]);
    totalDiff += adistance;
    if (adistance > (absolute_tolerance + (tolerance * norm(expected[i])))) {
      exceedingCount++;
    }
  }

  return ArrayComparisonStats::computeStats(totalDiff, exceedingCount,
                                            totalElements);
}

std::ostream &operator<<(std::ostream &os, ArrayComparisonStats const &stats) {
  return os << "Total difference is " << stats.totalDifference << "\n"
            << "Mean difference is " << stats.meanDifference << "\n"
            << "Total count is " << stats.exceedingCount << "\n"
            << "Percentage count is " << stats.exceedingFraction << "\n";
}

int main(int argc, char *argv[]) {
  const unsigned fftSize = 1024u;
  const size_t arraySize = sizeof(cufftComplex) * fftSize * fftSize;

  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  cu::DeviceMemory in_dev(arraySize), out_dev(arraySize),
      out_test_dev(arraySize);
  cu::HostMemory in_host(arraySize), out_host(arraySize), out_test(arraySize);

  cu::Stream my_stream;
  cufftComplex *in = in_host;
  cufftComplex *out = out_host;
  generateSignal(in, fftSize);

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