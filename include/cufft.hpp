#if !defined CUFFT_WRAPPER_H
#define CUFFT_WRAPPER_H

#include <cuda_fp16.h>
#include <cufft.h>
#include <cufftXt.h>

#include <complex>
#include <exception>

#include "cu.hpp"

namespace cufft {
class Error : public std::exception {
 public:
  Error(cufftResult result) : _result(result) {}

  virtual const char *what() const noexcept;

  operator cufftResult() const { return _result; }

 private:
  cufftResult _result;
};

template <typename Tin = cufftComplex, typename Tout = cufftComplex,
          unsigned DIM = 1>
class FFT {
 public:
  FFT(unsigned n, unsigned count);
  FFT(unsigned n, unsigned count, CUdeviceptr workArea, size_t workSize);

  FFT(FFT &) = delete;

  ~FFT() { checkCuFFTcall(cufftDestroy(plan)); }

  void setStream(CUstream stream) {
    checkCuFFTcall(cufftSetStream(plan, stream));
  }

  void execute(Tin *in, Tout *out, int direction = CUFFT_FORWARD) {
    // checkCuFFTcall(cufftExecC2C(plan, in, out, direction));
    checkCuFFTcall(cufftXtExec(plan, in, out, direction));
  }

 private:
  static void checkCuFFTcall(cufftResult result) {
    if (result != CUFFT_SUCCESS) throw Error(result);
  }

  cufftHandle plan;
};

template <>
inline FFT<cufftComplex, cufftComplex, 1>::FFT(unsigned n, unsigned count) {
  plan = 0;
  checkCuFFTcall(cufftPlan1d(&plan, n, CUFFT_C2C, count));
}

template <>
inline FFT<cufftComplex, cufftComplex, 1>::FFT(unsigned n, unsigned count,
                                               CUdeviceptr workArea,
                                               size_t workSize) {
  size_t neededWorkSize;

  plan = 0;
  checkCuFFTcall(cufftCreate(&plan));
  checkCuFFTcall(cufftSetAutoAllocation(plan, false));
  checkCuFFTcall(cufftMakePlan1d(plan, n, CUFFT_C2C, count, &neededWorkSize));

  if (workSize < neededWorkSize) throw Error(CUFFT_ALLOC_FAILED);

  checkCuFFTcall(cufftSetWorkArea(plan, (void *)workArea));
}

template <>
inline FFT<std::complex<half>, std::complex<half>, 1>::FFT(unsigned n,
                                                           unsigned count) {
  checkCuFFTcall(cufftCreate(&plan));

  long long size = n;
  size_t neededWorkSize;
  checkCuFFTcall(cufftXtMakePlanMany(plan, 1, &size, nullptr, 1, 1, CUDA_C_16F,
                                     nullptr, 1, 1, CUDA_C_16F, count,
                                     &neededWorkSize, CUDA_C_16F));
}
}  // namespace cufft

#endif
