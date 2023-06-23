#if !defined CUFFT_WRAPPER_H
#define CUFFT_WRAPPER_H

#include <cuda_fp16.h>
#include <cufft.h>
#include <cufftXt.h>

#include <complex>
#include <exception>

#include "cudawrappers/cu.hpp"

namespace cufft {
class Error : public std::exception {
 public:
  explicit Error(cufftResult result) : _result(result) {}

  const char *what() const noexcept override;

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
  FFT(unsigned nx, unsigned ny, unsigned stride, unsigned dist, unsigned count);

  FFT &operator=(FFT &) = delete;
  FFT(FFT &) = delete;

  FFT &operator=(FFT &&other) noexcept {
    if (other != this) {
      plan = other.plan;
      other.plan = 0;
    }
    return *this;
  }
  FFT(FFT &&other) noexcept { *this = std::move(other); }

  ~FFT() { checkCuFFTcall(cufftDestroy(plan)); }

  void setStream(CUstream stream) {
    checkCuFFTcall(cufftSetStream(plan, stream));
  }
  void execute(cu::DeviceMemory in, cu::DeviceMemory out,
               int direction = CUFFT_FORWARD) {
    execCuFFTXt(in, out, direction);
  }

 private:
  void execCuFFTXt(CUdeviceptr in, CUdeviceptr out, int direction) {
    checkCuFFTcall(cufftXtExec(plan, reinterpret_cast<void *>(in),
                               reinterpret_cast<void *>(out), direction));
  }
  static void checkCuFFTcall(cufftResult result) {
    if (result != CUFFT_SUCCESS) {
      throw Error(result);
    }
  }

  cufftHandle plan{};
};

template <>
inline FFT<cufftComplex, cufftComplex, 1>::FFT(unsigned n, unsigned count) {
  checkCuFFTcall(cufftCreate(&plan));
  checkCuFFTcall(cufftPlan1d(&plan, static_cast<int>(n), CUFFT_C2C,
                             static_cast<int>(count)));
}

template <>
inline FFT<cufftComplex, cufftComplex, 2>::FFT(unsigned nx, unsigned ny,
                                               unsigned stride, unsigned dist,
                                               unsigned count) {
  checkCuFFTcall(cufftCreate(&plan));
  std::unique_ptr<int> n(
      new int[2]{static_cast<int>(ny), static_cast<int>(nx)});

  checkCuFFTcall(cufftPlanMany(&plan, 2, n.get(), n.get(), stride, dist,
                               n.get(), stride, dist, CUFFT_C2C, count));
}

template <>
inline FFT<cufftComplex, cufftComplex, 2>::FFT(unsigned nx, unsigned ny) {
  checkCuFFTcall(cufftCreate(&plan));
  checkCuFFTcall(cufftPlan2d(&plan, nx, ny, CUFFT_C2C));
}

template <>
inline FFT<cufftComplex, cufftComplex, 1>::FFT(unsigned n, unsigned count,
                                               CUdeviceptr workArea,
                                               size_t workSize) {
  checkCuFFTcall(cufftCreate(&plan));
  checkCuFFTcall(cufftSetAutoAllocation(plan, false));

  size_t neededWorkSize{};
  checkCuFFTcall(cufftMakePlan1d(plan, static_cast<int>(n), CUFFT_C2C,
                                 static_cast<int>(count), &neededWorkSize));

  if (workSize < neededWorkSize) {
    throw Error(CUFFT_ALLOC_FAILED);
  }

  checkCuFFTcall(cufftSetWorkArea(plan, reinterpret_cast<void *>(workArea)));
}

template <>
inline FFT<std::complex<half>, std::complex<half>, 1>::FFT(unsigned n,
                                                           unsigned count) {
  checkCuFFTcall(cufftCreate(&plan));

  long long size = n;
  size_t neededWorkSize{};
  checkCuFFTcall(cufftXtMakePlanMany(plan, 1, &size, nullptr, 1, 1, CUDA_C_16F,
                                     nullptr, 1, 1, CUDA_C_16F, count,
                                     &neededWorkSize, CUDA_C_16F));
}
}  // namespace cufft

#endif
