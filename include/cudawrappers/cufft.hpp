#if !defined CUFFT_H
#define CUFFT_H

#if defined(__HIP__)
#include <hip/hip_fp16.h>
#include <hipfft/hipfft.h>
#include <hipfft/hipfftXt.h>
#else
#include <cuda_fp16.h>
#include <cufft.h>
#include <cufftXt.h>
#endif

#include <array>
#include <exception>

#include "cudawrappers/cu.hpp"

/*
 * Error handling helper function, copied from cuda-samples Common/helper_cuda.h
 */
static const char *_cudaGetErrorEnum(cufftResult error) {
  switch (error) {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";

    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "CUFFT_INCOMPLETE_PARAMETER_LIST";

    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE";

    case CUFFT_PARSE_ERROR:
      return "CUFFT_PARSE_ERROR";

    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";

    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";

    case CUFFT_NOT_SUPPORTED:
      return "CUFFT_NOT_SUPPORTED";
  }

  return "<unknown>";
}

namespace cufft {

/*
 * Error
 */
class Error : public std::exception {
 public:
  explicit Error(cufftResult result) : result_(result) {}

  const char *what() const noexcept override {
    return _cudaGetErrorEnum(result_);
  }

  operator cufftResult() const { return result_; }

 private:
  cufftResult result_;
};

/*
 * FFT
 */
class FFT {
 public:
  FFT() = default;
  FFT &operator=(FFT &) = delete;
  FFT(FFT &) = delete;
  FFT &operator=(FFT &&other) noexcept {
    if (&other != this) {
      plan_ = other.plan_;
      other.plan_ = 0;
    }
    return *this;
  }
  FFT(FFT &&other) noexcept { *this = std::move(other); }

  ~FFT() { checkCuFFTCall(cufftDestroy(plan_)); }

  void setStream(cu::Stream &stream) const {
    checkCuFFTCall(cufftSetStream(plan_, stream));
  }

  void execute(cu::DeviceMemory &in, cu::DeviceMemory &out,
               const int direction) const {
    void *in_ptr = reinterpret_cast<void *>(static_cast<CUdeviceptr>(in));
    void *out_ptr = reinterpret_cast<void *>(static_cast<CUdeviceptr>(out));
    checkCuFFTCall(cufftXtExec(plan_, in_ptr, out_ptr, direction));
  }

 protected:
  void checkCuFFTCall(cufftResult result) const {
    if (result != CUFFT_SUCCESS) {
      throw Error(result);
    }
  }

  cufftHandle *plan() { return &plan_; }

 private:
  cufftHandle plan_{};
};

/*
 * FFT1D
 */
template <cudaDataType_t T>
class FFT1D : public FFT {
 public:
#if defined(__HIP__)
  __host__
#endif
  FFT1D(const int nx) = delete;
#if defined(__HIP__)
  __host__
#endif
  FFT1D(const int nx, const int batch) = delete;
};

template <>
FFT1D<CUDA_C_32F>::FFT1D(const int nx, const int batch) {
  checkCuFFTCall(cufftCreate(plan()));
  checkCuFFTCall(cufftPlan1d(plan(), nx, CUFFT_C2C, batch));
}

template <>
FFT1D<CUDA_C_32F>::FFT1D(const int nx) : FFT1D(nx, 1) {}

template <>
FFT1D<CUDA_C_16F>::FFT1D(const int nx, const int batch) {
  checkCuFFTCall(cufftCreate(plan()));
  const int rank = 1;
  size_t ws = 0;
  std::array<long long, 1> n{nx};
  const long long idist = 1;
  const long long odist = 1;
  const int istride = 1;
  const int ostride = 1;
  checkCuFFTCall(cufftXtMakePlanMany(*plan(), rank, n.data(), nullptr, istride,
                                     idist, CUDA_C_16F, nullptr, ostride, odist,
                                     CUDA_C_16F, batch, &ws, CUDA_C_16F));
}

template <>
FFT1D<CUDA_C_16F>::FFT1D(const int nx) : FFT1D(nx, 1) {}

/*
 * FFT2D
 */
template <cudaDataType_t T>
class FFT2D : public FFT {
 public:
#if defined(__HIP__)
  __host__
#endif
  FFT2D(const int nx, const int ny) = delete;
#if defined(__HIP__)
  __host__
#endif
  FFT2D(const int nx, const int ny, const int stride, const int dist,
        const int batch) = delete;
};

template <>
FFT2D<CUDA_C_32F>::FFT2D(const int nx, const int ny) {
  checkCuFFTCall(cufftCreate(plan()));
  checkCuFFTCall(cufftPlan2d(plan(), nx, ny, CUFFT_C2C));
}

template <>
FFT2D<CUDA_C_32F>::FFT2D(const int nx, const int ny, const int stride,
                         const int dist, const int batch) {
  checkCuFFTCall(cufftCreate(plan()));
  std::array<int, 2> n{nx, ny};
  checkCuFFTCall(cufftPlanMany(plan(), 2, n.data(), n.data(), stride, dist,
                               n.data(), stride, dist, CUFFT_C2C, batch));
}

template <>
FFT2D<CUDA_C_16F>::FFT2D(const int nx, const int ny, const int stride,
                         const int dist, const int batch) {
  checkCuFFTCall(cufftCreate(plan()));
  const int rank = 2;
  size_t ws = 0;
  std::array<long long, 2> n{nx, ny};
  const int istride = stride;
  const int ostride = stride;
  const long long int idist = dist;
  const long long int odist = dist;
  checkCuFFTCall(cufftXtMakePlanMany(*plan(), rank, n.data(), nullptr, istride,
                                     idist, CUDA_C_16F, nullptr, ostride, odist,
                                     CUDA_C_16F, batch, &ws, CUDA_C_16F));
}

template <>
FFT2D<CUDA_C_16F>::FFT2D(const int nx, const int ny)
    : FFT2D(nx, ny, 1, nx * ny, 1) {}

/*
 * FFT1DR2C
 */
template <cudaDataType_t T>
class FFT1DR2C : public FFT {
 public:
#if defined(__HIP__)
  __host__
#endif
  FFT1DR2C(const int nx) = delete;
#if defined(__HIP__)
  __host__
#endif
  FFT1DR2C(const int nx, const int batch) = delete;

#if defined(__HIP__)
  __host__
#endif
  FFT1DR2C(const int nx, const int batch, long long inembed,
           long long ouembed) = delete;
};

template <>
FFT1DR2C<CUDA_R_32F>::FFT1DR2C(const int nx, const int batch, long long inembed,
                               long long ouembed) {
  checkCuFFTCall(cufftCreate(plan()));
  const int rank = 1;
  size_t ws = 0;
  std::array<long long, 1> n{nx};
  const long long idist = inembed;
  const long long odist = ouembed;
  const long long istride = 1;
  const long long ostride = 1;

  checkCuFFTCall(cufftXtMakePlanMany(
      *plan(), rank, n.data(), &inembed, istride, idist, CUDA_R_32F, &ouembed,
      ostride, odist, CUDA_C_32F, batch, &ws, CUDA_C_32F));
}

/*
 * FFT1D_C2R
 */
template <cudaDataType_t T>
class FFT1DC2R : public FFT {
 public:
#if defined(__HIP__)
  __host__
#endif
  FFT1DC2R(const int nx) = delete;
#if defined(__HIP__)
  __host__
#endif
  FFT1DC2R(const int nx, const int batch) = delete;
#if defined(__HIP__)
  __host__
#endif
  FFT1DC2R(const int nx, const int batch, long long inembed,
           long long ouembed) = delete;
};

template <>
FFT1DC2R<CUDA_C_32F>::FFT1DC2R(const int nx, const int batch, long long inembed,
                               long long ouembed) {
  checkCuFFTCall(cufftCreate(plan()));
  const int rank = 1;
  size_t ws = 0;
  std::array<long long, 1> n{nx};
  const long long idist = inembed;
  const long long odist = ouembed;
  const int istride = 1;
  const int ostride = 1;

  checkCuFFTCall(cufftXtMakePlanMany(
      *plan(), rank, n.data(), &inembed, istride, idist, CUDA_C_32F, &ouembed,
      ostride, odist, CUDA_R_32F, batch, &ws, CUDA_C_32F));
}

}  // namespace cufft

#endif  // CUFFT_H