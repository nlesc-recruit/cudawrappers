#if !defined CUFFT_WRAPPER_H
#define CUFFT_WRAPPER_H

#include <cuda_fp16.h>
#include <cufft.h>
#include <cufftXt.h>

#include <exception>

#include "cudawrappers/cu.hpp"

namespace {
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

    case CUFFT_LICENSE_ERROR:
      return "CUFFT_LICENSE_ERROR";

    case CUFFT_NOT_SUPPORTED:
      return "CUFFT_NOT_SUPPORTED";
  }

  return "<unknown>";
}

}  // namespace
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
  FFT(){};
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

  void setStream(CUstream stream) {
    checkCuFFTCall(cufftSetStream(plan_, stream));
  }

  void execute(cu::DeviceMemory &in, cu::DeviceMemory &out,
               int direction = CUFFT_FORWARD) {
    void *in_ptr = reinterpret_cast<void *>(static_cast<CUdeviceptr>(in));
    void *out_ptr = reinterpret_cast<void *>(static_cast<CUdeviceptr>(out));
    checkCuFFTCall(cufftXtExec(plan_, in_ptr, out_ptr, direction));
  }

  void execute(CUdeviceptr in, CUdeviceptr out, int direction) {
    void *in_ptr = reinterpret_cast<void *>(in);
    void *out_ptr = reinterpret_cast<void *>(out);
    checkCuFFTCall(cufftXtExec(plan_, in_ptr, out_ptr, direction));
  }

 protected:
  void checkCuFFTCall(cufftResult result) {
    if (result != CUFFT_SUCCESS) {
      throw Error(result);
    }
  }

  cufftHandle plan_;
};

/*
 * FFT1D
 */
template <cudaDataType_t T>
class FFT1D : public FFT {
 public:
  FFT1D(int nx) = delete;
  FFT1D(int nx, int batch) = delete;
};

template <>
FFT1D<CUDA_C_32F>::FFT1D(int nx, int batch) {
  checkCuFFTCall(cufftCreate(&plan_));
  checkCuFFTCall(cufftPlan1d(&plan_, nx, CUFFT_C2C, batch));
}

template <>
FFT1D<CUDA_C_32F>::FFT1D(int nx) : FFT1D(nx, 1) {}

template <>
FFT1D<CUDA_C_16F>::FFT1D(int nx, int batch) {
  checkCuFFTCall(cufftCreate(&plan_));
  const int rank = 1;
  size_t ws = 0;
  long long n[rank] = {nx};
  long long int idist = 1;
  long long int odist = 1;
  int istride = 1;
  int ostride = 1;
  checkCuFFTCall(cufftXtMakePlanMany(plan_, rank, n, NULL, istride, idist,
                                     CUDA_C_16F, NULL, ostride, odist,
                                     CUDA_C_16F, batch, &ws, CUDA_C_16F));
}

template <>
FFT1D<CUDA_C_16F>::FFT1D(int nx) : FFT1D(nx, 1) {}

/*
 * FFT2D
 */
template <cudaDataType_t T>
class FFT2D : public FFT {
 public:
  FFT2D(int nx, int ny) = delete;
  FFT2D(int nx, int ny, int stride, int dist, int batch) = delete;
};

template <>
FFT2D<CUDA_C_32F>::FFT2D(int nx, int ny) {
  checkCuFFTCall(cufftCreate(&plan_));
  checkCuFFTCall(cufftPlan2d(&plan_, nx, ny, CUFFT_C2C));
}

template <>
FFT2D<CUDA_C_32F>::FFT2D(int nx, int ny, int stride, int dist, int batch) {
  checkCuFFTCall(cufftCreate(&plan_));
  int n[2]{nx, ny};
  checkCuFFTCall(cufftPlanMany(&plan_, 2, n, n, stride, dist, n, stride, dist,
                               CUFFT_C2C, batch));
}

template <>
FFT2D<CUDA_C_16F>::FFT2D(int nx, int ny, int stride, int dist, int batch) {
  checkCuFFTCall(cufftCreate(&plan_));
  const int rank = 2;
  size_t ws = 0;
  long long n[rank] = {nx, ny};
  long long int idist = 1;
  long long int odist = 1;
  int istride = 1;
  int ostride = 1;
  checkCuFFTCall(cufftXtMakePlanMany(plan_, rank, n, NULL, istride, idist,
                                     CUDA_C_16F, NULL, ostride, odist,
                                     CUDA_C_16F, batch, &ws, CUDA_C_16F));
}

template <>
FFT2D<CUDA_C_16F>::FFT2D(int nx, int ny) : FFT2D(nx, ny, 1, nx * ny, 1) {}

}  // namespace cufft

#endif  // CUFFT_WRAPPER_H
