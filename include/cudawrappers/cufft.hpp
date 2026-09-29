#if !defined CUFFT_H
#define CUFFT_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <magic_enum/magic_enum.hpp>

#include "cudawrappers/cu.hpp"
#include "cudawrappers/cufft_backend.hpp"

/*
 * A self-contained wrapper whose API mirrors the cuFFT API as closely as
 * possible.  The user writes plain cuFFT-style code (cufft::FFT1D,
 * cufft::CUFFT_C2C, ...); under the hood every call is dispatched through a
 * per-backend function-pointer table so that the same code runs transparently
 * on an NVIDIA GPU (cuFFT) and on an AMD GPU (hipFFT).
 */

namespace cufft {

// The cuFFT plan handle is an int (a small id); the hipFFT plan handle is a
// pointer.  Store the handle pointer-sized so one ABI serves both libraries
// (cuFFT reads the low 32 bits of the small id, hipFFT uses the full pointer).
typedef uintptr_t cufftHandle;

enum cufftResult {
  CUFFT_SUCCESS = 0x0,
  CUFFT_INVALID_PLAN = 0x1,
  CUFFT_ALLOC_FAILED = 0x2,
  CUFFT_INVALID_TYPE = 0x3,
  CUFFT_INVALID_VALUE = 0x4,
  CUFFT_INTERNAL_ERROR = 0x5,
  CUFFT_EXEC_FAILED = 0x6,
  CUFFT_SETUP_FAILED = 0x7,
  CUFFT_INVALID_SIZE = 0x8,
  CUFFT_UNALIGNED_DATA = 0x9,
  CUFFT_INVALID_DEVICE = 0xB,
  CUFFT_NO_WORKSPACE = 0xD,
  CUFFT_NOT_IMPLEMENTED = 0xE,
  CUFFT_NOT_SUPPORTED = 0x10,
  CUFFT_MISSING_DEPENDENCY = 0x11,
  CUFFT_NVRTC_FAILURE = 0x12,
  CUFFT_NVJITLINK_FAILURE = 0x13,
  CUFFT_NVSHMEM_FAILURE = 0x14
};

enum cufftType {
  CUFFT_R2C = 0x2a,
  CUFFT_C2R = 0x2c,
  CUFFT_C2C = 0x29,
  CUFFT_D2Z = 0x6a,
  CUFFT_Z2Z = 0x69
};

enum cudaDataType {
  CUDA_R_16F = 2,
  CUDA_C_16F = 6,
  CUDA_R_16BF = 14,
  CUDA_C_16BF = 15,
  CUDA_R_32F = 0,
  CUDA_C_32F = 4,
  CUDA_R_64F = 1,
  CUDA_C_64F = 5,
  CUDA_R_4I = 16,
  CUDA_C_4I = 17,
  CUDA_R_4U = 18,
  CUDA_C_4U = 19,
  CUDA_R_8I = 3,
  CUDA_C_8I = 7,
  CUDA_R_8U = 8,
  CUDA_C_8U = 9,
  CUDA_R_16I = 20,
  CUDA_C_16I = 21,
  CUDA_R_16U = 22,
  CUDA_C_16U = 23,
  CUDA_R_32I = 10,
  CUDA_C_32I = 11,
  CUDA_R_32U = 12,
  CUDA_C_32U = 13
};

constexpr int CUFFT_FORWARD = -1;
constexpr int CUFFT_INVERSE = 1;

enum cufftCompatibility {
  CUFFT_COMPATIBILITY_NATIVE = 0x0,
  CUFFT_COMPATIBILITY_FFTW_PADDING = 0x1,
  CUFFT_COMPATIBILITY_FFTW_SHIFT = 0x2,
  CUFFT_COMPATIBILITY_DUPLICATE_HANDLES = 0x4
};

enum libraryPropertyType {
  CUFFT_MAJOR_VERSION = 0x0,
  CUFFT_MINOR_VERSION = 0x1,
  CUFFT_PATCH_LEVEL = 0x2
};

enum cufftCallbackType {
  CUFFT_CB_LOAD_IN = 0,
  CUFFT_CB_STORE_OUT = 1,
  CUFFT_CB_LD_STORE_IN = 2,
  CUFFT_CB_LD_STORE_OUT = 3
};

enum cufftXtWorkAreaPolicy {
  CUFFT_WORKAREA_MINIMAL = 0,
  CUFFT_WORKAREA_USER = 1,
  CUFFT_WORKAREA_PERFORMANCE = 2
};

enum cufftXtQueryType {
  CUFFT_QUERY_1D_FACTORS = 0x00,
  CUFFT_QUERY_UNDEFINED = 0x01
};

enum cufftProperty {
  NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT = 0x1,
  NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS = 0x2
};

struct cufftComplex {
  float x, y;
};
struct cufftDoubleComplex {
  double x, y;
};
typedef float cufftReal;
typedef double cufftDoubleReal;

/*
 * Error
 */
class Error : public std::exception {
 public:
  explicit Error(cufftResult result) : result_(result) {}

  const char* what() const noexcept override {
    message_ = std::string(magic_enum::enum_name(result_));
    return message_.c_str();
  }

  operator cufftResult() const { return result_; }

 private:
  cufftResult result_;
  mutable std::string message_ = "";
};

/*
 * FFT
 *
 * Holds one plan and wraps the cuFFT functions for resource management and
 * error detection.  The backend (cuFFT vs hipFFT) is resolved per object from
 * the backend that was active when the plan was created, so plans for NVIDIA
 * and AMD devices can coexist in one process.
 */
class FFT {
 public:
  FFT() = default;
  FFT(const FFT&) = delete;
  FFT& operator=(const FFT&) = delete;
  FFT(FFT&& other) noexcept : _backendIdx(other._backendIdx), plan_(other.plan_) {
    other.plan_ = 0;
  }
  FFT& operator=(FFT&& other) noexcept {
    if (&other != this) {
      plan_ = other.plan_;
      other.plan_ = 0;
      _backendIdx = other._backendIdx;
    }
    return *this;
  }

  ~FFT() {
    if (plan_ != 0) {
      checkCuFFTCall(backend().destroy(plan_));
    }
  }

  void setStream(cu::Stream& stream) const {
    checkCuFFTCall(backend().setStream(plan_, stream));
  }

  size_t getSize() const {
    size_t ws{};
    checkCuFFTCall(backend().getSize(plan_, &ws));
    return ws;
  }

  size_t getSize1d(int nx, int batch) const {
    size_t ws{};
    checkCuFFTCall(backend().getSize1d(plan_, nx, batch, &ws));
    return ws;
  }

  size_t getSizeMany(int rank, std::array<int, 3> n, std::array<int, 3> inembed,
                     int istride, int idist, std::array<int, 3> onembed,
                     int ostride, int odist, cufftType type, int batch) const {
    size_t ws{};
    checkCuFFTCall(backend().getSizeMany(plan_, rank, n.data(), inembed.data(),
                                        istride, idist, onembed.data(), ostride,
                                        odist, type, batch, &ws));
    return ws;
  }

  size_t getSizeMany64(int rank, std::array<long long, 3> n,
                       std::array<long long, 3> inembed, long long istride,
                       long long idist, std::array<long long, 3> onembed,
                       long long ostride, long long odist, cufftType type,
                       long long batch) const {
    size_t ws{};
    checkCuFFTCall(backend().getSizeMany64(plan_, rank, n.data(), inembed.data(),
                                          istride, idist, onembed.data(),
                                          ostride, odist, type, batch, &ws));
    return ws;
  }

  void setWorkArea(void* workArea) const {
    checkCuFFTCall(backend().setWorkArea(plan_, workArea));
  }

  void setAutoAllocation(int autoAllocate) const {
    checkCuFFTCall(backend().setAutoAllocation(plan_, autoAllocate));
  }

  void setCompatibilityMode(cufftCompatibility mode) const {
    checkCuFFTCall(backend().setCompatibilityMode(plan_, mode));
  }

  void setWorkAreaPolicy(cufftXtWorkAreaPolicy policy, size_t* workSize) const {
    checkCuFFTCall(backend().xtSetWorkAreaPolicy(plan_, policy, workSize));
  }

  void setPlanPropertyInt64(cufftProperty property, long long value) const {
    ensurePlan();
    checkCuFFTCall(backend().setPlanPropertyInt64(plan_, property, value));
  }

  long long getPlanPropertyInt64(cufftProperty property) const {
    ensurePlan();
    long long value{};
    checkCuFFTCall(backend().getPlanPropertyInt64(plan_, property, &value));
    return value;
  }

  void resetPlanProperty(cufftProperty property) const {
    ensurePlan();
    checkCuFFTCall(backend().resetPlanProperty(plan_, property));
  }

  void xtQueryPlan(void* queryStruct, cufftXtQueryType queryType) const {
    checkCuFFTCall(backend().xtQueryPlan(plan_, queryStruct, queryType));
  }

  void setJITCallback(const char* symbol, const void* fatbin, size_t size,
                      cufftCallbackType type, void** callerInfo) const {
    checkCuFFTCall(backend().xtSetJITCallback(plan_, symbol, fatbin, size, type,
                                              callerInfo));
  }

  void execute(cu::DeviceMemory& in, cu::DeviceMemory& out,
               const int direction) const {
    void* in_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(in));
    void* out_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(out));
    checkCuFFTCall(backend().xtExec(plan_, in_ptr, out_ptr, direction));
  }

  void execC2C(cu::DeviceMemory& in, cu::DeviceMemory& out,
               const int direction) const {
    void* in_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(in));
    void* out_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(out));
    checkCuFFTCall(backend().execC2C(plan_, in_ptr, out_ptr, direction));
  }

  void execR2C(cu::DeviceMemory& in, cu::DeviceMemory& out) const {
    void* in_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(in));
    void* out_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(out));
    checkCuFFTCall(backend().execR2C(plan_, in_ptr, out_ptr, CUFFT_FORWARD));
  }

  void execC2R(cu::DeviceMemory& in, cu::DeviceMemory& out) const {
    void* in_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(in));
    void* out_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(out));
    checkCuFFTCall(backend().execC2R(plan_, in_ptr, out_ptr, CUFFT_INVERSE));
  }

  void execZ2Z(cu::DeviceMemory& in, cu::DeviceMemory& out,
               const int direction) const {
    void* in_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(in));
    void* out_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(out));
    checkCuFFTCall(backend().execZ2Z(plan_, in_ptr, out_ptr, direction));
  }

  void execD2Z(cu::DeviceMemory& in, cu::DeviceMemory& out) const {
    void* in_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(in));
    void* out_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(out));
    checkCuFFTCall(backend().execD2Z(plan_, in_ptr, out_ptr, CUFFT_FORWARD));
  }

  void execZ2D(cu::DeviceMemory& in, cu::DeviceMemory& out) const {
    void* in_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(in));
    void* out_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(out));
    checkCuFFTCall(backend().execZ2D(plan_, in_ptr, out_ptr, CUFFT_INVERSE));
  }

  void execC2C64(cu::DeviceMemory& in, cu::DeviceMemory& out,
                 const int direction) const {
    void* in_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(in));
    void* out_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(out));
    checkCuFFTCall(backend().execC2C64(plan_, in_ptr, out_ptr, direction));
  }

  void execR2C64(cu::DeviceMemory& in, cu::DeviceMemory& out) const {
    void* in_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(in));
    void* out_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(out));
    checkCuFFTCall(backend().execR2C64(plan_, in_ptr, out_ptr, CUFFT_FORWARD));
  }

  void execC2R64(cu::DeviceMemory& in, cu::DeviceMemory& out) const {
    void* in_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(in));
    void* out_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(out));
    checkCuFFTCall(backend().execC2R64(plan_, in_ptr, out_ptr, CUFFT_INVERSE));
  }

  void execZ2Z64(cu::DeviceMemory& in, cu::DeviceMemory& out,
                 const int direction) const {
    void* in_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(in));
    void* out_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(out));
    checkCuFFTCall(backend().execZ2Z64(plan_, in_ptr, out_ptr, direction));
  }

  void execD2Z64(cu::DeviceMemory& in, cu::DeviceMemory& out) const {
    void* in_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(in));
    void* out_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(out));
    checkCuFFTCall(backend().execD2Z64(plan_, in_ptr, out_ptr, CUFFT_FORWARD));
  }

  void execZ2D64(cu::DeviceMemory& in, cu::DeviceMemory& out) const {
    void* in_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(in));
    void* out_ptr = reinterpret_cast<void*>(static_cast<CUdeviceptr>(out));
    checkCuFFTCall(backend().execZ2D64(plan_, in_ptr, out_ptr, CUFFT_INVERSE));
  }

  void setGPUs(const int nGPUs, int* whichGPUs) const {
    checkCuFFTCall(backend().xtSetGPUs(plan_, nGPUs, whichGPUs));
  }

  void setCallback(void** callbacks, cufftCallbackType cbType,
                   void** userData) const {
    checkCuFFTCall(backend().xtSetCallback(plan_, callbacks, cbType, userData));
  }

  void setCallbackSharedSize(cufftCallbackType cbType, size_t sharedSize) const {
    ensurePlan();
    checkCuFFTCall(
        backend().xtSetCallbackSharedSize(plan_, cbType, sharedSize));
  }

  void clearCallback(cufftCallbackType cbType) const {
    checkCuFFTCall(backend().xtClearCallback(plan_, cbType));
  }

 protected:
  void checkCuFFTCall(int result) const {
    if (result != CUFFT_SUCCESS) {
      throw Error(static_cast<cufftResult>(result));
    }
  }

  void ensurePlan() const {
    if (plan_ == 0) {
      checkCuFFTCall(backend().create(&plan_));
    }
  }

  FFTBackend& backend() const { return getFFTBackend(_backendIdx); }

  cufftHandle* plan() { return &plan_; }

  int _backendIdx{cu::activeBackendIdx()};
  mutable cufftHandle plan_{};
};

/*
 * FFT1D
 */
template <cudaDataType T>
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
inline FFT1D<CUDA_C_32F>::FFT1D(const int nx, const int batch) {
  checkCuFFTCall((backend().create(plan())));
  checkCuFFTCall((backend().plan1d(plan(), nx, CUFFT_C2C, batch)));
}

template <>
inline FFT1D<CUDA_C_32F>::FFT1D(const int nx) : FFT1D(nx, 1) {}

template <>
inline FFT1D<CUDA_C_16F>::FFT1D(const int nx, const int batch) {
  checkCuFFTCall((backend().create(plan())));
  const int rank = 1;
  size_t ws = 0;
  std::array<long long, 1> n{nx};
  const long long idist = 1;
  const long long odist = 1;
  const long long istride = 1;
  const long long ostride = 1;
  checkCuFFTCall((backend().xtMakePlanMany(
      *plan(), rank, n.data(), nullptr, istride, idist, CUDA_C_16F, nullptr,
      ostride, odist, CUDA_C_16F, batch, &ws, CUDA_C_16F)));
}

template <>
inline FFT1D<CUDA_C_16F>::FFT1D(const int nx) : FFT1D(nx, 1) {}

/*
 * FFT2D
 */
template <cudaDataType T>
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
inline FFT2D<CUDA_C_32F>::FFT2D(const int nx, const int ny) {
  checkCuFFTCall((backend().create(plan())));
  checkCuFFTCall((backend().plan2d(plan(), nx, ny, CUFFT_C2C)));
}

template <>
inline FFT2D<CUDA_C_32F>::FFT2D(const int nx, const int ny, const int stride,
                                const int dist, const int batch) {
  checkCuFFTCall((backend().create(plan())));
  std::array<int, 2> n{nx, ny};
  checkCuFFTCall((backend().planMany(
      plan(), 2, n.data(), n.data(), stride, dist, n.data(), stride, dist,
      CUFFT_C2C, batch)));
}

template <>
inline FFT2D<CUDA_C_16F>::FFT2D(const int nx, const int ny, const int stride,
                                const int dist, const int batch) {
  checkCuFFTCall((backend().create(plan())));
  const int rank = 2;
  size_t ws = 0;
  std::array<long long, 2> n{nx, ny};
  const long long istride = stride;
  const long long ostride = stride;
  const long long idist = dist;
  const long long odist = dist;
  checkCuFFTCall((backend().xtMakePlanMany(
      *plan(), rank, n.data(), nullptr, istride, idist, CUDA_C_16F, nullptr,
      ostride, odist, CUDA_C_16F, batch, &ws, CUDA_C_16F)));
}

template <>
inline FFT2D<CUDA_C_16F>::FFT2D(const int nx, const int ny)
    : FFT2D(nx, ny, 1, nx * ny, 1) {}

/*
 * FFT1DR2C
 */
template <cudaDataType T>
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
inline FFT1DR2C<CUDA_R_32F>::FFT1DR2C(const int nx, const int batch,
                                      long long inembed, long long ouembed) {
  checkCuFFTCall((backend().create(plan())));
  const int rank = 1;
  size_t ws = 0;
  std::array<long long, 1> n{nx};
  const long long idist = inembed;
  const long long odist = ouembed;
  const long long istride = 1;
  const long long ostride = 1;

  checkCuFFTCall((backend().xtMakePlanMany(
      *plan(), rank, n.data(), &inembed, istride, idist, CUDA_R_32F, &ouembed,
      ostride, odist, CUDA_C_32F, batch, &ws, CUDA_C_32F)));
}

/*
 * FFT1D_C2R
 */
template <cudaDataType T>
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
inline FFT1DC2R<CUDA_C_32F>::FFT1DC2R(const int nx, const int batch,
                                      long long inembed, long long ouembed) {
  checkCuFFTCall((backend().create(plan())));
  const int rank = 1;
  size_t ws = 0;
  std::array<long long, 1> n{nx};
  const long long idist = inembed;
  const long long odist = ouembed;
  const long long istride = 1;
  const long long ostride = 1;

  checkCuFFTCall((backend().xtMakePlanMany(
      *plan(), rank, n.data(), &inembed, istride, idist, CUDA_C_32F, &ouembed,
      ostride, odist, CUDA_R_32F, batch, &ws, CUDA_C_32F)));
}

/*
 * FFT3D
 */
template <cudaDataType T>
class FFT3D : public FFT {
 public:
#if defined(__HIP__)
  __host__
#endif
  FFT3D(const int nx, const int ny, const int nz) = delete;
};

template <>
inline FFT3D<CUDA_C_32F>::FFT3D(const int nx, const int ny, const int nz) {
  checkCuFFTCall((backend().create(plan())));
  size_t ws = 0;
  checkCuFFTCall(
      (backend().makePlan3d(plan_, nx, ny, nz, CUFFT_C2C, &ws)));
}

/*
 * Work-size estimation (no plan involved).
 */
inline size_t estimate1d(int nx, cufftType type, int batch) {
  size_t ws{};
  int result = getFFTBackend(cu::activeBackendIdx()).estimate1d(nx, type, batch, &ws);
  if (result != CUFFT_SUCCESS) {
    throw Error(static_cast<cufftResult>(result));
  }
  return ws;
}

inline size_t estimate2d(int nx, int ny, cufftType type) {
  size_t ws{};
  int result = getFFTBackend(cu::activeBackendIdx()).estimate2d(nx, ny, type, &ws);
  if (result != CUFFT_SUCCESS) {
    throw Error(static_cast<cufftResult>(result));
  }
  return ws;
}

inline size_t estimate3d(int nx, int ny, int nz, cufftType type) {
  size_t ws{};
  int result =
      getFFTBackend(cu::activeBackendIdx()).estimate3d(nx, ny, nz, type, &ws);
  if (result != CUFFT_SUCCESS) {
    throw Error(static_cast<cufftResult>(result));
  }
  return ws;
}

inline size_t estimateMany(int rank, int* n, int* inembed, int istride,
                          int idist, int* onembed, int ostride, int odist,
                          cufftType type, int batch) {
  size_t ws{};
  int result = getFFTBackend(cu::activeBackendIdx())
                  .estimateMany(rank, n, inembed, istride, idist, onembed,
                                ostride, odist, type, batch, &ws);
  if (result != CUFFT_SUCCESS) {
    throw Error(static_cast<cufftResult>(result));
  }
  return ws;
}

/*
 * Version and library properties (no plan involved).
 */
inline int getVersion() {
  int version{};
  int result = getFFTBackend(cu::activeBackendIdx()).getVersion(&version);
  if (result != CUFFT_SUCCESS) {
    throw Error(static_cast<cufftResult>(result));
  }
  return version;
}

inline int getProperty(libraryPropertyType type) {
  int value{};
  int result = getFFTBackend(cu::activeBackendIdx()).getProperty(type, &value);
  if (result != CUFFT_SUCCESS) {
    throw Error(static_cast<cufftResult>(result));
  }
  return value;
}

}  // namespace cufft

#endif
