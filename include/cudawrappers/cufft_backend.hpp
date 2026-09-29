#if !defined CUFFT_BACKEND_H
#define CUFFT_BACKEND_H

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "cudawrappers/cu_backend.hpp"

// The cuFFT (cufftHandle) and hipFFT (hipfftHandle) plan handles differ: cuFFT
// uses a small integer id, hipFFT (on AMD) a pointer to the plan structure.
// A single pointer-sized handle type below serves both ABIs so one
// function-pointer table, loaded by dlsym, can drive either library: cuFFT
// reads the low 32 bits of the id, hipFFT uses the full pointer, and the
// stream and buffer pointers are plain pointers on both.  The enumerators
// (cufftType, cudaDataType, direction, ...) share identical values on both,
// so a plan created and executed through this table works on an NVIDIA GPU
// (cuFFT) and transparently on an AMD GPU (hipFFT).
//
// Every function that is absent from a given library gets a stub returning
// CUFFT_NOT_SUPPORTED (0x10) so that calling it throws a proper cufft::Error
// instead of crashing through a null pointer.  Only the library itself being
// unloadable is a hard error, raised as soon as the backend table is built.

typedef uintptr_t cufftHandle_b;

// Every entry point of the wrapper.  Each entry is (field name, symbol suffix,
// signature); the symbol looked up in the library is "cufft"+suffix or
// "hipfft"+suffix.
#define CUFFT_BACKEND_SYMBOLS(X)                                               \
  X(create, Create, cufftHandle_b*)                                             \
  X(destroy, Destroy, cufftHandle_b)                                            \
  X(plan1d, Plan1d, cufftHandle_b*, int, int, int)                              \
  X(plan2d, Plan2d, cufftHandle_b*, int, int, int)                              \
  X(plan3d, Plan3d, cufftHandle_b*, int, int, int, int)                         \
  X(planMany, PlanMany, cufftHandle_b*, int, int*, int*, int, int, int*, int,   \
    int, int, int)                                                             \
  X(makePlan1d, MakePlan1d, cufftHandle_b, int, int, int, size_t*)              \
  X(makePlan2d, MakePlan2d, cufftHandle_b, int, int, int, size_t*)              \
  X(makePlan3d, MakePlan3d, cufftHandle_b, int, int, int, int, size_t*)         \
  X(makePlanMany, MakePlanMany, cufftHandle_b, int, int*, int*, int, int,       \
    int*, int, int, int, int, size_t*)                                          \
  X(makePlanMany64, MakePlanMany64, cufftHandle_b, int, long long*,            \
    long long*, long long, long long, long long*, long long, long long, int,    \
    long long, size_t*)                                                         \
  X(xtMakePlanMany, XtMakePlanMany, cufftHandle_b, int, long long*,            \
    long long*, long long, long long, int, long long*, long long, long long,    \
    int, long long, size_t*, int)                                               \
  X(estimate1d, Estimate1d, int, int, int, size_t*)                             \
  X(estimate2d, Estimate2d, int, int, int, size_t*)                             \
  X(estimate3d, Estimate3d, int, int, int, int, size_t*)                        \
  X(estimateMany, EstimateMany, int, int*, int*, int, int, int*, int, int,      \
    int, int, size_t*)                                                          \
  X(getSize, GetSize, cufftHandle_b, size_t*)                                    \
  X(getSize1d, GetSize1d, cufftHandle_b, int, int, size_t*)                      \
  X(getSizeMany, GetSizeMany, cufftHandle_b, int, int*, int*, int, int, int*,   \
    int, int, int, int, size_t*)                                                \
  X(getSizeMany64, GetSizeMany64, cufftHandle_b, int, long long*, long long*,   \
    long long, long long, long long*, long long, long long, int, long long,     \
    size_t*)                                                                    \
  X(setStream, SetStream, cufftHandle_b, void*)                                  \
  X(setWorkArea, SetWorkArea, cufftHandle_b, void*)                             \
  X(setAutoAllocation, SetAutoAllocation, cufftHandle_b, int)                   \
  X(setCompatibilityMode, SetCompatibilityMode, cufftHandle_b, int)              \
  X(getVersion, GetVersion, int*)                                                \
  X(getProperty, GetProperty, int, int*)                                         \
  X(execC2C, ExecC2C, cufftHandle_b, void*, void*, int)                         \
  X(execR2C, ExecR2C, cufftHandle_b, void*, void*, int)                         \
  X(execC2R, ExecC2R, cufftHandle_b, void*, void*, int)                         \
  X(execZ2Z, ExecZ2Z, cufftHandle_b, void*, void*, int)                         \
  X(execD2Z, ExecD2Z, cufftHandle_b, void*, void*, int)                         \
  X(execZ2D, ExecZ2D, cufftHandle_b, void*, void*, int)                         \
  X(execC2C64, ExecC2C64, cufftHandle_b, void*, void*, int)                     \
  X(execR2C64, ExecR2C64, cufftHandle_b, void*, void*, int)                     \
  X(execC2R64, ExecC2R64, cufftHandle_b, void*, void*, int)                     \
  X(execZ2Z64, ExecZ2Z64, cufftHandle_b, void*, void*, int)                     \
  X(execD2Z64, ExecD2Z64, cufftHandle_b, void*, void*, int)                     \
  X(execZ2D64, ExecZ2D64, cufftHandle_b, void*, void*, int)                     \
  X(xtExec, XtExec, cufftHandle_b, void*, void*, int)                           \
  X(xtSetGPUs, XtSetGPUs, cufftHandle_b, int, int*)                             \
  X(xtSetCallback, XtSetCallback, cufftHandle_b, void**, int, void**)           \
  X(xtSetCallbackSharedSize, XtSetCallbackSharedSize, cufftHandle_b, int,      \
    size_t)                                                                     \
  X(xtClearCallback, XtClearCallback, cufftHandle_b, int)                       \
  X(xtSetJITCallback, XtSetJITCallback, cufftHandle_b, const char*,            \
    const void*, size_t, int, void**)                                           \
  X(xtSetWorkAreaPolicy, XtSetWorkAreaPolicy, cufftHandle_b, int, size_t*)      \
  X(xtQueryPlan, XtQueryPlan, cufftHandle_b, void*, int)                        \
  X(setPlanPropertyInt64, SetPlanPropertyInt64, cufftHandle_b, int, long long)   \
  X(getPlanPropertyInt64, GetPlanPropertyInt64, cufftHandle_b, int, long long*)  \
  X(resetPlanProperty, ResetPlanProperty, cufftHandle_b, int)

struct FFTBackend {
  void* lib;
  int is_cuda;

#define X(name, sym, ...) int (*name)(__VA_ARGS__);
  CUFFT_BACKEND_SYMBOLS(X)
#undef X
};

namespace {

// Missing optional functions become a stub that always yields
// CUFFT_NOT_SUPPORTED (0x10).
template <typename R, typename... Args>
R unsupportedCall(Args...) {
  return static_cast<R>(0x10);
}

template <typename R, typename... Args>
R (*unsupportedStub(R (*)(Args...)))(Args...) {
  return &unsupportedCall<R, Args...>;
}

}  // namespace

// --- Backend loaders (header-only) ---

inline FFTBackend loadCudaFFTBackend() {
  FFTBackend b{};
  b.lib = dlopen("libcufft.so.12", RTLD_LAZY | RTLD_GLOBAL);
  if (!b.lib) {
    b.lib = dlopen("libcufft.so", RTLD_LAZY | RTLD_GLOBAL);
  }
  if (!b.lib) {
    return b;
  }
  b.is_cuda = 1;

#define LOAD(name, sym, ...) \
  b.name = reinterpret_cast<decltype(FFTBackend::name)>(dlsym(b.lib, "cufft" #sym));
  CUFFT_BACKEND_SYMBOLS(LOAD)
#undef LOAD

#define NORMALIZE(name, sym, ...) \
  if (!b.name) b.name = unsupportedStub(b.name);
  CUFFT_BACKEND_SYMBOLS(NORMALIZE)
#undef NORMALIZE

  return b;
}

inline FFTBackend loadHipFFTBackend() {
  FFTBackend b{};
  // hipFFT registers device code in its constructor and needs a current device
  // for that; without one it aborts during dlopen on some ROCm versions.
  void* hipLib = getFlavorBackend(false).lib;
  using SetDeviceFn = int (*)(int);
  static SetDeviceFn setDevice =
      hipLib ? reinterpret_cast<SetDeviceFn>(dlsym(hipLib, "hipSetDevice")) : nullptr;
  if (setDevice) {
    setDevice(0);
  }
  b.lib = dlopen("libhipfft.so", RTLD_LAZY | RTLD_GLOBAL);
  if (!b.lib) {
    b.lib = dlopen("libhipfft.so.0", RTLD_LAZY | RTLD_GLOBAL);
  }
  if (!b.lib) {
    return b;
  }
  b.is_cuda = 0;

#define LOAD(name, sym, ...) \
  b.name = reinterpret_cast<decltype(FFTBackend::name)>(dlsym(b.lib, "hipfft" #sym));
  CUFFT_BACKEND_SYMBOLS(LOAD)
#undef LOAD

  // cuFFT-only entry points that hipFFT does not implement become
  // CUFFT_NOT_SUPPORTED stubs (the LOAD above already left them null).
#define NORMALIZE(name, sym, ...) \
  if (!b.name) b.name = unsupportedStub(b.name);
  CUFFT_BACKEND_SYMBOLS(NORMALIZE)
#undef NORMALIZE

  return b;
}

// --- Backend management (aligned with the driver backends) ---
//
// The FFT library of a flavor is loaded lazily, the first time a plan is created
// on that flavor.  At that moment a cu::Context for the flavor already exists,
// so the vendor library's constructor (which in ROCm registers device code and
// needs a current, usable device) can initialize safely.  Probing or eagerly
// loading every flavor up front would, on a host whose other vendor's devices
// are busy or inaccessible, hit that constructor from a state with no usable
// device and raise an exception inside dlopen that a C++ try/catch cannot
// catch.

inline std::vector<bool>& fftBackendLoadedState();

inline std::vector<FFTBackend>& getFFTBackends() {
  static std::vector<FFTBackend> backends;
  static bool inited = false;
  if (!inited) {
    inited = true;
    const std::vector<Backend>& drivers = getBackends();
    backends.resize(drivers.size());
    fftBackendLoadedState().assign(drivers.size(), false);
    for (size_t i = 0; i < drivers.size(); ++i) {
      backends[i].is_cuda = drivers[i].is_cuda;
      // Not yet loaded: stub every entry point so any accidental call yields
      // CUFFT_NOT_SUPPORTED instead of a crash.
#define STUB(name, sym, ...) backends[i].name = unsupportedStub(backends[i].name);
      CUFFT_BACKEND_SYMBOLS(STUB)
#undef STUB
    }
  }
  return backends;
}

inline size_t getFFTBackendCount() { return getFFTBackends().size(); }

inline bool getFFTBackendUsable(int idx) { return getFFTBackends().at(idx).lib != nullptr; }

// Whether each driver-backend's FFT library has been loaded yet.  Kept as a
// separate static so getFFTBackends() can stay a plain static vector.
inline std::vector<bool>& fftBackendLoadedState() {
  static std::vector<bool> loaded;
  return loaded;
}

inline FFTBackend& getFFTBackend(int idx) {
  std::vector<FFTBackend>& backends = getFFTBackends();
  std::vector<bool>& loaded = fftBackendLoadedState();
  if (!loaded.at(idx)) {
    const std::vector<Backend>& drivers = getBackends();
    loaded.at(idx) = true;
    FFTBackend b;
    bool ok = false;
    try {
      b = drivers.at(idx).is_cuda ? loadCudaFFTBackend() : loadHipFFTBackend();
      ok = b.lib != nullptr;
    } catch (...) {
      ok = false;
    }
    if (!ok) {
      // A present driver flavor whose FFT library cannot be loaded can never
      // create a plan: fail as soon as possible.
      throw std::runtime_error(
          "cudawrappers: " +
          std::string(drivers.at(idx).is_cuda ? "cuFFT" : "hipFFT") +
          " library could not be loaded (dlopen failed)");
    }
    backends.at(idx) = b;
  }
  return backends.at(idx);
}

#endif
