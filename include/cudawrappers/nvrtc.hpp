#if !defined NVRTC_H
#define NVRTC_H
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(__HIP__)
#if __has_include(<cuda.h>)
#include <cuda.h>
#endif
#if __has_include(<nvrtc.h>)
#include <nvrtc.h>
#endif
#endif

#include <cudawrappers/config.h>

#include <cudawrappers/cu.hpp>

namespace nvrtc {
namespace detail {

inline std::vector<std::string> tokenize(const std::string &input,
                                         const std::string &delimiter) {
  std::string s = input;
  size_t pos = 0;
  std::vector<std::string> tokens;
  while ((pos = s.find(delimiter)) != std::string::npos) {
    tokens.push_back(s.substr(0, pos));
    s.erase(0, pos + delimiter.length());
    pos = s.find(delimiter);
  }
  tokens.push_back(s);
  return tokens;
}

// Raw device architecture name as known to the backend: "sm_<cc>" for NVIDIA
// (e.g. "sm_120") or the gcnArchName for AMD (e.g. "gfx1201").  This is not a
// wrapped CUDA Driver API function (the CUDA flavor synthesizes it from the
// compute-capability attributes, the HIP flavor reads hipGetDeviceProperties),
// so it lives in detail instead of on cu::Device.
inline std::string archName(const cu::Device &device) {
  const size_t max_arch_length{64};
  std::array<char, max_arch_length> arch{};
  int r = getBackend(device.getBackendIdx())
              .deviceGetArchName(arch.data(), static_cast<int>(arch.size()),
                                 static_cast<CUdevice>(device));
  cu::checkCudaCall(r);
  return {arch.data()};
}

// Load the NVRTC builtins library, required by libnvrtc on some systems.
inline void loadNvrtcBuiltins() {
  if (!dlopen("libnvrtc-builtins.so", RTLD_LAZY) &&
      !dlopen("libnvrtc-builtins.so.13", RTLD_LAZY) &&
      !dlopen("libnvrtc-builtins.so.12", RTLD_LAZY)) {
    throw std::runtime_error("Failed to load libnvrtc-builtins.so");
  }
}

// Runtime-selected compiler API: either NVIDIA NVRTC or AMD HIPRTC. Both
// libraries expose identical entry-point shapes, so one set of function
// pointers serves both backends.
struct RtcApi {
  void *lib{nullptr};
  bool isHip{false};

  int (*createProgram)(void **, const char *, const char *, int,
                       const char *const *, const char *const *){nullptr};
  int (*destroyProgram)(void **){nullptr};
  int (*compileProgram)(void *, int, const char *const *){nullptr};
  int (*getCodeSize)(const void *, size_t *){nullptr};
  int (*getCode)(const void *, char *){nullptr};
  int (*getBinarySize)(const void *, size_t *){nullptr};
  int (*getBinary)(const void *, char *){nullptr};
  int (*getLogSize)(const void *, size_t *){nullptr};
  int (*getLog)(const void *, char *){nullptr};
  int (*addNameExpression)(void *, const char *){nullptr};
  int (*getLoweredName)(const void *, const char *, const char **){nullptr};
  const char *(*getErrorString)(int){nullptr};
  int (*version)(int *, int *){nullptr};
  int (*getNumSupportedArchs)(int *){nullptr};
  int (*getSupportedArchs)(int *){nullptr};

  static const RtcApi &get(bool isHip);
};

template <typename T>
T dlsymOrThrow(void *lib, const char *name) {
  void *sym = dlsym(lib, name);
  if (!sym)
    throw std::runtime_error(std::string("nvrtc: cannot resolve ") + name);
  return reinterpret_cast<T>(sym);
}

inline const RtcApi &RtcApi::get(bool isHip) {
  struct Instance {
    RtcApi api;
    explicit Instance(bool hip) : api(make(hip)) {}
    static RtcApi make(bool hip) {
      RtcApi api{};
      api.isHip = hip;
      if (hip) {
        for (const char *name : {"libhiprtc.so", "libhiprtc.so.7",
                                 "libhiprtc.so.6", "libhiprtc.so.5"}) {
          api.lib = dlopen(name, RTLD_LAZY | RTLD_GLOBAL);
          if (api.lib) break;
        }
        if (!api.lib)
          throw std::runtime_error("nvrtc: Failed to load libhiprtc.so");
        api.createProgram = dlsymOrThrow<decltype(api.createProgram)>(
            api.lib, "hiprtcCreateProgram");
        api.destroyProgram = dlsymOrThrow<decltype(api.destroyProgram)>(
            api.lib, "hiprtcDestroyProgram");
        api.compileProgram = dlsymOrThrow<decltype(api.compileProgram)>(
            api.lib, "hiprtcCompileProgram");
        api.getCodeSize = dlsymOrThrow<decltype(api.getCodeSize)>(
            api.lib, "hiprtcGetCodeSize");
        api.getCode =
            dlsymOrThrow<decltype(api.getCode)>(api.lib, "hiprtcGetCode");
        api.getBinarySize = dlsymOrThrow<decltype(api.getBinarySize)>(
            api.lib, "hiprtcGetCodeSize");
        api.getBinary =
            dlsymOrThrow<decltype(api.getBinary)>(api.lib, "hiprtcGetCode");
        api.getLogSize = dlsymOrThrow<decltype(api.getLogSize)>(
            api.lib, "hiprtcGetProgramLogSize");
        api.getLog =
            dlsymOrThrow<decltype(api.getLog)>(api.lib, "hiprtcGetProgramLog");
        api.addNameExpression = dlsymOrThrow<decltype(api.addNameExpression)>(
            api.lib, "hiprtcAddNameExpression");
        api.getLoweredName = dlsymOrThrow<decltype(api.getLoweredName)>(
            api.lib, "hiprtcGetLoweredName");
        api.getErrorString = dlsymOrThrow<decltype(api.getErrorString)>(
            api.lib, "hiprtcGetErrorString");
        api.version = reinterpret_cast<decltype(api.version)>(
            dlsym(api.lib, "hiprtcVersion"));
        api.getNumSupportedArchs =
            reinterpret_cast<decltype(api.getNumSupportedArchs)>(
                dlsym(api.lib, "hiprtcGetNumSupportedArchs"));
        api.getSupportedArchs =
            reinterpret_cast<decltype(api.getSupportedArchs)>(
                dlsym(api.lib, "hiprtcGetSupportedArchs"));
      } else {
        loadNvrtcBuiltins();
        for (const char *name : {"libnvrtc.so", "libnvrtc.so.13",
                                 "libnvrtc.so.12", "libnvrtc.so.11"}) {
          api.lib = dlopen(name, RTLD_LAZY | RTLD_GLOBAL);
          if (api.lib) break;
        }
        if (!api.lib)
          throw std::runtime_error("nvrtc: Failed to load libnvrtc.so");
        api.createProgram = dlsymOrThrow<decltype(api.createProgram)>(
            api.lib, "nvrtcCreateProgram");
        api.destroyProgram = dlsymOrThrow<decltype(api.destroyProgram)>(
            api.lib, "nvrtcDestroyProgram");
        api.compileProgram = dlsymOrThrow<decltype(api.compileProgram)>(
            api.lib, "nvrtcCompileProgram");
        api.getCodeSize =
            dlsymOrThrow<decltype(api.getCodeSize)>(api.lib, "nvrtcGetPTXSize");
        api.getCode =
            dlsymOrThrow<decltype(api.getCode)>(api.lib, "nvrtcGetPTX");
        api.getBinarySize = dlsymOrThrow<decltype(api.getBinarySize)>(
            api.lib, "nvrtcGetCUBINSize");
        api.getBinary =
            dlsymOrThrow<decltype(api.getBinary)>(api.lib, "nvrtcGetCUBIN");
        api.getLogSize = dlsymOrThrow<decltype(api.getLogSize)>(
            api.lib, "nvrtcGetProgramLogSize");
        api.getLog =
            dlsymOrThrow<decltype(api.getLog)>(api.lib, "nvrtcGetProgramLog");
        api.addNameExpression = dlsymOrThrow<decltype(api.addNameExpression)>(
            api.lib, "nvrtcAddNameExpression");
        api.getLoweredName = dlsymOrThrow<decltype(api.getLoweredName)>(
            api.lib, "nvrtcGetLoweredName");
        api.getErrorString = dlsymOrThrow<decltype(api.getErrorString)>(
            api.lib, "nvrtcGetErrorString");
        api.version = reinterpret_cast<decltype(api.version)>(
            dlsym(api.lib, "nvrtcVersion"));
        api.getNumSupportedArchs =
            reinterpret_cast<decltype(api.getNumSupportedArchs)>(
                dlsym(api.lib, "nvrtcGetNumSupportedArchs"));
        api.getSupportedArchs =
            reinterpret_cast<decltype(api.getSupportedArchs)>(
                dlsym(api.lib, "nvrtcGetSupportedArchs"));
      }
      return api;
    }
  };
  // Lazily initialized, one instance per backend.
  if (isHip) {
    static RtcApi api = Instance::make(true);
    return api;
  }
  static RtcApi api = Instance::make(false);
  return api;
}
}  // namespace detail
}  // namespace nvrtc

namespace nvrtc {
class Error : public std::exception {
 public:
  explicit Error(int result, bool isHip = false)
      : _result(result), _isHip(isHip) {}

  const char *what() const noexcept {
    try {
      const detail::RtcApi &api = detail::RtcApi::get(_isHip);
      if (api.getErrorString) return api.getErrorString(_result);
    } catch (...) {
    }
    return "nvrtc error";
  }

  operator int() const { return _result; }

 private:
  int _result;
  bool _isHip;
};

inline void checkNvrtcCall(int result, bool isHip) {
  if (result != 0) throw Error(result, isHip);
}

inline std::vector<std::string> cudaIncludePaths() {
  return detail::tokenize(CUDA_INCLUDE_DIRS, ";");
}

inline std::vector<std::string> hipIncludePaths() {
  return detail::tokenize(HIP_INCLUDE_DIRS, ";");
}

// Deprecated: returns the paths of the compile-time default backend.
inline std::vector<std::string> findIncludePaths() {
#if defined(__HIP__)
  return hipIncludePaths();
#else
  return cudaIncludePaths();
#endif
}

inline std::string findIncludePath() {
  std::vector<std::string> paths = findIncludePaths();

  if (paths.empty()) {
    throw std::runtime_error("Could not find NVRTC include path");
  }

  // Join paths for backward compatibility
  std::string result = paths[0];
  for (size_t i = 1; i < paths.size(); ++i) {
    result += " -I" + paths[i];
  }

  return result;
}

class Program {
 public:
  // The parameters mirror nvrtcCreateProgram's arguments exactly. The
  // backend (NVRTC vs. HIPRTC) is not a constructor argument; it is
  // inferred at compile() time: a "--offload-arch" option marks a HIP
  // compilation (NVRTC has no such option), otherwise the compile-time
  // default backend is used.
  Program(
      const std::string &src, const std::string &name,
      const std::vector<std::string> &headers = std::vector<std::string>(),
      const std::vector<std::string> &includeNames = std::vector<std::string>())
      : _src(src),
        _name(name),
        _headers(headers),
        _includeNames(includeNames) {}

  explicit Program(const std::string &filename) : _name(filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
      throw std::runtime_error("Error opening file '" + filename +
                               "' in cudawrappers::nvrtc");
    }
    _src.assign(std::istreambuf_iterator<char>{ifs}, {});
  }

  ~Program() {
    if (_api && _api->destroyProgram) _api->destroyProgram(&program);
  }

  Program(const Program &) = delete;
  Program &operator=(const Program &) = delete;

  void compile(const std::vector<std::string> &options) {
    // hipRTC selects its target with "--offload-arch" (e.g.
    // "--offload-arch=gfx1201"); NVRTC has no such option (it uses
    // "-arch=sm_..."), so its presence marks a HIP compilation. When no
    // option gives an indication, fall back to the compile-time default.
    ensureCreated(defaultIsHip() || hasOffloadArch(options));
    std::vector<const char *> c_options;
    std::transform(options.begin(), options.end(),
                   std::back_inserter(c_options),
                   [](const std::string &option) { return option.c_str(); });
    checkNvrtcCall(
        _api->compileProgram(program, static_cast<int>(c_options.size()),
                             c_options.data()),
        _api->isHip);
  }

  // PTX (CUDA) or code object (HIP), depending on the selected backend.
  std::string getPTX() {
    ensureCreated();
    size_t size{};
    std::string ptx;

    checkNvrtcCall(_api->getCodeSize(program, &size), _api->isHip);
    ptx.resize(size);
    checkNvrtcCall(_api->getCode(program, ptx.data()), _api->isHip);
    return ptx;
  }

  std::vector<char> getCUBIN() {
    ensureCreated();
    size_t size{};
    std::vector<char> cubin;

    checkNvrtcCall(_api->getBinarySize(program, &size), _api->isHip);
    cubin.resize(size);
    checkNvrtcCall(_api->getBinary(program, cubin.data()), _api->isHip);
    return cubin;
  }

  std::string getLog() {
    ensureCreated();
    size_t size{};
    std::string log;

    checkNvrtcCall(_api->getLogSize(program, &size), _api->isHip);
    log.resize(size);
    checkNvrtcCall(_api->getLog(program, log.data()), _api->isHip);
    return log;
  }

  void addNameExpression(const std::string &name) {
    ensureCreated();
    checkNvrtcCall(_api->addNameExpression(program, name.c_str()), _api->isHip);
  }

  const char *getLoweredName(const std::string &name) {
    ensureCreated();
    const char *lowered_name;
    checkNvrtcCall(_api->getLoweredName(program, name.c_str(), &lowered_name),
                   _api->isHip);
    return lowered_name;
  }

 private:
  static bool hasOffloadArch(const std::vector<std::string> &options) {
    for (const std::string &option : options)
      if (option.rfind("--offload-arch", 0) == 0) return true;
    return false;
  }

  static bool defaultIsHip() {
#if defined(__HIP__)
    return true;
#else
    return false;
#endif
  }

  // Lazily create the program handle on first use: the opaque handle must be
  // created by the same library that later compiles and queries it, and the
  // library is only known once the compile options (or the compile-time
  // default) have been seen.
  void ensureCreated(bool isHip = defaultIsHip()) {
    if (_api) return;
    _api = &detail::RtcApi::get(isHip);
    std::vector<const char *> c_headers;
    std::transform(_headers.begin(), _headers.end(),
                   std::back_inserter(c_headers),
                   [](const std::string &header) { return header.c_str(); });

    std::vector<const char *> c_includeNames;
    std::transform(
        _includeNames.begin(), _includeNames.end(),
        std::back_inserter(c_includeNames),
        [](const std::string &includeName) { return includeName.c_str(); });

    checkNvrtcCall(_api->createProgram(&program, _src.c_str(), _name.c_str(),
                                       static_cast<int>(c_headers.size()),
                                       c_headers.data(), c_includeNames.data()),
                   _api->isHip);
  }

  const detail::RtcApi *_api{nullptr};
  void *program{nullptr};
  std::string _src;
  std::string _name;
  std::vector<std::string> _headers;
  std::vector<std::string> _includeNames;
};

inline std::pair<int, int> version() {
#if defined(__HIP__)
  const detail::RtcApi &api = detail::RtcApi::get(true);
#else
  const detail::RtcApi &api = detail::RtcApi::get(false);
#endif
  int major{}, minor{};
  if (api.version) checkNvrtcCall(api.version(&major, &minor), api.isHip);
  return {major, minor};
}

inline std::vector<int> getSupportedArchs() {
#if defined(__HIP__)
  const detail::RtcApi &api = detail::RtcApi::get(true);
#else
  const detail::RtcApi &api = detail::RtcApi::get(false);
#endif
  if (!api.getNumSupportedArchs || !api.getSupportedArchs) return {};
  int count{};
  checkNvrtcCall(api.getNumSupportedArchs(&count), api.isHip);
  std::vector<int> archs(count);
  checkNvrtcCall(api.getSupportedArchs(archs.data()), api.isHip);
  return archs;
}

namespace util {

// Composite compute capability: 100*major + 10*minor (norm-compatible with
// __CUDA_ARCH__; e.g. sm_120 -> 1200, gfx1201 -> 1200).  Works on both
// backends because cudawrappers maps the CUDA compute-capability attributes
// onto the corresponding HIP attributes.
inline int capability(const cu::Device &device) {
  const int major =
      device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>();
  const int minor =
      device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();
  return 100 * major + 10 * minor;
}

// The architecture flag to pass to the runtime compiler for a device that
// will run the compiled code: "-arch=sm_120a" for NVIDIA (every architecture
// since sm_90 gets the 'a' suffix, enabling architecture-dependent
// extensions) or "--offload-arch=gfx1201" for AMD.
inline std::string archOption(const cu::Device &device) {
  std::string arch = detail::archName(device);  // "sm_120" or "gfx1201"
  if (device.isCuda()) {
    if (capability(device) >= 900 && arch.back() != 'a' && arch.back() != 'f')
      arch += 'a';
    return "-arch=" + arch;
  }
  return "--offload-arch=" + arch;
}

// "-D__HIP_ARCH__=<composite capability>", the vendor-neutral device-code
// macro.  HIPRTC does not predefine it (unlike NVRTC's __CUDA_ARCH__), so
// pass it explicitly when device code needs it.
inline std::string archDefine(const cu::Device &device) {
  return "-D__HIP_ARCH__=" + std::to_string(capability(device));
}

// Clang builtin-header directory shipped with ROCm, which hipRTC needs via
// -resource-dir= because its bundled clang cannot locate it at run time.
inline std::optional<std::filesystem::path> findClangResourceDir(
    const std::filesystem::path &rocmRoot) {
  for (const std::filesystem::path &clangDir :
       {rocmRoot / "lib/llvm/lib/clang", rocmRoot / "lib/clang"}) {
    std::error_code ec;
    for (const auto &entry :
         std::filesystem::directory_iterator(clangDir, ec)) {
      if (!entry.is_directory()) continue;
      if (std::filesystem::exists(entry.path() / "include" / "stddef.h"))
        return entry.path();
    }
  }
  return std::nullopt;
}

// The -I (and -resource-dir) options needed to compile for the backend of
// the given device: the NVRTC include directories (plus the cccl subdir on
// CUDA 13, which provides <cuda/std/...>) or the ROCm include directories
// (plus the clang builtin headers for hipRTC).  All directories are found by
// CMake at configure time and baked into cudawrappers/config.h.
inline std::vector<std::string> includeOptions(const cu::Device &device) {
  std::vector<std::string> options;

  if (device.isCuda()) {
    for (const std::string &dir : cudaIncludePaths()) {
      if (dir.empty()) continue;
      options.push_back("-I" + dir);
      // CUDA_Toolkit_INCLUDE_DIRS does not always list the cccl subdir
      // (which holds <cuda/std/...> pulled in by <cooperative_groups/...>);
      // add it explicitly when present so NVRTC can find those headers.
      std::filesystem::path ccclDir = std::filesystem::path(dir) / "cccl";
      if (std::filesystem::exists(ccclDir / "cuda" / "std" / "type_traits"))
        options.push_back("-I" + ccclDir.string());
    }
    return options;
  }

  std::vector<std::string> dirs;
  for (const std::string &dir : hipIncludePaths())
    if (!dir.empty()) dirs.push_back(dir);

#if defined(CUDAWRAPPERS_CLANG_RESOURCE_DIR)
  // The clang builtin-header directory found by CMake at configure time; use
  // it when it is present on the system running the compilation.
  if (std::filesystem::exists(CUDAWRAPPERS_CLANG_RESOURCE_DIR)) {
    for (const std::string &dir : dirs) options.push_back("-I" + dir);
    options.push_back("-resource-dir=" CUDAWRAPPERS_CLANG_RESOURCE_DIR);
    return options;
  }
#endif

  // Fallback for when CMake did not provide the clang resource dir (or it is
  // absent at run time, e.g. when the executable runs on a machine with a
  // different ROCm installation): locate the ROCm root, and with it the
  // builtin headers, by searching the usual locations.
  std::filesystem::path rocmRoot;
  std::vector<std::string> rocmRoots;
  if (const char *env = std::getenv("ROCM_PATH")) rocmRoots.push_back(env);
  if (const char *env = std::getenv("HIP_PATH")) rocmRoots.push_back(env);
  rocmRoots.push_back("/opt/rocm");

  for (const std::string &root : rocmRoots) {
    std::filesystem::path includeDir = std::filesystem::path(root) / "include";
    if (std::filesystem::exists(includeDir / "hip" / "hip_runtime.h")) {
      if (dirs.empty()) dirs.push_back(includeDir.string());
      rocmRoot = includeDir.parent_path();
      break;
    }
  }

  for (const std::string &dir : dirs) options.push_back("-I" + dir);

  if (!rocmRoot.empty())
    if (auto resourceDir = findClangResourceDir(rocmRoot))
      options.push_back("-resource-dir=" + resourceDir->string());

  return options;
}

// The full base option set for compiling for the device's backend: the
// architecture flag followed by the include options.
inline std::vector<std::string> compileOptions(const cu::Device &device) {
  std::vector<std::string> options{archOption(device)};
  std::vector<std::string> includes = includeOptions(device);
  options.insert(options.end(), includes.begin(), includes.end());
  return options;
}

}  // namespace util
}  // namespace nvrtc

#endif
