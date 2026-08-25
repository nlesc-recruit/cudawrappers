#if !defined NVRTC_H
#define NVRTC_H
#include <dlfcn.h>
#include <link.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
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
  if (!sym) throw std::runtime_error(std::string("nvrtc: cannot resolve ") +
                                     name);
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
        for (const char *name :
             {"libhiprtc.so", "libhiprtc.so.7", "libhiprtc.so.6",
              "libhiprtc.so.5"}) {
          api.lib = dlopen(name, RTLD_LAZY | RTLD_GLOBAL);
          if (api.lib) break;
        }
        if (!api.lib)
          throw std::runtime_error("nvrtc: Failed to load libhiprtc.so");
        api.createProgram =
            dlsymOrThrow<decltype(api.createProgram)>(api.lib, "hiprtcCreateProgram");
        api.destroyProgram =
            dlsymOrThrow<decltype(api.destroyProgram)>(api.lib, "hiprtcDestroyProgram");
        api.compileProgram =
            dlsymOrThrow<decltype(api.compileProgram)>(api.lib, "hiprtcCompileProgram");
        api.getCodeSize =
            dlsymOrThrow<decltype(api.getCodeSize)>(api.lib, "hiprtcGetCodeSize");
        api.getCode = dlsymOrThrow<decltype(api.getCode)>(api.lib, "hiprtcGetCode");
        api.getBinarySize =
            dlsymOrThrow<decltype(api.getBinarySize)>(api.lib, "hiprtcGetCodeSize");
        api.getBinary =
            dlsymOrThrow<decltype(api.getBinary)>(api.lib, "hiprtcGetCode");
        api.getLogSize =
            dlsymOrThrow<decltype(api.getLogSize)>(api.lib, "hiprtcGetProgramLogSize");
        api.getLog = dlsymOrThrow<decltype(api.getLog)>(api.lib, "hiprtcGetProgramLog");
        api.addNameExpression =
            dlsymOrThrow<decltype(api.addNameExpression)>(api.lib, "hiprtcAddNameExpression");
        api.getLoweredName =
            dlsymOrThrow<decltype(api.getLoweredName)>(api.lib, "hiprtcGetLoweredName");
        api.getErrorString =
            dlsymOrThrow<decltype(api.getErrorString)>(api.lib, "hiprtcGetErrorString");
        api.version = reinterpret_cast<decltype(api.version)>(
            dlsym(api.lib, "hiprtcVersion"));
        api.getNumSupportedArchs = reinterpret_cast<decltype(api.getNumSupportedArchs)>(
            dlsym(api.lib, "hiprtcGetNumSupportedArchs"));
        api.getSupportedArchs = reinterpret_cast<decltype(api.getSupportedArchs)>(
            dlsym(api.lib, "hiprtcGetSupportedArchs"));
      } else {
        loadNvrtcBuiltins();
        for (const char *name :
             {"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12",
              "libnvrtc.so.11"}) {
          api.lib = dlopen(name, RTLD_LAZY | RTLD_GLOBAL);
          if (api.lib) break;
        }
        if (!api.lib)
          throw std::runtime_error("nvrtc: Failed to load libnvrtc.so");
        api.createProgram =
            dlsymOrThrow<decltype(api.createProgram)>(api.lib, "nvrtcCreateProgram");
        api.destroyProgram =
            dlsymOrThrow<decltype(api.destroyProgram)>(api.lib, "nvrtcDestroyProgram");
        api.compileProgram =
            dlsymOrThrow<decltype(api.compileProgram)>(api.lib, "nvrtcCompileProgram");
        api.getCodeSize =
            dlsymOrThrow<decltype(api.getCodeSize)>(api.lib, "nvrtcGetPTXSize");
        api.getCode = dlsymOrThrow<decltype(api.getCode)>(api.lib, "nvrtcGetPTX");
        api.getBinarySize =
            dlsymOrThrow<decltype(api.getBinarySize)>(api.lib, "nvrtcGetCUBINSize");
        api.getBinary =
            dlsymOrThrow<decltype(api.getBinary)>(api.lib, "nvrtcGetCUBIN");
        api.getLogSize =
            dlsymOrThrow<decltype(api.getLogSize)>(api.lib, "nvrtcGetProgramLogSize");
        api.getLog = dlsymOrThrow<decltype(api.getLog)>(api.lib, "nvrtcGetProgramLog");
        api.addNameExpression =
            dlsymOrThrow<decltype(api.addNameExpression)>(api.lib, "nvrtcAddNameExpression");
        api.getLoweredName =
            dlsymOrThrow<decltype(api.getLoweredName)>(api.lib, "nvrtcGetLoweredName");
        api.getErrorString =
            dlsymOrThrow<decltype(api.getErrorString)>(api.lib, "nvrtcGetErrorString");
        api.version = reinterpret_cast<decltype(api.version)>(
            dlsym(api.lib, "nvrtcVersion"));
        api.getNumSupportedArchs = reinterpret_cast<decltype(api.getNumSupportedArchs)>(
            dlsym(api.lib, "nvrtcGetNumSupportedArchs"));
        api.getSupportedArchs = reinterpret_cast<decltype(api.getSupportedArchs)>(
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
  // backendIdx: global cudawrappers device/backend index selecting between
  // NVRTC (CUDA) and HIPRTC (HIP) at runtime. A negative value selects the
  // compile-time default backend.
  Program(const std::string &src, const std::string &name,
          const std::vector<std::string> &headers = std::vector<std::string>(),
          const std::vector<std::string> &includeNames = std::vector<std::string>(),
          int backendIdx = -1)
      : api(detail::RtcApi::get(resolveIsHip(backendIdx))) {
    std::vector<const char *> c_headers;
    std::transform(headers.begin(), headers.end(),
                   std::back_inserter(c_headers),
                   [](const std::string &header) { return header.c_str(); });

    std::vector<const char *> c_includeNames;
    std::transform(
        includeNames.begin(), includeNames.end(),
        std::back_inserter(c_includeNames),
        [](const std::string &includeName) { return includeName.c_str(); });

    checkNvrtcCall(api.createProgram(&program, src.c_str(), name.c_str(),
                                     static_cast<int>(c_headers.size()),
                                     c_headers.data(), c_includeNames.data()),
                   api.isHip);
  }

  explicit Program(const std::string &filename, int backendIdx = -1)
      : api(detail::RtcApi::get(resolveIsHip(backendIdx))) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
      throw std::runtime_error("Error opening file '" + filename +
                               "' in cudawrappers::nvrtc");
    }
    std::string source(std::istreambuf_iterator<char>{ifs}, {});
    checkNvrtcCall(
        api.createProgram(&program, source.c_str(), filename.c_str(), 0,
                          nullptr, nullptr),
        api.isHip);
  }

  ~Program() {
    if (api.destroyProgram) api.destroyProgram(&program);
  }

  Program(const Program &) = delete;
  Program &operator=(const Program &) = delete;

  void compile(const std::vector<std::string> &options) {
    std::vector<const char *> c_options;
    std::transform(options.begin(), options.end(),
                   std::back_inserter(c_options),
                   [](const std::string &option) { return option.c_str(); });
    checkNvrtcCall(
        api.compileProgram(program, static_cast<int>(c_options.size()),
                           c_options.data()),
        api.isHip);
  }

  // PTX (CUDA) or code object (HIP), depending on the selected backend.
  std::string getPTX() {
    size_t size{};
    std::string ptx;

    checkNvrtcCall(api.getCodeSize(program, &size), api.isHip);
    ptx.resize(size);
    checkNvrtcCall(api.getCode(program, ptx.data()), api.isHip);
    return ptx;
  }

  std::vector<char> getCUBIN() {
    size_t size{};
    std::vector<char> cubin;

    checkNvrtcCall(api.getBinarySize(program, &size), api.isHip);
    cubin.resize(size);
    checkNvrtcCall(api.getBinary(program, cubin.data()), api.isHip);
    return cubin;
  }

  std::string getLog() {
    size_t size{};
    std::string log;

    checkNvrtcCall(api.getLogSize(program, &size), api.isHip);
    log.resize(size);
    checkNvrtcCall(api.getLog(program, log.data()), api.isHip);
    return log;
  }

  void addNameExpression(const std::string &name) {
    checkNvrtcCall(api.addNameExpression(program, name.c_str()), api.isHip);
  }

  const char *getLoweredName(const std::string &name) {
    const char *lowered_name;
    checkNvrtcCall(api.getLoweredName(program, name.c_str(), &lowered_name),
                   api.isHip);
    return lowered_name;
  }

 private:
  static bool resolveIsHip(int backendIdx) {
    if (backendIdx < 0) {
#if defined(__HIP__)
      return true;
#else
      return false;
#endif
    }
    return !cu::Device::backendIsCuda(backendIdx);
  }

  const detail::RtcApi &api;
  void *program{nullptr};
};

inline std::pair<int, int> version() {
#if defined(__HIP__)
  const detail::RtcApi &api = detail::RtcApi::get(true);
#else
  const detail::RtcApi &api = detail::RtcApi::get(false);
#endif
  int major{}, minor{};
  if (api.version)
    checkNvrtcCall(api.version(&major, &minor), api.isHip);
  return {major, minor};
}

inline std::vector<int> getSupportedArchs() {
#if defined(__HIP__)
  const detail::RtcApi &api = detail::RtcApi::get(true);
#else
  const detail::RtcApi &api = detail::RtcApi::get(false);
#endif
  if (!api.getNumSupportedArchs || !api.getSupportedArchs)
    return {};
  int count{};
  checkNvrtcCall(api.getNumSupportedArchs(&count), api.isHip);
  std::vector<int> archs(count);
  checkNvrtcCall(api.getSupportedArchs(archs.data()), api.isHip);
  return archs;
}

}  // namespace nvrtc

#endif
