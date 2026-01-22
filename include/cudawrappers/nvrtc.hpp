#if !defined NVRTC_H
#define NVRTC_H
#include <dlfcn.h>
#include <link.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(__HIP__)
#include <cuda.h>
#include <nvrtc.h>
#else
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <cudawrappers/macros.hpp>
#endif
#include <cudawrappers/config.h>
namespace {
std::vector<std::string> tokenize(const std::string &input,
                                  const std::string &delimiter) {
  std::string s = input;
  size_t pos = 0;
  std::string token;
  std::vector<std::string> tokens;
  while ((pos = s.find(delimiter)) != std::string::npos) {
    token = s.substr(0, pos);
    tokens.push_back(token);
    s.erase(0, pos + delimiter.length());
  }
  tokens.push_back(s);
  return tokens;
}

void loadNvrtcBuiltins() {
  if (!dlopen("libnvrtc-builtins.so", RTLD_LAZY)) {
    throw std::runtime_error("Failed to load libnvrtc-builtins.so");
  }
}
}  // namespace

namespace nvrtc {
class Error : public std::exception {
 public:
  explicit Error(nvrtcResult result) : _result(result) {}

  const char *what() const noexcept { return nvrtcGetErrorString(_result); }

  operator nvrtcResult() const { return _result; }

 private:
  nvrtcResult _result;
};

inline void checkNvrtcCall(nvrtcResult result) {
  if (result != NVRTC_SUCCESS) throw Error(result);
}

inline std::vector<std::string> findIncludePaths() {
#if defined(__HIP__)
  std::string path = HIP_INCLUDE_DIRS;
#else
  std::string path = CUDA_INCLUDE_DIRS;
#endif

  std::vector<std::string> paths = tokenize(path, ";");

#if CUDA_VERSION >= 13000
  const std::string cccl_suffix = "cccl";

  // Check whether any of the paths contain /cccl
  for (const std::string &path : paths) {
    size_t pos = path.rfind("/" + cccl_suffix);
    if (pos != std::string::npos &&
        pos == path.size() - (cccl_suffix.size() + 1)) {
      return paths;
    }
  }

  // Try to find the path that contains /cccl
  for (const auto &path : paths) {
    std::filesystem::path cccl_path = std::filesystem::path(path) / cccl_suffix;

    // Add the path if it exists
    if (std::filesystem::exists(cccl_path) &&
        std::filesystem::is_directory(cccl_path)) {
      paths.emplace_back(path + "/" + cccl_suffix);
      break;
    }
  }
#endif

  return paths;
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
  Program(const std::string &src, const std::string &name,
          const std::vector<std::string> &headers = std::vector<std::string>(),
          const std::vector<std::string> &includeNames =
              std::vector<std::string>()) {
#if !defined(__HIP__)
    loadNvrtcBuiltins();
#endif
    std::vector<const char *> c_headers;
    std::transform(headers.begin(), headers.end(),
                   std::back_inserter(c_headers),
                   [](const std::string &header) { return header.c_str(); });

    std::vector<const char *> c_includeNames;
    std::transform(
        includeNames.begin(), includeNames.end(),
        std::back_inserter(c_includeNames),
        [](const std::string &includeName) { return includeName.c_str(); });

    checkNvrtcCall(nvrtcCreateProgram(&program, src.c_str(), name.c_str(),
                                      static_cast<int>(c_headers.size()),
                                      c_headers.data(), c_includeNames.data()));
  }

  explicit Program(const std::string &filename) {
#if !defined(__HIP__)
    loadNvrtcBuiltins();
#endif
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
      throw std::runtime_error("Error opening file '" + filename +
                               "' in cudawrappers::nvrtc");
    }
    std::string source(std::istreambuf_iterator<char>{ifs}, {});
    checkNvrtcCall(nvrtcCreateProgram(&program, source.c_str(),
                                      filename.c_str(), 0, nullptr, nullptr));
  }

  ~Program() { checkNvrtcCall(nvrtcDestroyProgram(&program)); }

  void compile(const std::vector<std::string> &options) {
    std::vector<const char *> c_options;
    std::transform(options.begin(), options.end(),
                   std::back_inserter(c_options),
                   [](const std::string &option) { return option.c_str(); });
    checkNvrtcCall(nvrtcCompileProgram(
        program, static_cast<int>(c_options.size()), c_options.data()));
  }

  std::string getPTX() {
    size_t size{};
    std::string ptx;

    checkNvrtcCall(nvrtcGetPTXSize(program, &size));
    ptx.resize(size);
    checkNvrtcCall(nvrtcGetPTX(program, const_cast<char *>(ptx.data())));
    return ptx;
  }

#if CUDA_VERSION >= 11020
  std::vector<char> getCUBIN() {
    size_t size{};
    std::vector<char> cubin;

    checkNvrtcCall(nvrtcGetCUBINSize(program, &size));
    cubin.resize(size);
    checkNvrtcCall(nvrtcGetCUBIN(program, &cubin[0]));
    return cubin;
  }
#endif

  std::string getLog() {
    size_t size{};
    std::string log;

    checkNvrtcCall(nvrtcGetProgramLogSize(program, &size));
    log.resize(size);
    checkNvrtcCall(nvrtcGetProgramLog(program, &log[0]));
    return log;
  }

  void addNameExpression(const std::string &name) {
    checkNvrtcCall(nvrtcAddNameExpression(program, name.c_str()));
  }

  const char *getLoweredName(const std::string &name) {
    const char *lowered_name;
    checkNvrtcCall(nvrtcGetLoweredName(program, name.c_str(), &lowered_name));
    return lowered_name;
  }

 private:
  nvrtcProgram program{};
};
}  // namespace nvrtc

#endif
