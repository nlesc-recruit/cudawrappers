#if !defined NVRTC_H
#define NVRTC_H
#include <link.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>

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

inline std::string findIncludePath() {
  std::string path;

  if (dl_iterate_phdr(
          [](struct dl_phdr_info *info, size_t, void *arg) -> int {
            std::string &path = *static_cast<std::string *>(arg);
            path = info->dlpi_name;
            return path.find("libnvrtc.so") != std::string::npos;
          },
          &path))
    for (size_t pos; (pos = path.find_last_of("/")) != std::string::npos;) {
      path.erase(pos);  // remove last part of path

      struct stat buffer;
      const std::string filename = path + "/include/cuda.h";
      if (stat(filename.c_str(), &buffer) == 0) {
        return path + "/include";
      }
    }

  throw std::runtime_error("Could not find NVRTC include path");
}

class Program {
 public:
  Program(const std::string &src, const std::string &name,
          const std::vector<std::string> &headers = std::vector<std::string>(),
          const std::vector<std::string> &includeNames =
              std::vector<std::string>()) {
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
    checkNvrtcCall(nvrtcGetPTX(program, &ptx[0]));
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

 private:
  nvrtcProgram program{};
};
}  // namespace nvrtc

#endif
