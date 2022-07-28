#if !defined NVRTC_H
#define NVRTC_H

#include <algorithm>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>

namespace nvrtc {
class Error : public std::exception {
 public:
  Error(nvrtcResult result) : _result(result) {}

  virtual const char *what() const noexcept;

  operator nvrtcResult() const { return _result; }

 private:
  nvrtcResult _result;
};

inline void checkNvrtcCall(nvrtcResult result) {
  if (result != NVRTC_SUCCESS) throw Error(result);
}

class Program {
 public:
  Program(const std::string &src, const std::string &name, int numHeaders = 0,
          const char *headers[] = nullptr,
          const char *includeNames[] =
              nullptr)  // TODO: use std::vector<std::string>
  {
    checkNvrtcCall(nvrtcCreateProgram(&program, src.c_str(), name.c_str(),
                                      numHeaders, headers, includeNames));
  }

  Program(const std::string &filename) {
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
    checkNvrtcCall(
        nvrtcCompileProgram(program, c_options.size(), c_options.data()));
  }

  std::string getPTX() {
    size_t size;
    std::string ptx;

    checkNvrtcCall(nvrtcGetPTXSize(program, &size));
    ptx.resize(size);
    checkNvrtcCall(nvrtcGetPTX(program, &ptx[0]));
    return ptx;
  }

#if CUDA_VERSION >= 11020
  std::vector<char> getCUBIN() {
    size_t size;
    std::vector<char> cubin;

    checkNvrtcCall(nvrtcGetCUBINSize(program, &size));
    cubin.resize(size);
    checkNvrtcCall(nvrtcGetCUBIN(program, &cubin[0]));
    return cubin;
  }
#endif

  std::string getLog() {
    size_t size;
    std::string log;

    checkNvrtcCall(nvrtcGetProgramLogSize(program, &size));
    log.resize(size);
    checkNvrtcCall(nvrtcGetProgramLog(program, &log[0]));
    return log;
  }

 private:
  nvrtcProgram program;
};
}  // namespace nvrtc

#endif
