#include <nvrtc.h>
#include <iostream>
#include <stdexcept>
#include <vector>

static void checkNvrtc(nvrtcResult r, const char* msg) {
    if (r != NVRTC_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": " + nvrtcGetErrorString(r));
    }
}

#include <dlfcn.h>
#include <string>

std::string findCudaLibDir() {
    Dl_info info;
    if (dladdr((void*)&nvrtcCreateProgram, &info) == 0) {
        throw std::runtime_error("dladdr failed for nvrtc");
    }
    std::string path = info.dli_fname;
    auto pos = path.find_last_of('/');
    if (pos != std::string::npos) path.erase(pos);
    return path; // directory containing libnvrtc.so
}

void prependEnvVar(const std::string& envVarName, const std::string& dir) {
    const char* oldPath = getenv(envVarName.c_str());
    std::string newPath = dir;
    if (oldPath) {
      newPath += ":" + std::string(oldPath);
    }
    setenv(envVarName.c_str(), newPath.c_str(), 1);
}


int main() {
    dlopen("libnvrtc-builtins.so", RTLD_LAZY);
    // auto cudaLibDir = findCudaLibDir();
    // prependEnvVar("LD_LIBRARY_PATH", cudaLibDir);

    const char* src = R"(
    extern "C" __global__ void saxpy(float a, const float* x, const float* y, float* z) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        z[i] = a * x[i] + y[i];
    })";

    nvrtcProgram prog;
    checkNvrtc(nvrtcCreateProgram(&prog, src, "saxpy.cu", 0, nullptr, nullptr),
               "nvrtcCreateProgram");

    const char* opts[] = {
        "--std=c++14"
        // optionally: "--include-path=/usr/local/cuda/include"
    };

    nvrtcResult compileResult = nvrtcCompileProgram(prog, 1, opts);

    // Print log if compilation fails
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    std::vector<char> log(logSize);
    nvrtcGetProgramLog(prog, log.data());
    if (logSize > 1) std::cerr << log.data() << std::endl;

    checkNvrtc(compileResult, "nvrtcCompileProgram");

    // Retrieve generated PTX
    size_t ptxSize;
    checkNvrtc(nvrtcGetPTXSize(prog, &ptxSize), "nvrtcGetPTXSize");
    std::vector<char> ptx(ptxSize);
    checkNvrtc(nvrtcGetPTX(prog, ptx.data()), "nvrtcGetPTX");

    std::cout << "PTX size: " << ptxSize << " bytes" << std::endl;

    std::cout << "PTX code:\n" << std::string(ptx.data(), ptxSize) << std::endl;

    nvrtcDestroyProgram(&prog);
}
