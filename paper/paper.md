______________________________________________________________________

title: 'The CUDA wrappers'
tags:

- C++
- GPU computing
- CUDA
- HIP
- RAII
- runtime compilation
- FFT
- profiling
  authors:
- name: Bram Veenboer
  orcid: 0000-0001-9607-1142
  affiliation: "1"
- name: Leon Oostrum
  orcid: 0000-0001-8724-8372
  affiliation: "2"
- name: John W. Romein
  orcid: 0000-0002-1915-5067
  affiliation: "1"
  affiliations:
- name: ASTRON (Netherlands Institute for Radio Astronomy)
  index: 1
- name: Netherlands eScience Center
  index: 2
  date: April 2026
  bibliography: paper.bib

______________________________________________________________________

# Summary

CUDA Wrappers is a header-only C++ library that simplifies host-side GPU application programming by wrapping NVIDIA's CUDA Driver API, NVRTC (runtime compilation), cuFFT, NVML (GPU monitoring), and NVTX (profiling) libraries. The library applies modern C++ principles [@myers2004], particularly RAII (Resource Acquisition Is Initialization) [@stroustrup2013] and exception-based error handling, to eliminate boilerplate code, automatically manage GPU resources, and provide transparent support for both NVIDIA GPUs (via CUDA) [@cuda] and AMD GPUs (via HIP) [@rocm_hip]. By reducing the amount of code needed to access the CUDA libraries by 50-80%, CUDA Wrappers enables developers to focus on algorithm development rather than resource management and error handling. The library is header-only, making integration straightforward, and maintains a consistent, intuitive API across all wrapped functionality.

# Statement of need

GPU programming with raw CUDA APIs presents several well-known challenges:

1. **Manual resource management**: Developers must explicitly manage device memory, streams, events, and contexts, with no automatic cleanup on errors or scope exit, leading to resource leaks.
1. **Error handling overhead**: Every CUDA call returns a status code that must be checked individually, resulting in repetitive if-statements and increasing code complexity. This intertwining of error handling and normal program flow reduces readability and makes code harder to maintain.
1. **Boilerplate code**: Simple GPU operations require extensive scaffolding, obscuring the core algorithm logic.
1. **Limited portability**: CUDA code is tightly coupled to NVIDIA hardware; supporting AMD GPUs requires parallel codebases or extensive conditional compilation.

Existing solutions like CUDA-C++ Runtime API offer some convenience but still require explicit error checking and manual context management. Other C++ wrappers [@cuda_api_wrappers] provide partial coverage of the CUDA Driver API but lack support for runtime compilation (NVRTC), FFT, and GPU monitoring. High-level libraries like Thrust [@hoberock2011] abstract GPU programming but are less suitable for low-level resource control and custom kernels, while emerging standards like SYCL [@khronos_sycl] have a steeper learning curve and target a different ecosystem.

CUDA Wrappers addresses these gaps by:

- **Automatic resource management** via RAII principles: GPU memory, streams, and contexts are cleaned up automatically when they go out of scope, even on exception [@cppreference_raii].
- **Exception-based error handling**: All wrapped CUDA calls throw C++ exceptions on failure, eliminating the need for manual status-code checking.
- **Comprehensive library coverage**: Wrappers for Driver API, NVRTC, cuFFT, NVML, and NVTX enable complete GPU workflows within a single library.
- **Transparent multi-vendor support**: A single codebase compiles for both NVIDIA (CUDA) and AMD (HIP) GPUs via compile-time backend selection.

The library is used as a fundamental building block in multiple production systems developed by ASTRON (the Netherlands Institute for Radio Astronomy), including the Imaging Domain Grid (IDG), Tensor Core Correlator, GPU-Filter, and related signal-processing pipelines. This established track record in demanding, real-world applications (radio astronomical data processing) and about two decades of GPU application development validates the design and reliability of the library.

# Software design

## Architecture and key components

CUDA Wrappers is structured as a collection of modular, header-only wrappers organized by CUDA subsystem:

- **`cu.hpp`**: Core wrapper for the CUDA Driver API, providing RAII containers for devices, contexts, streams, events, memory, and kernels.
- **`nvrtc.hpp`**: Runtime kernel compilation with source embedding and error handling.
- **`cufft.hpp`**: Type-safe FFT operations (1D/2D, complex-to-complex, real-to-complex transforms).
- **`nvml.hpp`**: GPU device monitoring (NVIDIA-only; clock, power consumption, memory usage).
- **`nvtx.hpp`**: Performance profiling instrumentation.
- **`macros.hpp`**: HIP backend translation layer for NVIDIA/AMD portability.

Each wrapper follows the same design philosophy: acquire resources in constructors, release them in destructors, and throw exceptions on failure. The APIs of the CUDA Wrappers are intentionally similar to the original CUDA library APIs, so developers familiar with CUDA can adopt the wrappers quickly.

## Core design principles

### RAII for automatic resource management

All GPU resources (memory, streams, contexts, modules) are encapsulated in C++ objects:

```cpp
cu::init();                                     // Initialize the environment
cu::Device device(0);                           // Initialize the first GPU
cu::Context context(CU_CTX_SCHED_AUTO, device); // Initialize context
context.setCurrent();                           // Make sure the context is current
cu::Stream stream;                              // Initialize a stream
cu::DeviceMemory mem(1024 * 1024);              // Allocate device memory
```

This pattern ensures that even if an exception is thrown, all resources are properly released, preventing resource leaks and dangling pointers, a common source of bugs in GPU code.

### Exception-based error handling

Every CUDA call is wrapped to convert error codes into C++ exceptions:

```cpp
try {
    cu::DeviceMemory mem(invalid_size);  // Throws cu::Error on failure
} catch (const cu::Error& e) {
    std::cerr << "GPU error: " << e.what() << std::endl;
    // Cleanup happens automatically
}
```

This eliminates the tedious pattern of checking every CUDA call's return value, making code significantly more readable and less prone to missed error cases. In environments that support C++23, exception diagnostics can be further enhanced with `std::basic_stacktrace` to help locate the origin of a thrown exception.

### Runtime compilation for flexibility

`nvrtc::Program` enables just-in-time kernel compilation, supporting dynamic kernel generation and optimization:

```cpp
const char* kernel_source = R"(
  __global__ void add(int* a, int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
  }
)";

nvrtc::Program program(kernel_source, "kernel.cu");
program.compile({"--use_fast_math", "-O3", "--gpu-architecture=compute_86"});
cu::Module module(program.getPTX());
cu::Function add_kernel(module, "add");
```

This capability is especially useful when parameters are not known when the host code is compiled, but are known just before launching a kernel. In that case, the host can pass these values as compile definitions to NVRTC (for example via `-DNR_STATIONS=...`), so the generated device code sees constants instead of runtime variables. It also allows runtime-detected GPU architecture and generation options to be passed to the compiler, for example `--gpu-architecture=compute_86`, so the kernel is optimized for the actual target device. This often enables stronger compiler optimizations such as constant folding, unrolling, and simplified address arithmetic.

It also allows clean expression of multidimensional data layouts using compile-time dimensions in the kernel source, for example:

```cpp
typedef cuda::std::complex<float> Complex;
typedef Complex (*InputType)[NR_STATIONS][NR_CHANNELS][NR_INTEGRATIONS]
                           [NR_SAMPLES_PER_INTEGRATION];
typedef Complex (*OutputType)[NR_INTEGRATIONS][NR_BASELINES][NR_CHANNELS]
                            [NR_POLARIZATIONS][NR_POLARIZATIONS];
```

This approach keeps kernels readable while still generating specialized code for each problem configuration. Using explicit array types also enables clear indexing such as `output[integration][baseline][channel][x_pol][y_pol]`, instead of hard-to-read flattened expressions like `output[(((integration * nr_baselines + baseline) * nr_channels + channel) * nr_polarizations + x_pol) * nr_polarizations + y_pol]`.

### Transparent multi-GPU support (CUDA/HIP)

Cudawrappers abstracts differences between NVIDIA and AMD GPU APIs via compile-time backend selection. The same C++ source compiles for both backends:

```bash
# Compile for NVIDIA GPUs (default)
cmake -B build

# Compile for AMD GPUs
cmake -B build -DCUDAWRAPPERS_BACKEND=HIP
```

Internal macro translation (in `macros.hpp`) maps CUDA types and calls to HIP equivalents, enabling a single codebase to target multiple GPU architectures.

## API design philosophy

- **Consistent interfaces**: All wrappers follow similar patterns (RAII, exception-based error handling).
- **Type safety**: Template-based designs for compile-time checking (e.g., `FFT1D<float>` vs. `FFT1D<double>`).
- **Optional components**: Selective compilation via `-DCUDAWRAPPERS_COMPONENTS=cu;nvrtc;cufft` to minimize dependencies.
- **Header-only**: No compiled library artifacts; simple `#include` and link flags sufficient for integration.
- **C++17 standard** \[@cpp17\]: Modern language features for cleaner, more expressive code.

# State of the field

Several alternative approaches to GPU programming exist:

- **Raw CUDA/HIP**: Maximum control but requires extensive manual resource and error management.
- **NVIDIA CUDA C++ API**: Official higher-level wrapper; simpler than raw CUDA but still requires explicit error checking.
- **cuda-api-wrappers**: Open-source wrapper focused on Driver API; lacks NVRTC, cuFFT, and NVML support [@cuda_api_wrappers].
- **Thrust**: High-level GPU algorithms library [@hoberock2011]; less suitable for low-level resource control.
- **SYCL**: Vendor-agnostic programming model [@khronos_sycl]; steeper learning curve and less established ecosystem.

CUDA Wrappers fills a practical niche: it provides RAII and exception-based error handling (reducing cognitive load and boilerplate), comprehensive coverage of CUDA subsystems, and transparent dual-backend support, all in a lightweight, header-only package suitable for production HPC and scientific computing environments.

# Applications and research impact

CUDA Wrappers serves as a foundational building block for several production GPU-accelerated systems developed by ASTRON and collaborators:

- **IDG (Imaging-Domain Gridding)** \[@idg2017\]: GPU-accelerated gridding library (for radio-astronomical imaging) [https://git.astron.nl/RD/idg](https://git.astron.nl/RD/idg)
- **Tensor Core Correlator** \[@tcc2021\]: Tensor-core accelerated correlator library [https://git.astron.nl/RD/tensor-core-correlator](https://git.astron.nl/RD/tensor-core-correlator)
- **Tensor Core Beamformer** \[@tcbf2025\]: Tensor-core accelerated beamforming library for multidisciplinary signal processing [https://github.com/nlesc-recruit/ccglib](https://github.com/nlesc-recruit/ccglib)
- **GPU-Filter**: GPU-accelerated signal filtering library for radio astronomy. [https://git.astron.nl/RD/gpu-filter](https://git.astron.nl/RD/gpu-filter)
- **ASTRA**: GPU-accelerated signal-processing toolkit for radio astronomy [https://git.astron.nl/cobalt/astra](https://git.astron.nl/cobalt/astra)
- **Dedisp** \[@fdd2022\]: GPU-accelerated dedispersion library for pulsar search. [https://git.astron.nl/RD/dedisp](https://git.astron.nl/RD/dedisp)
- **Cobalt**: Real-time GPU-accelerated correlator and beamformer application [https://git.astron.nl/cobalt/cobalt](https://git.astron.nl/cobalt/cobalt)
- **CCGlib**: General-purpose library for tensor-core accelerated complex GEMM. [https://github.com/nlesc-recruit/ccglib](https://github.com/nlesc-recruit/ccglib)

These systems collectively process terabytes of radio-astronomical data daily, validating CUDA Wrappers' reliability, performance, and design in demanding, production environments. The library has also been adopted in graduate-level GPU programming courses and other GPU applications developed at NLeSC, demonstrating pedagogical value and practical impact.

# Software features and development

## Recent major features (v1.0.0, April 2026)

- **CUDA 13 support**: Updated for latest NVIDIA CUDA releases and CUDA Core Compute Library (CCCL) compatibility.
- **NVTX 3 integration**: Modern profiling framework (automatic for CUDA >= 12.8).
- **Runtime kernel compilation enhancements**: `nvrtc::addNameExpression`, `nvrtc::getLoweredName`, and improved `nvrtc::findIncludePaths`.
- **Advanced memory operations**: `cu::Stream::launchHostFunc` for host-device synchronization; async memory allocation and 2D transfers.
- **GPU task graphs**: `cu::Graph` for complex computation pipelines and kernel scheduling.
- **HIP 7 compatibility**: Extended support for AMD ROCM platforms.
- **Selectable components**: Build only needed subsystems via CMake configuration.

## Development timeline

The library has evolved from early prototypes before 2022 to production-ready status:

- **v0.1.0 (2022)**: Initial release with basic CUDA Driver API wrapper.
- **v0.5.0 (2023)**: Added cuFFT, host memory support, and async operations.
- **v0.9.0 (2025)**: HIP compatibility, expanded memory operations, C++14 upgrade.
- **v1.0.0 (2026)**: CUDA 13, C++17 standard, production-ready with enhanced NVRTC and GPU graphs.

# Technical specifications

- **Language**: Modern C++ (C++17 standard as of v1.0)
- **Dependencies**: CUDA >= 10.0 (tested up to 13.0) or ROCM >= 6.1 (HIP)
- **Build System**: CMake >= 3.17
- **Platforms**: Linux (x86_64), with HIP support for AMD ROCM
- **Architecture**: Header-only library; no compiled artifacts
- **Testing**: Comprehensive test suite (CTest integration) covering all wrapped libraries
- **License**: Apache 2.0

# Reproducibility and availability

CUDA Wrappers is open-source and available on GitHub [@recruit_github] at [https://github.com/nlesc-recruit/cudawrappers](https://github.com/nlesc-recruit/cudawrappers), with related ASTRON software infrastructure hosted on ASTRON GitLab [@astron_gitlab]. The software is registered with Zenodo (DOI: 10.5281/zenodo.6076447) and follows semantic versioning. Detailed documentation, build instructions, and usage examples are provided in the repository README and inline code documentation. The comprehensive test suite enables easy verification of correct installation and functionality across different CUDA/HIP environments.

# Acknowledgements

CUDA Wrappers was developed by the Netherlands eScience Center and ASTRON (the Netherlands Institute for Radio Astronomy)
under grant number ETEC.2020.025 (RECRUIT) and supported by the RADIOBLOCKS grant from the European Commission (HORIZON-INFRA-2022-TECH-01, Grant Agreement nr. 101093934). The library benefits from contributions and feedback from the ASTRON scientific and engineering teams and the broader GPU computing community. We thank all contributors and users who have provided bug reports, feature requests, and improvements.

# AI usage disclosure

In line with JOSS guidance, we disclose that generative AI tools were used during development and manuscript preparation. From approximately 2025 onward, we used ChatGPT, DeepSeek, and GitHub Copilot (in the IDE) to support implementation and debugging tasks, including code refactoring and development of CMake logic for embedding CUDA kernel sources into libraries. We also used GitHub's built-in AI reviewer to help identify small but potentially impactful issues, such as incomplete version-number updates. The model versions evolved during the project period and were not pinned to a single fixed release.

AI assistance complemented, rather than replaced, the project's established quality controls, including pre-commit checks (linters and formatters), Codacy, and human code review. During manuscript preparation, generative AI was used to transform an initial rough draft into a more structured draft. The final text was reviewed, edited, and approved by the human authors.

# References

- CUDA Toolkit Documentation [@cuda]
- HIP documentation [@rocm_hip]
- Modern C++ Design: Generic Programming and Design Patterns Applied [@myers2004]
- The C++ Programming Language [@stroustrup2013]
- Thrust: A Productivity-Oriented Library for CUDA [@hoberock2011]
- SYCL Specification [@khronos_sycl]
- CUDA API Wrappers [@cuda_api_wrappers]
- Image-Domain Gridding on Graphics Processors [@idg2017]
- The Tensor-Core Correlator [@tcc2021]
- The Tensor-Core Beamformer: A High-Speed Signal-Processing Library for Multidisciplinary Use [@tcbf2025]
- Fourier-domain dedispersion [@fdd2022]
- Netherlands Institute for Radio Astronomy Open Source [@recruit_github]
- ASTRON GitLab repository [@astron_gitlab]
- C++17 Features Overview [@cpp17]
- Resource Acquisition Is Initialization (RAII) [@cppreference_raii]
