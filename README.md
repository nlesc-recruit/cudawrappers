[![github url](https://img.shields.io/badge/github-url-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/nlesc-recruit/cudawrappers)
[![github license badge](https://img.shields.io/github/license/nlesc-recruit/cudawrappers)](https://github.com/nlesc-recruit/cudawrappers)
[![DOI](https://zenodo.org/badge/424944643.svg)](https://zenodo.org/badge/latestdoi/424944643)
[![cii badge](https://bestpractices.coreinfrastructure.org/projects/5686/badge)](https://bestpractices.coreinfrastructure.org/projects/5686)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B%20%20%E2%97%8F%20%20%E2%97%8F-orange)](https://fair-software.eu)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/d38b0338fda24733ab41a64915af8248)](https://www.codacy.com/gh/nlesc-recruit/cudawrappers/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nlesc-recruit/cudawrappers&amp;utm_campaign=Badge_Grade)
[![citation metadata](https://github.com/nlesc-recruit/cudawrappers/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/nlesc-recruit/cudawrappers/actions/workflows/cffconvert.yml)

# cudawrappers

This library is a C++ wrapper for the Nvidia C libraries (e.g. CUDA driver, nvrtc, cuFFT etc.). The main purposes are:

1. _easier resource management_, leading to _lower risk of programming errors_;
2. _better fault handling_ (through exceptions);
3. _more compact user code_.

Originally, the API enforced RAII to even further reduce the risk of faulty code, but enforcing RAII and compatibility with (unmanaged) objects obtained outside this API are mutually exclusive.

## Requirements

| Software    | Minimum version |
| ----------- | ----------- |
| CUDA        | 10.0 or later |
| CMake       | 3.17 or later |
| gcc         | 9.3 or later  |
| OS          | Linux distro (amd64) |

| Hardware    | Type |
| ----------- | ----------- |
| GPU architecture        | [NVIDIA PASCAL](https://www.nvidia.com/en-in/geforce/products/10series/architecture/) or newer|

## Developer documentation

See [README.dev.md](README.dev.md) for documentation on setting up your development environment.

## Used by

This section aims to provide an overview of projects that use this repo's library (or something very similar), e.g. through git submodules or by including copies of this library in their source tree:

1. https://git.astron.nl/RD/dedisp/
1. https://git.astron.nl/RD/idg
1. https://git.astron.nl/RD/tensor-core-correlator

## Alternatives

This section provides an overview of similar tools in this space, and how they are different.

### cuda-api-wrappers

url: https://github.com/eyalroz/cuda-api-wrappers

- Aims to provide wrappers for the CUDA runtime API
- Development has slowed a bit recently
- Has 1 or 2 main developers
- Has gained quite a bit of attention (e.g. 440 stars; 57 forks)

The project is planning to support more of the Driver API (for fine-grained control of CUDA devices) and NVRTC API (for runtime compilation of kernels); there is a release candidate ([`v0.5.0-rc1`](https://github.com/eyalroz/cuda-api-wrappers/tree/v0.5.0-rc1)). It doesn't provide support for cuFFT and cuBLAS though.

### cuda-wrapper

url: https://github.com/halmd-org/cuda-wrapper

- Aims to provide a C++ wrapper for the CUDA Driver and Runtime APIs

### CudaPlusPlus

url: https://github.com/apardyl/cudaplusplus

- Aims to provide a C++ wrapper for the CUDA Driver API
- Project appears inactive

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for a guide on how to contribute.
