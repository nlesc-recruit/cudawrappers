[![github url](https://img.shields.io/badge/github-url-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/nlesc-recruit/CUDA-wrappers)
[![github license badge](https://img.shields.io/github/license/nlesc-recruit/CUDA-wrappers)](https://github.com/nlesc-recruit/CUDA-wrappers)
[![DOI](https://zenodo.org/badge/424944643.svg)](https://zenodo.org/badge/latestdoi/424944643)
[![cii badge](https://bestpractices.coreinfrastructure.org/projects/5686/badge)](https://bestpractices.coreinfrastructure.org/projects/5686)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B%20%20%E2%97%8F%20%20%E2%97%8F-orange)](https://fair-software.eu)

# CUDA-wrappers

C++ Wrappers for the CUDA Driver API and related tools

## Alternatives

This section provides an overview of similar tools in this space, and how they are different.

### cuda-api-wrappers

url: https://github.com/eyalroz/cuda-api-wrappers

- Aims to provide wrappers for the CUDA runtime API
- Development has slowed a bit recently
- Has 1 or 2 main developers
- Has gained quite a bit of attention (e.g. 440 stars; 57 forks)

The project is planning to support more of the Driver API (for fine-grained control of CUDA devices) and NVRTC API (for runtime compilation of kernels); there is a release candidate ([`v0.5.0-rc1`](https://github.com/eyalroz/cuda-api-wrappers/tree/v0.5.0-rc1)). It doesn't provide support for cuFFT, and cuBLAS though.

### cuda-wrapper

url: https://github.com/halmd-org/cuda-wrapper

- Aims to provide a C++ wrapper for the CUDA Driver and Runtime APIs

### CudaPlusPlus

url: https://github.com/apardyl/cudaplusplus

- Aims to provide a C++ wrapper for the CUDA Driver API
- Project appears inactive

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for a guide on how to contribute and [README.dev.md](README.dev.md) for documentation on setting up your development environment.

## License

This code is licensed under Apache 2.0. Copyright is with John Romein, Netherlands Institute for Radio Astronomy (ASTRON).
