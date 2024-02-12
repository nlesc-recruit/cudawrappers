# Change Log

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Added
- Added `cu::Context::getDevice()`
- Added `cu::Module` constructor with `CUjit_option` map argument
- Added `DeviceMemory::size`
- Added `HostMemory::size`
- Added `Function::name`
- Added `cu::Stream::getContext()`
- Added overloaded versions of `cu::Stream::memcpyDtoHAsync` and `cu::Stream::memcpyDtoHAsync` that take CUdeviceptr as an argument
- Added `Function::setAttribute()`

### Changed
- Fixed the `cu::Module(CUmodule&)` constructor
- Added `Function::getAttribute` is now const
- The `cu::DeviceMemory` constructor now works with `size == 0`

### Fixed
- Fix compatibility with C++20 and C++23
- Fix `cu::HostMemory` constructor for registered memory
- Fix `cu::DeviceMemory` operator `T *()` for managed memory
- Fix `Stream::memAllocAsync` returns `DeviceMemory` with initialized size

### Removed

## [0.6.0] - 2023-10-06

### Added
### Changed
- Made the library header only
- Improved CMake configuration
- Moved asynchronous `::zero` from `Device` to `Stream`
- Replaced `include_cuda_code` helper with `target_embed_source`
- Changed some arguments from native to wrapped type
### Removed

## [0.5.0] - 2023-09-25
### Added
- cufft wrappers for 1D and 2D complex-to-complex FFTs
- cu::HostMemory constuctor for pre-allocated memory
- cu::DeviceMemory constructor for managed memory
- cu::Stream::cuMemPrefetchAsync for pre-fetching of managed memory
- cu::Stream::memAllocAsync and cu::Stream::memFreeAsync
- cu::Context::getFreeMemory and cu::Context::getTotalMemory

### Changed
- The vector_add example has now become a test
- Added `lib` prefix to shared libraries

### Removed
- `getDevice` function of `Context`, use `Device` constructor instead

## [0.4.0] - 2023-06-23
### Added

- CTest for testing
- nvtx library
- bump2version

### Changed

- Miscellaneous improvements to CMake and CI

### Removed

- `Source` class. Use `nvrtc::Program` instead.
- Commented out code that was not used (anymore)

## [0.3.0] - 2022-03-08

### Added

- API documentation

### Changed

- Fixed build issues of `vector_add` example in `tests`
- Improved, linter rules

### Removed

- moved usage examples to separate repositories

## [0.2.0] - 2022-03-02

### Added

- Several best practices were implemented, such as citation file, user and developer documentation, linters, formatters, pre-commit hooks, GitHub workflows, badges, and issue and pull request templates.

### Changed

- The name of the repository and the library are now `cudawrappers`.
- The folder structure has changed to better separate header and source files.

## [0.1.0] - 2022-02-14

### Added

- First release with existing code.

[Unreleased]: https://github.com/nlesc-recruit/cudawrappers/compare/0.3.0...HEAD
[0.3.0]: https://github.com/nlesc-recruit/cudawrappers/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/nlesc-recruit/cudawrappers/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/nlesc-recruit/cudawrappers/releases/tag/0.1.0
