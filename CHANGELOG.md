# Change Log

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Added
- cufft wrappers for 1D and 2D complex-to-complex FFTs
- cu::HostMemory constuctor for pre-allocated memory

### Changed
- The vector_add example has now become a test

### Removed
- `getDevice` function of `Context`, use `Device` constructor instead

## [0.4.0]
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
