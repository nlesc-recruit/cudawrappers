# Developer documentation

If you're looking for user documentation, go [here](README.md).

## Development install

### Get your own copy of the repository

Before you can do development work on the template, you'll need to check out a local copy of the repository:

```shell
cd <where you keep your GitHub repositories>
git clone https://github.com/nlesc-recruit/CUDA-wrappers.git
cd CUDA-wrappers
```

### Prepare your environment

:construction: See issues #7, #31

#### Linters and Formatters

We use the following linters and formatters in this project:

- [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
- [clang-tidy](https://clang.llvm.org/extra/clang-tidy)
- [cmake-format and cmake-lint](https://cmake-format.readthedocs.io/en/latest/installation.html)
- [cppcheck](https://cppcheck.sourceforge.io)
- [flawfinder](https://dwheeler.com/flawfinder/)

The formatter `clang-format` will format all source files, and `cmake-format` will format all CMake-related files.
The linters will check for errors and bugs, but also style-related issues. So run the formatters before running the linters.

Check how to run them in the [Building](#building) section below.

In addition, you can install VSCode extensions for many of these linters. Here is a short list:

- [VSCode extension for cmake-format](https://marketplace.visualstudio.com/items?itemName=cheshirekow.cmake-format)
- [VSCode extension clang-tidy](https://marketplace.visualstudio.com/items?itemName=notskm.clang-tidy)
- [VSCode extension Clang-Format](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format)

## Building

:construction: See issue #33

Assume #33 will say something like

```sh
cmake -S . -B build
```

---

After building the project you can now run the formatters and linters.
You can do that by entering the `build` folder and running the `format` and `lint` make targets:

```sh
cd build
make format
make lint
```

For more details check the [Linters and Formatters](#linters-and-formatters) section.

You can run the individual tools by calling

```sh
make <tool>
```

where `<tool>` can be any of the following:

- `cmake-format`
- `cmake-lint`
- `clang-format`
- `clang-tidy`
- `cppcheck`
- `flawfinder`

## Running the tests

:construction:

## Building the documentation

:construction: See issue #29

## Making a release

:construction: See issue #30

### Preparation

1. Make sure the `CHANGELOG.md` has been updated
1. Verify that the information in `CITATION.cff` is correct
1. Make sure that the `version` in [CITATION.cff](CITATION.cff) have been bumped to the to-be-released version of the template

### GitHub

1. Make sure that the GitHub-Zenodo integration is enabled for https://github.com/nlesc-recruit/CUDA-wrappers
1. Go to https://github.com/nlesc-recruit/CUDA-wrappers/releases and click `Draft a new release`
