# Developer documentation

If you're looking for user documentation, go [here](README.md).

## Development install

### Get your own copy of the repository

Before you can do development work on the template, you'll need to check out a local copy of the repository:

```shell
cd <where you keep your GitHub repositories>
git clone https://github.com/nlesc-recruit/cudawrappers.git
cd cudawrappers
```

### Prepare your environment

:construction: See issues #7, #31

#### Build tools

Summary of what you need :

- `gcc` 9 or above
- `g++` 9 or above
- `make` 4 or above
- `cmake` 3.22 or above

Check that you have the correct `gcc`, `g++` and `make` versions using

```shell
gcc --version
g++ --version
make --version
```

On a Debian-like system you can install them with

```shell
sudo apt install build-essential
```

Next, you need CMake 3.17 or above. Check if you have the correct version installed with `cmake --version`.
If your CMake version is not adequate, you can install CMake manually by downloading the latest **stable** version from the [CMake downloads page](https://cmake.org/download/).

For instance, the commands below download install CMake 3.22.

```shell
cd $HOME # Or move to your preferrer location
wget https://github.com/Kitware/CMake/releases/download/v3.22.2/cmake-3.22.2.tar.gz
tar -zxf cmake-3.22.2.tar.gz
./bootstrap
make
sudo make install
```

If you can't run `sudo` to install `cmake` - for instance if you are in a cluster without privileges - you should add the path to the CMake `bin` folder to your path.
For instance, you can add the following to your `.bashrc`:

```shell
export PATH=$HOME/cmake-3.22.2/bin:$PATH
```

Remeber to update your environment either by logging out and in again, or running `source $HOME/.bashrc`.

#### CUDA and NVidia

You need a GPU with a [Nvidia Pascal](https://www.nvidia.com/en-in/geforce/products/10series/architecture/) architecture or newer to properly test this library.
Additionally, you need to install current Nvidia drivers. Ideally the latest drivers.
The earliest driver version we tested was 455.32.

You also need CUDA 10 or newer, which can be installed in a Debian-like system with the following command:

```shell
sudo apt install nvidia-cuda-toolkit
```

Check that `nvcc` is working with `nvcc --version`.

#### Linters and Formatters

We use the following linters and formatters in this project:

- [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
- [clang-tidy](https://clang.llvm.org/extra/clang-tidy)
- [cmake-format and cmake-lint](https://cmake-format.readthedocs.io/en/latest/installation.html)
- [cppcheck](https://cppcheck.sourceforge.io)
- [flawfinder](https://dwheeler.com/flawfinder/)

You can install these tools on a Debian-like system as follows:

```shell
# install cppcheck, clang-tidy, clang-format from your package manager
sudo apt install cppcheck clang-format clang-tidy

# install python3 and pip
sudo apt install python3 python3-pip

# install flawfinder, cmake-format, cmake-lint from PyPI using pip
python3 -m pip install flawfinder cmakelang
```

The formatter `clang-format` will format all source files, and `cmake-format` will format all CMake-related files.
The linters will check for errors and bugs, but also style-related issues. So run the formatters before running the linters.

Check how to run them in the [Building](#building) section below.

In addition, you can install VSCode extensions for many of these linters. Here is a short list:

- [VSCode extension for cmake-format](https://marketplace.visualstudio.com/items?itemName=cheshirekow.cmake-format)
- [VSCode extension clang-tidy](https://marketplace.visualstudio.com/items?itemName=notskm.clang-tidy)
- [VSCode extension Clang-Format](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format)

#### Linters on Codacy

We use [Codacy](codacy.com) for online linting information.
Codacy runs `cppcheck` and `flawfinder` online but to run `clang-tidy` we have to create a GitHub action, run `clang-tidy` there and push the information to Codacy.
Check the file [codacy.yml](.github/workflows/codacy.yml) for details on how that is done.

To run `clang-tidy` on Codacy for this project or a fork, you will need to define a `CODACY_PROJECT_TOKEN` secret.
For the `main` branch and pull requests originating from inside this repo, there is no need to create a new token.
But if it gets revoked, or for forks, follow the steps in the [Codacy API tokens page](https://docs.codacy.com/codacy-api/api-tokens/) for details on how to create one.

After a pull request is created, a Codacy test should appear. Follow the link there or [here](https://app.codacy.com/gh/nlesc-recruit/CUDA-wrappers) for the results.

#### pre-commit hooks

`pre-commit` is a tool that can automatically run linters, formatters, or any other executables whenever you commit code with `git commit`.

If you think having such automated checks is helpful for development, you can install the pre-commit CLI from PyPI using pip:

```shell
# Install the tool in user space
python3 -m pip install --user pre-commit
```

For other install options, look [here](https://pre-commit.com/#installation).

Enable the pre-commit hooks defined in `.pre-commit-config.yaml` with:

```shell
pre-commit install
```

Once enabled, future `git commit`s will trigger the pre-commit hooks. Depending on which files are changed by a given commit, some checks will be skipped. Here is an example after making some changes to a `CMakeLists.txt` file:

```shell
$ git commit -m "test precommit hook"
clang-format.........................................(no files to check)Skipped
clang-tidy...........................................(no files to check)Skipped
cppcheck.............................................(no files to check)Skipped
cmake-format.............................................................Failed
- hook id: cmake-format
- files were modified by this hook
cmake-lint...............................................................Passed
Validate repo CITATION.cff file......................(no files to check)Skipped
```

You can uninstall the pre-commit hooks by

```shell
pre-commit uninstall
```

Running pre-commit hooks individually is also possible with:

```shell
pre-commit run <name of the task>
```

For example,

```shell
pre-commit run clang-format
pre-commit run clang-tidy
pre-commit run cmake-format
pre-commit run cmake-lint
pre-commit run cppcheck
pre-commit run validate-cff
```

If you would like to use only some of the checks (for example only those that do not make code changes but only raise warnings), you  can do so by copying the project's default configuration to a file that has `'.user'` in the filename -- it will be gitignored.

```shell
cp .pre-commit-config.yaml .pre-commit-config.user.yaml
```

Then, after editing the new `'.user'` config file, uninstall the project default hooks if you have them:

```shell
pre-commit uninstall
```

and install your preferred hooks:

```shell
pre-commit install -c .pre-commit-config.user.yaml
```

When running a user configuration, you are still able to run the hooks from the project default configuration, like so:

```shell
# Run on staged files
pre-commit run cmake-format

# Run on a named file
pre-commit run cmake-format --file CMakeLists.txt
```

See [https://pre-commit.com/](https://pre-commit.com/) for more information.

## Building

:construction: See issue #33

The following commands will compile and create a library `libcudawrappers.so`.

```sh
cmake -S . -B build
make --directory=build
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

1. Make sure the `CHANGELOG.md` describes what was added, changed, or removed since the previous version. Limit the scope of the description to include only those things that affect semantic versioning (so things like changes to a github action do not need to be included in the CHANGELOG). See [semver.org](https://semver.org) for more details
1. Verify that the information in `CITATION.cff` is correct (authors, dates, etc.)
1. Make sure that any version strings anywhere in the software have been updated (e.g. CITATION.cff, CMakeLists.txt, etc.)

### GitHub

1. Make sure that the GitHub-Zenodo integration is enabled for https://github.com/nlesc-recruit/cudawrappers
1. Go to https://github.com/nlesc-recruit/cudawrappers/releases and click `Draft a new release`
