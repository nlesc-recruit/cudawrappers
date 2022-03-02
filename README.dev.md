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

:construction: See issues #7, #31, #32

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
