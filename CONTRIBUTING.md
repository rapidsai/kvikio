# Contributing to KvikIO

Contributions to KvikIO fall into the following three categories.

1. To report a bug, request a new feature, or report a problem with
    documentation, please file an [issue](https://github.com/rapidsai/kvikio/issues/new/choose)
    describing in detail the problem or new feature. The RAPIDS team evaluates
    and triages issues, and schedules them for a release. If you believe the
    issue needs priority attention, please comment on the issue to notify the
    team.
2. To propose and implement a new Feature, please file a new feature request
    [issue](https://github.com/rapidsai/kvikio/issues/new/choose). Describe the
    intended feature and discuss the design and implementation with the team and
    community. Once the team agrees that the plan looks good, go ahead and
    implement it, using the [code contributions](#code-contributions) guide below.
3. To implement a feature or bug-fix for an existing outstanding issue, please
    Follow the [code contributions](#code-contributions) guide below. If you
    need more context on a particular issue, please ask in a comment.

As contributors and maintainers to this project,
you are expected to abide by KvikIO's code of conduct.
More information can be found at: [Contributor Code of Conduct](https://docs.rapids.ai/resources/conduct/).

## Code contributions

### Requirements

To install users should have a working Linux machine with CUDA Toolkit
installed (v11.4+) and a working compiler toolchain (C++17 and cmake).

#### C++

The C++ bindings are header-only and depends on CUDA Driver and Runtime API.
In order to build and run the example code, CMake is required.

#### Python

The Python packages depends on the following packages:

* Cython
* Pip

For testing:
* pytest
* cupy

### Build KvikIO from source

#### C++
To build the C++ example, go to the `cpp` subdiretory and run:
```
mkdir build
cd build
cmake ..
make
```
Then run the example:
```
./examples/basic_io
```

#### Python

To build and install the extension, go to the `python` subdiretory and run:
```
python -m pip install .
```
One might have to define `CUDA_HOME` to the path to the CUDA installation.

In order to test the installation, run the following:
```
pytest tests/
```

And to test performance, run the following:
```
python benchmarks/single-node-io.py
```

### Code Formatting

#### Using pre-commit hooks

KvikIO uses [pre-commit](https://pre-commit.com/) to execute all code linters and formatters. These
tools ensure a consistent code format throughout the project. Using pre-commit ensures that linter
versions and options are aligned for all developers. Additionally, there is a CI check in place to
enforce that committed code follows our standards.

To use `pre-commit`, install via `conda` or `pip`:

```bash
conda install -c conda-forge pre-commit
```

```bash
pip install pre-commit
```

Then run pre-commit hooks before committing code:

```bash
pre-commit run
```

By default, pre-commit runs on staged files (only changes and additions that will be committed).
To run pre-commit checks on all files, execute:

```bash
pre-commit run --all-files
```

Optionally, you may set up the pre-commit hooks to run automatically when you make a git commit. This can be done by running:

```bash
pre-commit install
```

Now code linters and formatters will be run each time you commit changes.

You can skip these checks with `git commit --no-verify` or with the short version `git commit -n`.

#### Summary of pre-commit hooks

The following section describes some of the core pre-commit hooks used by the repository.
See `.pre-commit-config.yaml` for a full list.

C++/CUDA is formatted with [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html).

Python code runs several linters including [Black](https://black.readthedocs.io/en/stable/),
[isort](https://pycqa.github.io/isort/), and [flake8](https://flake8.pycqa.org/en/latest/).

[Codespell](https://github.com/codespell-project/codespell) is used to find spelling
mistakes, and this check is run as a pre-commit hook. To apply the suggested spelling fixes,
you can run  `codespell -i 3 -w .` from the repository root directory.
This will bring up an interactive prompt to select which spelling fixes to apply.

## Attribution
 * Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
 * Portions adopted from https://github.com/dask/dask/blob/master/docs/source/develop.rst
