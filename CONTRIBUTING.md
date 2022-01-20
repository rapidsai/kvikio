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
* Setuptools

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


#### Python

KvikIO uses [Black](https://black.readthedocs.io/en/stable/),
[isort](https://readthedocs.org/projects/isort/), and
[flake8](http://flake8.pycqa.org/en/latest/) to ensure a consistent code format
throughout the project.

These tools are used to auto-format the Python code, as well as check the Cython
code in the repository. Additionally, there is a CI check in place to enforce
that committed code follows our standards. You can use the tools to
automatically format your python code by running:

```bash
isort python
black python
```

and then check the syntax of your Python and Cython code by running:

```bash
flake8 python
flake8 --config=python/.flake8.cython
```

Additionally, many editors have plugins that will apply `isort` and `Black` as
you edit files, as well as use `flake8` to report any style / syntax issues.

#### C++

KvikIO uses [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html)

In order to format the C++ files, navigate to the root directory and run:
```
python3 scripts/run-clang-format.py -inplace
```

Additionally, many editors have plugins or extensions that you can set up to automatically run `clang-format` either manually or on file save.

## Attribution
 * Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
 * Portions adopted from https://github.com/dask/dask/blob/master/docs/source/develop.rst
