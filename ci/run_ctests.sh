#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -xeuo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/tests/libkvikio/"

# Run basic tests
./BASIC_IO_EXAMPLE
./BASIC_NO_CUDA_EXAMPLE

# Running the tests directly, works fine
./cpp_tests

# But running them through ctest fails?!?
ctest -VV --no-tests=error --output-on-failure "$@"
