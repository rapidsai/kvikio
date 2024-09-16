#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/tests/libkvikio/"

# Run basic tests
rapids-logger "Run BASIC_IO_EXAMPLE"
./BASIC_IO_EXAMPLE
rapids-logger "Run BASIC_NO_CUDA_EXAMPLE"
./BASIC_NO_CUDA_EXAMPLE

# Run gtests
rapids-logger "Run gtests"
./cpp_tests
# TODO: how to use ctest instead of executing the test directly?
# The following line fails with a "ctest doesn't exist" in CI.
# ctest --no-tests=error --output-on-failure "$@"
