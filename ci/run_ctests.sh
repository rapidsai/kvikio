#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/tests/libkvikio/"

# Run basic tests
./BASIC_IO_TEST
./BASIC_NO_CUDA_TEST

# Run gtests
ctest --no-tests=error --output-on-failure "$@"
