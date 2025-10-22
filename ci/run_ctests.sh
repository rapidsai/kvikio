#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/tests/libkvikio/"

# Run basic tests
./BASIC_IO_EXAMPLE
./BASIC_NO_CUDA_EXAMPLE

# Run gtests
ctest --no-tests=error --output-on-failure "$@"
