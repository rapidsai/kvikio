#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support customizing the ctests' install location
# First, try the installed location (CI/conda environments)
installed_test_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/tests/libkvikio/"
# Fall back to the build directory (devcontainer environments)
devcontainers_test_location="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../cpp/build/latest"

if [[ -d "${installed_test_location}" ]]; then
    cd "${installed_test_location}"
    # Conda packages install binaries directly in this directory
    examples_dir="."
elif [[ -d "${devcontainers_test_location}" ]]; then
    cd "${devcontainers_test_location}"
    # Devcontainer builds install binaries in an examples subdirectory
    examples_dir="examples"
else
    echo "Error: Test location not found. Searched:" >&2
    echo "  - ${installed_test_location}" >&2
    echo "  - ${devcontainers_test_location}" >&2
    exit 1
fi

# Run basic tests
${examples_dir}/BASIC_IO_EXAMPLE
${examples_dir}/BASIC_NO_CUDA_EXAMPLE

# Run gtests
ctest --no-tests=error --output-on-failure "$@"
