#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/kvikio

# If running CUDA 11.8 on arm64, we skip tests marked "cufile" since
# cuFile didn't support arm until 12.4
[[ "${CUDA_VERSION}" == "11.8.0" && "${RUNNER_ARCH}" == "ARM64" ]] \
  && PYTEST_MARK=( -m 'not cufile' ) || PYTEST_MARK=()

pytest --cache-clear --verbose "${PYTEST_MARK[@]}" "$@" tests
