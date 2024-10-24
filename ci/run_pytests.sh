#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/kvikio

# If running CUDA 11.8 on arm64, we skip tests marked "cufile".
# cuFile didn't support arm until 12.4
PYTEST_MARK=$( \
  [["${CUDA_VERSION}" == "11.8.0" && "${NVARCH}" == "sbsa" ]] \
  && echo "-m 'not cufile'" || echo "" \
)

python -m pytest ${PYTEST_MARK} --cache-clear --verbose "$@" tests
