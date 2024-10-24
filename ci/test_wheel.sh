#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
WHEELHOUSE="${PWD}/dist/"
RAPIDS_PY_WHEEL_NAME="kvikio_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python "${WHEELHOUSE}"

python -m pip install "$(echo ${WHEELHOUSE}/kvikio_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test]"

# If running CUDA 11.8 on arm64, we skip tests marked "cufile".
# cuFile didn't support arm until 12.4
PYTEST_MARK=$( \
  [["${CUDA_VERSION}" == "11.8.0" && "${NVARCH}" == "sbsa" ]] \
  && echo "-m 'not cufile'" || echo "" \
)

python -m pytest ${PYTEST_MARK} ./python/kvikio/tests
