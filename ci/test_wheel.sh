#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -eou pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download and install the libkvikio and kvikio wheels built in the previous step
LIBKVIKIO_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libkvikio_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
KVIKIO_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="kvikio_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

rapids-pip-retry install -v \
  "$(echo "${LIBKVIKIO_WHEELHOUSE}"/libkvikio_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
  "$(echo "${KVIKIO_WHEELHOUSE}"/kvikio_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

# If running CUDA 11.8 on arm64, we skip tests marked "cufile" since
# cuFile didn't support arm until 12.4
[[ "${CUDA_VERSION}" == "11.8.0" && "${RUNNER_ARCH}" == "ARM64" ]] \
  && PYTEST_MARK=( -m 'not cufile' ) || PYTEST_MARK=()

python -m pytest --cache-clear --verbose "${PYTEST_MARK[@]}" ./python/kvikio/tests
