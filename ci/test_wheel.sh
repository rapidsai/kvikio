#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download and install the libkvikio and kvikio wheels built in the previous step
LIBKVIKIO_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libkvikio_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)


if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
  KVIKIO_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" kvikio --stable --cuda "$RAPIDS_CUDA_VERSION")")
else
  KVIKIO_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="kvikio_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
fi

rapids-pip-retry install -v \
  "$(echo "${LIBKVIKIO_WHEELHOUSE}"/libkvikio_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
  "$(echo "${KVIKIO_WHEELHOUSE}"/kvikio_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

python -m pytest --cache-clear --verbose ./python/kvikio/tests
