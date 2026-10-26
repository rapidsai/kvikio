#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download and install the libkvikio and kvikio wheels built in the previous step
LIBKVIKIO_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libkvikio kvikio --cuda "$RAPIDS_CUDA_VERSION")")
KVIKIO_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python kvikio kvikio --stable --cuda "$RAPIDS_CUDA_VERSION")")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

python -m venv libkvikio-env
. libkvikio-env/bin/activate

rapids-pip-retry install \
  -v \
  --prefer-binary \
  --constraint "${PIP_CONSTRAINT}" \
  "$(echo "${LIBKVIKIO_WHEELHOUSE}"/libkvikio_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)"
python -c "import libkvikio; libkvikio.load_library()"
deactivate

python -m venv kvikio-env
. kvikio-env/bin/activate

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
  -v \
  --prefer-binary \
  --constraint "${PIP_CONSTRAINT}" \
  "$(echo "${LIBKVIKIO_WHEELHOUSE}"/libkvikio_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
  "$(echo "${KVIKIO_WHEELHOUSE}"/kvikio_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

python -m pytest --cache-clear --verbose ./python/kvikio/tests
