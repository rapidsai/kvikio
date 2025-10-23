#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_name="kvikio"
package_dir="python/kvikio"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Ensure 'kvikio' wheel builds always use the 'libkvikio' just built in the same CI run
#
# Using env variable PIP_CONSTRAINT (initialized by 'rapids-init-pip') is necessary to ensure the constraints
# are used when creating the isolated build environment.
LIBKVIKIO_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libkvikio_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
echo "libkvikio-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBKVIKIO_WHEELHOUSE}"/libkvikio_*.whl)" >> "${PIP_CONSTRAINT}"

./ci/build_wheel.sh "${package_name}" "${package_dir}"

python -m auditwheel repair \
    --exclude libkvikio.so \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
