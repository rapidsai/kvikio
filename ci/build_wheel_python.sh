#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_name="kvikio"
package_dir="python/kvikio"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Ensure 'kvikio' wheel builds always use the 'libkvikio' just built in the same CI run
#
# env variable 'PIP_CONSTRAINT' is set up by rapids-init-pip. It constrains all subsequent
# 'pip install', 'pip download', etc. calls (except those used in 'pip wheel', handled separately in build scripts)
LIBKVIKIO_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libkvikio kvikio --cuda "$RAPIDS_CUDA_VERSION")")
echo "libkvikio-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBKVIKIO_WHEELHOUSE}"/libkvikio_*.whl)" >> "${PIP_CONSTRAINT}"


# TODO: move this variable into `ci-wheel`
# Format Python limited API version string
RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"
export RAPIDS_PY_API

./ci/build_wheel.sh "${package_name}" "${package_dir}" --stable

python -m auditwheel repair \
    --exclude libkvikio.so \
    --exclude librapids_logger.so \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_python kvikio kvikio --stable --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
