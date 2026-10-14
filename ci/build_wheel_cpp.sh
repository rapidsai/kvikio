#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_name="libkvikio"
package_dir="python/libkvikio"

rapids-logger "Generating build requirements"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true" \
| tee /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
rapids-pip-retry install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt \
    pytest

# OpenSSL 3 requires Perl core modules beyond those used by older image builds.
if ! perl -MIPC::Cmd -e 1 >/dev/null 2>&1; then
    rapids-logger "Installing OpenSSL source build prerequisites"
    if command -v dnf >/dev/null 2>&1; then
        dnf install -y perl-core
    elif command -v apt-get >/dev/null 2>&1; then
        apt-get update
        apt-get install -y --no-install-recommends perl
    else
        echo "OpenSSL source builds require Perl core modules, including IPC::Cmd" >&2
        exit 1
    fi
fi

# Validate dependency selection and bundled TLS configuration without requiring a GPU.
python -m pytest ci/tests/test_conda_build_args.py ci/tests/test_curl_selection.py ci/tests/test_openssl_config.py

# build with '--no-build-isolation', for better sccache hit rate
# 0 really means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
export PIP_NO_BUILD_ISOLATION=0

./ci/build_wheel.sh "${package_name}" "${package_dir}"

python -m auditwheel repair \
    --exclude librapids_logger.so \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_cpp libkvikio kvikio --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
