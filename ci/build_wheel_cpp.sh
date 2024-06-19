#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="libkvikio"
package_dir="python/libkvikio"

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

cd "${package_dir}"

python -m pip install wheel
# libkvikio is a header-only C++ library with no Python code, so
# it is entirely platform-agnostic. We cannot use auditwheel for
# retagging since it has no extension modules, so we use `wheel`
# directly instead.
python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check
python -m wheel tags --platform any dist/* --remove

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 cpp dist
