#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="libkvikio"
package_dir="python/libkvikio"

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

pyproject_file="${package_dir}/pyproject.toml"

cd "${package_dir}"

python -m pip install wheel
python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check
python -m wheel tags --platform any dist/* --remove

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 cpp dist
