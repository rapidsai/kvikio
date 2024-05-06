#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="kvikio"
package_dir="python/kvikio"

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
commit=$(git rev-parse HEAD)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

pyproject_file="${package_dir}/pyproject.toml"

sed -i "s/name = \"${package_name}\"/name = \"${package_name}${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}
echo "${version}" > VERSION
sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" "${package_dir}/${package_name}/_version.py"

alpha_spec=''
if ! rapids-is-release-build; then
    alpha_spec=',>=0.0.0a0'
fi
sed -r -i "s/libkvikio==(.*)\"/libkvikio${PACKAGE_CUDA_SUFFIX}==\1${alpha_spec}\"/g" ${pyproject_file}

CPP_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libkvikio_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libkvikio_dist)

cd "${package_dir}"

python -m pip install wheel
python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check --find-links ${CPP_WHEELHOUSE}

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
