#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="kvikio"
package_dir="python/kvikio"

source rapids-configure-sccache
source rapids-date-string


RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"


rapids-generate-version > ./VERSION

CPP_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libkvikio_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libkvikio_dist)

cd "${package_dir}"

python -m pip install wheel
python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check --find-links ${CPP_WHEELHOUSE}

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
