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

# ensure 'kvikio' wheel builds always use the 'libkvikio' just built in the same CI run
#
# using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment
echo "libkvikio-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${CPP_WHEELHOUSE}/libkvikio_*.whl)" > ./constraints.txt

sccache --zero-stats

PIP_CONSTRAINT="${PWD}/constraints.txt" \
SKBUILD_CMAKE_ARGS="-DUSE_NVCOMP_RUNTIME_WHEEL=ON" \
    python -m pip wheel . -w dist -v --no-deps --disable-pip-version-check

sccache --show-adv-stats

mkdir -p final_dist
python -m auditwheel repair \
    --exclude libnvcomp.so.4 \
    -w final_dist \
    dist/*

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python final_dist
