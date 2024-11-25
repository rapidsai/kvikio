#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="kvikio"
package_dir="python/kvikio"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Ensure 'kvikio' wheel builds always use the 'libkvikio' just built in the same CI run
#
# Using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment
RAPIDS_PY_WHEEL_NAME="libkvikio_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libkvikio_dist
echo "libkvikio-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/libkvikio_dist/libkvikio_*.whl)" > /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"

export SKBUILD_CMAKE_ARGS="-DUSE_NVCOMP_RUNTIME_WHEEL=ON"
./ci/build_wheel.sh "${package_name}" "${package_dir}"

python -m auditwheel repair \
    --exclude libkvikio.so \
    --exclude libnvcomp.so.4 \
    -w ${package_dir}/final_dist \
    ${package_dir}/dist/*

./ci/validate_wheel.sh ${package_dir} final_dist

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python ${package_dir}/final_dist
