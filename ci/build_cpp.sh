#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version)
export RAPIDS_PACKAGE_VERSION

RAPIDS_ARTIFACTS_DIR=${RAPIDS_ARTIFACTS_DIR:-"${PWD}/artifacts"}
mkdir -p "${RAPIDS_ARTIFACTS_DIR}"
export RAPIDS_ARTIFACTS_DIR

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

# Construct the extra variants according to the architecture
if [[ "$(arch)" == "x86_64" ]]; then
    cat > variants.yaml << EOF
    c_compiler_version:
      - 14

    cxx_compiler_version:
      - 14

    cuda_version:
      - ${RAPIDS_CUDA_VERSION%.*}
EOF
else
    cat > variants.yaml << EOF
    zip_keys:
    - [c_compiler_version, cxx_compiler_version, cuda_version]

    c_compiler_version:
    - 12
    - 14

    cxx_compiler_version:
    - 12
    - 14

    cuda_version:
    - 12.1 # The last version to not support cufile
    - ${RAPIDS_CUDA_VERSION%.*}
EOF
fi

# --no-build-id allows for caching with `sccache`
# more info is available at
# https://rattler.build/latest/tips_and_tricks/#using-sccache-or-ccache-with-rattler-build
rattler-build build --recipe conda/recipes/libkvikio \
                    --variant-config variants.yaml \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats

# remove build_cache directory to avoid uploading the entire source tree
# tracked in https://github.com/prefix-dev/rattler-build/issues/1424
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache
