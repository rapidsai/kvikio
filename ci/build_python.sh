#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-configure-sccache
source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

rapids-logger "Prepending channel ${CPP_CHANNEL} to RATTLER_CHANNELS"

RATTLER_CHANNELS=("--channel" "${CPP_CHANNEL}" "${RATTLER_CHANNELS[@]}")

# Only use stable ABI package naming for Python >= 3.11
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
  RECIPE_PATH="conda/recipes/kvikio"
  # Also export name to use for GitHub artifact key
  RAPIDS_PACKAGE_NAME="$(rapids-package-name conda_python kvikio --stable --cuda)"
  export RAPIDS_PACKAGE_NAME
else
  RECIPE_PATH="conda/recipes/kvikio_py310"
fi

sccache --stop-server 2>/dev/null || true

rapids-logger "Building kvikio"

# --no-build-id allows for caching with `sccache`
# more info is available at
# https://rattler.build/latest/tips_and_tricks/#using-sccache-or-ccache-with-rattler-build
rattler-build build --recipe "${RECIPE_PATH}" \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

# remove build_cache directory to avoid uploading the entire source tree
# tracked in https://github.com/prefix-dev/rattler-build/issues/1424
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache

# Only use stable ABI package naming for Python >= 3.11
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
  RAPIDS_PACKAGE_NAME="$(rapids-package-name conda_python kvikio --stable --cuda)"
  export RAPIDS_PACKAGE_NAME
fi
