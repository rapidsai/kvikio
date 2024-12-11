#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
conda config --set path_conflict prevent

sccache --zero-stats

rapids-conda-retry mambabuild \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/kvikio

sccache --show-adv-stats

rapids-upload-conda-to-s3 python
