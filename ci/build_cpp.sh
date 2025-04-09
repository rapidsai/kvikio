#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"
conda config --set path_conflict prevent

sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry build conda/recipes/libkvikio

sccache --show-adv-stats

rapids-upload-conda-to-s3 cpp
