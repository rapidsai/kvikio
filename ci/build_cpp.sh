#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels
# Setting channel priority per-repo until all RAPIDS can build using strict channel priority
# This will be replaced when we port this recipe to `rattler-build`
conda config --set channel_priority strict
# `rapids-configure-conda-channels` should only insert `rapidsai` channel into release builds
conda config --remove channels rapidsai

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
