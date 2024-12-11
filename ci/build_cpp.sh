#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"
conda config --set path_conflict prevent

sccache --zero-stats

rapids-conda-retry mambabuild conda/recipes/libkvikio

sccache --show-adv-stats

rapids-upload-conda-to-s3 cpp
