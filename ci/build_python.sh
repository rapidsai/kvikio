#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-conda-retry mambabuild \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/kvikio

rapids-upload-conda-to-s3 python
