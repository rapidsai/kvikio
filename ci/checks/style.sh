#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
#######################
# kvikio Style Tester #
#######################

PATH=/conda/bin:$PATH

# Activate common conda env
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

FORMAT_FILE_URL=https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-23.06/cmake-format-rapids-cmake.json
export RAPIDS_CMAKE_FORMAT_FILE=/tmp/rapids_cmake_ci/cmake-formats-rapids-cmake.json
mkdir -p $(dirname ${RAPIDS_CMAKE_FORMAT_FILE})
wget -O ${RAPIDS_CMAKE_FORMAT_FILE} ${FORMAT_FILE_URL}

# Run pre-commit checks
pre-commit run --hook-stage manual --all-files --show-diff-on-failure
