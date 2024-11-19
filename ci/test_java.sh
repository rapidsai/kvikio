#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate java testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_java \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-print-env

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  libkvikio

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# CI/CD machines don't support running GDS, so the test will only make sure the library builds for now
rapids-logger "Run Java tests"
mkdir -p /mnt/nvme
rm -f /mnt/nvme/java_test
touch -f /mnt/nvme/java_test
pushd java
mvn clean install -DskipTests
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
