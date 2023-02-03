#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"
SUITEERROR=0

rapids-print-env

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  libkvikio libkvikio-tests

rapids-logger "Check GPU usage"
nvidia-smi

set +e

# Run BASIC_IO_TEST
"$CONDA_PREFIX"/bin/tests/libkvikio/BASIC_IO_TEST

exitcode=$?
if (( ${exitcode} != 0 )); then
    SUITEERROR=${exitcode}
    echo "FAILED: BASIC_IO_TEST"
fi

exit ${SUITEERROR}