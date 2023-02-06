#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
##############################################
# kvikIO GPU build and test script for CI    #
##############################################
set -e
NUMARGS=$#
ARGS=$*

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Set home to the job's workspace
export HOME="${WORKSPACE}"

# Switch to project root; also root of repo checkout
cd "${WORKSPACE}"

# Determine CUDA release version
export CUDA_REL=${CUDA_VERSION%.*}

# Parse git describe
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`
unset GIT_DESCRIBE_TAG

export RAPIDS_DATE_STRING=$(date +%y%m%d)

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment variables"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Check compiler versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

################################################################################
# TEST - C++
################################################################################

cd "$WORKSPACE/cpp/examples/downstream"

gpuci_logger "Build downstream C++ example"
mkdir build
cd build
cmake ..
cmake --build .
./downstream_example

################################################################################
# TEST - Run py.test for kvikio
################################################################################

gpuci_logger "Build and install libkvikio and kvikio"
cd "${WORKSPACE}"
export CONDA_BLD_DIR="${WORKSPACE}/.conda-bld"
gpuci_mamba_retry install -c conda-forge boa
gpuci_conda_retry mambabuild --croot ${CONDA_BLD_DIR} conda/recipes/libkvikio
gpuci_conda_retry mambabuild --croot ${CONDA_BLD_DIR} conda/recipes/kvikio --python=$PYTHON -c "${CONDA_BLD_DIR}"
gpuci_mamba_retry install -c "${CONDA_BLD_DIR}" libkvikio kvikio

gpuci_logger "Install test dependencies"
gpuci_mamba_retry install -c conda-forge -c rapidsai-nightly cudf

gpuci_logger "Python py.test for kvikio"
cd "${WORKSPACE}/python"
py.test --cache-clear --basetemp="${WORKSPACE}/cudf-cuda-tmp" --junitxml="${WORKSPACE}/junit-kvikio.xml" -v

cd "${WORKSPACE}"
gpuci_logger "Clean previous conda builds"
gpuci_mamba_retry uninstall libkvikio kvikio
rm -rf "${CONDA_BLD_DIR}"

gpuci_logger "Build and run libkvikio-debug"
export CMAKE_EXTRA_ARGS="-DCMAKE_BUILD_TYPE=Debug"
gpuci_conda_retry mambabuild --croot ${CONDA_BLD_DIR} --no-remove-work-dir --keep-old-work conda/recipes/libkvikio
gpuci_mamba_retry install -c "${CONDA_BLD_DIR}" libkvikio

# Check that `libcuda.so` is NOT being linked
LDD_BASIC_IO=$(ldd "${CONDA_BLD_DIR}/work/cpp/build/examples/basic_io")
if [[ "$LDD_BASIC_IO" == *"libcuda.so"* ]]; then
  echo "[ERROR] examples/basic_io shouldn't link to libcuda.so: ${LDD_BASIC_IO}"
  return 1
fi

# Run basic_io
"${CONDA_BLD_DIR}/work/cpp/build/examples/basic_io"

gpuci_logger "Clean previous conda builds"
gpuci_mamba_retry uninstall libkvikio
rm -rf "${CONDA_BLD_DIR}"

gpuci_logger "Build and run libkvikio-no-cufile"
export CMAKE_EXTRA_ARGS="-DCMAKE_DISABLE_FIND_PACKAGE_cuFile=TRUE"
gpuci_conda_retry mambabuild --croot ${CONDA_BLD_DIR} --no-remove-work-dir --keep-old-work conda/recipes/libkvikio
gpuci_mamba_retry install -c "${CONDA_BLD_DIR}" libkvikio

# Run basic_io
"${CONDA_BLD_DIR}/work/cpp/build/examples/basic_io"

if [ -n "${CODECOV_TOKEN}" ]; then
    codecov -t $CODECOV_TOKEN
fi

return ${EXITCODE}
