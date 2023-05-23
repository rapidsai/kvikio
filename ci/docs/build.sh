#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
###################################
# KvikIO Docs build script for CI #
###################################

if [ -z "$PROJECT_WORKSPACE" ]; then
    echo ">>>> ERROR: Could not detect PROJECT_WORKSPACE in environment"
    echo ">>>> WARNING: This script contains git commands meant for automated building, do not run locally"
    exit 1
fi

export DOCS_WORKSPACE=$WORKSPACE/docs
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export HOME=$WORKSPACE
export PROJECT_WORKSPACE=/rapids/kvikio
export LIBCUDF_KERNEL_CACHE_PATH="$HOME/.jitify-cache"
export PROJECTS=(kvikio libkvikio)

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Install KvikIO"
gpuci_conda_retry install -y -c rapidsai-nightly kvikio

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

# Build Doxygen docs
gpuci_logger "Build Doxygen docs"
cd $PROJECT_WORKSPACE/cpp/doxygen
doxygen Doxyfile

# Build Python docs
gpuci_logger "Build Python docs"
cd $PROJECT_WORKSPACE/docs
make html

#Commit to Website
cd $DOCS_WORKSPACE

for PROJECT in ${PROJECTS[@]}; do
    if [ ! -d "api/$PROJECT/$BRANCH_VERSION" ]; then
        mkdir -p api/$PROJECT/$BRANCH_VERSION
    fi
    rm -rf $DOCS_WORKSPACE/api/$PROJECT/$BRANCH_VERSION/*
done


mv $PROJECT_WORKSPACE/cpp/doxygen/html/* $DOCS_WORKSPACE/api/libkvikio/$BRANCH_VERSION
mv $PROJECT_WORKSPACE/docs/build/html/* $DOCS_WORKSPACE/api/kvikio/$BRANCH_VERSION
