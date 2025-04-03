#!/usr/bin/env bash

# Run this script with sudo in order to collect performance counters

export TMPDIR=/mnt/nvme
export KVIKIO_COMPAT_MODE=ON

my_bin=/home/coder/kvikio/cpp/build/latest/gtests/cpp_tests

# Warm up to fill the file cache
$my_bin

# Profile
nsys profile \
-o test \
-t nvtx,cuda,osrt \
-f true \
-b none \
--gpu-metrics-devices=0 \
--cpuctxsw=none \
--gpuctxsw=true \
--cuda-memory-usage=true \
$my_bin
