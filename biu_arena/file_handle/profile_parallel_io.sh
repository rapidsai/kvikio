#!/usr/bin/env bash

# Run this script with sudo in order to collect performance counters

# Temporary fix for aarch64
export LD_LIBRARY_PATH=/home/coder/kvikio/python/kvikio/build/pip/cuda-12.8/release/_deps/nvcomp_proprietary_binary-src/lib

my_python=/home/coder/.local/share/venvs/rapids/bin/python
my_program=parallel_io.py

# Warm up to fill the file cache
$my_python $my_program

# Profile
nsys profile \
-o parallel_io \
-t nvtx,cuda,osrt \
-f true \
-b none \
--gpu-metrics-devices=0 \
--cpuctxsw=none \
--gpuctxsw=true \
--cuda-memory-usage=true \
$my_python $my_program
