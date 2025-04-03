#!/usr/bin/env bash

# Run this script with sudo in order to collect performance counters

my_program=/home/coder/kvikio/cpp/build/latest/gtests/cpp_tests

nsys profile \
-o min \
-t nvtx,cuda,osrt \
-f true \
-b none \
--gpu-metrics-devices=0 \
--cpuctxsw=none \
--gpuctxsw=true \
--cuda-memory-usage=true \
$my_program
