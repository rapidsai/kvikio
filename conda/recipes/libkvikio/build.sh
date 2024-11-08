#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

echo '$PREFIX:'
find $PREFIX
echo '$BUILD_PREFIX:'
find $BUILD_PREFIX
./build.sh -v -n libkvikio --cmake-args=\"--trace-expand\"
