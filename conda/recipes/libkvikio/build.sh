#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

echo '$PREFIX:'
find $PREFIX
echo "\$PREFIX: $PREFIX" > /dev/stderr
./build.sh -v -n libkvikio
