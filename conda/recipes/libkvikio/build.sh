#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

echo '$PREFIX:'
find $PREFIX
./build.sh -v -n libkvikio
