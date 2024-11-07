#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

echo '$PREFIX/include:'
ls $PREFIX/include
./build.sh -v -n libkvikio
