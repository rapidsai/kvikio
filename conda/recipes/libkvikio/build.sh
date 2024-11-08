#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

nvcc -c empty.cu -v

CUDACXX=nvcc ./build.sh -v -n libkvikio
