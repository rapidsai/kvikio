#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

touch empty.cu
nvcc -c empty.cu -v
rm empty.cu

CUDACXX=nvcc ./build.sh -v -n libkvikio --cmake-args=\"--trace-expand\"
