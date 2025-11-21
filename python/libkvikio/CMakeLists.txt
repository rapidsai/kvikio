# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cpm)
include(rapids-cuda)
rapids_cuda_init_architectures(libkvikio-python)
rapids_cpm_init()

project(
  libkvikio-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX
)

# Check if kvikio is already available. If so, it's the user's responsibility to ensure that the
# CMake package is also available at build time of the Python kvikio package.
find_package(kvikio "${RAPIDS_VERSION}")

if(kvikio_FOUND)
  return()
endif()

unset(kvikio_FOUND)

set(KvikIO_BUILD_BENCHMARKS OFF)
set(KvikIO_BUILD_EXAMPLES OFF)
set(KvikIO_BUILD_TESTS OFF)

set(CUDA_STATIC_RUNTIME ON)

add_subdirectory(../../cpp kvikio-cpp)
