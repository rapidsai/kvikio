# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.3 FATAL_ERROR)

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

# Find C++ libraries from RAPIDS wheels in site-packages.
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)

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
set(KvikIO_BUILD_NSYS_PLUGIN ON)
# Treat warnings as errors in wheel builds.
set(KvikIO_WARNINGS_AS_ERRORS ON)

# Wheels bundle curl and need its security fixes plus a recent OpenSSL on the manylinux image.
set(KvikIO_CURL_VERSION
    8.23.0
    CACHE STRING "Minimum curl version and CPM source version"
)
option(KvikIO_USE_STATIC_OPENSSL "Bundle pinned OpenSSL static libraries for wheels" ON)

add_subdirectory(../../cpp kvikio-cpp)
