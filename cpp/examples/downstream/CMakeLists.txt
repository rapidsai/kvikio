# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

project(
  KvikIODownstreamExample
  VERSION 1.00
  LANGUAGES CXX
)

# Get CPM see <https://github.com/cpm-cmake/CPM.cmake>
include(cmake/get_cpm.cmake)

# Get KvikIO see <https://github.com/rapidsai/kvikio>
include(cmake/get_kvikio.cmake)

add_executable(downstream_example downstream_example.cpp)

target_link_libraries(downstream_example PRIVATE kvikio::kvikio)
