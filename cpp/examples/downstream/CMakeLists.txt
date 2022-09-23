# =============================================================================
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

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

# Notice, even though KvikIO is a header-only library, we link to it here.
# Linking to `kvikio::kvikio` makes CMake include the headers of KvikIO when building.
target_link_libraries(downstream_example PRIVATE kvikio::kvikio)
