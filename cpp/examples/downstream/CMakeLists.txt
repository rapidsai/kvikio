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

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

include(../../cmake/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

project(
  KvikIODownstreamExample
  VERSION 22.04.00
  LANGUAGES CXX
)

rapids_cmake_build_type(Release)
rapids_cpm_init()

# TODO: use release tags when they become available
option(KvikIO_BUILD_EXAMPLES "" OFF)  # No need to build the KvikIO example
rapids_cpm_find(
  KvikIO 22.04
  GLOBAL_TARGETS kvikio::kvikio
  CPM_ARGS GIT_REPOSITORY https://github.com/rapidsai/kvikio.git
  SOURCE_SUBDIR cpp
  GIT_TAG branch-22.04
)

rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET downstream-example-exports
  INSTALL_EXPORT_SET downstream-example-exports
)

rapids_find_package(
  KvikIO REQUIRED
  BUILD_EXPORT_SET downstream-example-exports
  INSTALL_EXPORT_SET downstream-example-exports
)

add_executable(downstream_example downstream_example.cpp)
target_include_directories(downstream_example PRIVATE)
target_link_libraries(downstream_example PRIVATE kvikio CUDA::cudart)
