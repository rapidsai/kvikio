# =============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

include(../../cpp/cmake/rapids_config.cmake)
include(rapids-cpm)
rapids_cpm_init()
include(rapids-cuda)
rapids_cuda_init_architectures(kvikio-python)

# TODO: Should we symlink FindcuFile.cmake into python/cmake? find cuFile
include(../../cpp/cmake/Modules/FindcuFile.cmake)

project(
  libkvikio-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX
)

set(KvikIO_BUILD_EXAMPLES OFF)
set(KvikIO_BUILD_TESTS OFF)
set(CUDA_STATIC_RUNTIME ON)

add_subdirectory(../../cpp kvikio-cpp)