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

project(
  kvikio-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# TODO: Should we symlink FindcuFile.cmake into python/cmake? find cuFile
include(../../cpp/cmake/Modules/FindcuFile.cmake)

find_package(KvikIO REQUIRED "${RAPIDS_VERSION}")

find_package(CUDAToolkit REQUIRED)

include(rapids-cython-core)
rapids_cython_init()

add_subdirectory(cmake)

set(cython_lib_dir kvikio)

add_subdirectory(kvikio/_lib)
