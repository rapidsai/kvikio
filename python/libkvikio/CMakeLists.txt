# =============================================================================
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cpp/cmake/rapids_config.cmake)
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
if(USE_NVCOMP_RUNTIME_WHEEL)
  set(KvikIO_EXPORT_NVCOMP OFF)
endif()
set(CUDA_STATIC_RUNTIME ON)

add_subdirectory(../../cpp kvikio-cpp)

if(USE_NVCOMP_RUNTIME_WHEEL)
  set(rpaths "$ORIGIN/../../nvidia/nvcomp")
  foreach(tgt IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
    set_property(
      TARGET ${tgt}
      PROPERTY INSTALL_RPATH ${rpaths}
      APPEND
    )
  endforeach()
endif()
