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

project(
  kvikio-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

rapids_cuda_init_architectures(kvikio-python)

find_package(KvikIO "${RAPIDS_VERSION}" REQUIRED)

find_package(CUDAToolkit REQUIRED)

if(NOT KvikIO_FOUND)
  add_subdirectory(../../cpp kvikio-cpp)
  set(cython_lib_dir kvikio)
  install(TARGETS kvikio DESTINATION ${cython_lib_dir})
endif()

include(rapids-cython-core)
rapids_cython_init()

# It would be better to factor nvcomp out into its own wheel. Until that is available, we vendor it
# here.
include(cmake/Modules/WheelHelpers.cmake)
install_aliased_imported_targets(
  TARGETS kvikio-python nvcomp::nvcomp nvcomp::nvcomp_gdeflate nvcomp::nvcomp_bitcomp DESTINATION
  ${cython_lib_dir}
)

add_subdirectory(cmake)
add_subdirectory(kvikio/_lib)
