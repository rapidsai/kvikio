# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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

set(kvikio_version 22.06.00)

project(
  kvikio-python
  VERSION ${kvikio_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C
            CXX
            CUDA)

option(FIND_KVIKIO_CPP "Search for existing KVIKIO C++ installations before defaulting to local files"
       OFF)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

# TODO: Should we symlink FindcuFile.cmake into python/cmake?
# find cuFile
include(../cpp/cmake/Modules/FindcuFile.cmake)

# Ignore unused variable warning.
set(ignored_variable "${SKBUILD}")

if(FIND_KVIKIO_CPP)
    find_package(KvikIO ${kvikio_version})
else()
    set(KVIKIO_FOUND OFF)
endif()

find_package(CUDAToolkit REQUIRED)

if(NOT KVIKIO_FOUND)
  add_subdirectory(../cpp kvikio-cpp)
endif()

add_subdirectory(cmake)
add_subdirectory(kvikio/_lib)

