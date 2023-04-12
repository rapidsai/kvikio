# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)

project(
  LegateKvikIO
  VERSION 23.04.00
  LANGUAGES C CXX
)

# This is for convenience only when doing editable builds to avoid setting the flag
if(NOT LegateKvikIO_ROOT)
  set(LegateKvikIO_ROOT ${CMAKE_SOURCE_DIR}/build)
endif()

set(BUILD_SHARED_LIBS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# We always want to build our bindings, so we trick the legate wrappers to never search for an
# installed version which is currently deduced by seeing if it is invoked via scikit-build. See
# <https://github.com/nv-legate/legate.core/pull/645>
find_package(legate_core REQUIRED)
set(SKBUILD OFF)
legate_add_cpp_subdirectory(cpp TARGET legate_kvikio EXPORT legate_kvikio-export)
legate_default_python_install(legate_kvikio EXPORT legate_kvikio-export)

# Generates `install_info.py`
legate_add_cffi(${CMAKE_SOURCE_DIR}/cpp/task_opcodes.hpp TARGET legate_kvikio)
