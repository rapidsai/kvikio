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

include(cmake/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

project(
  KvikIO
  VERSION 22.04.00
  LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Write the version header
rapids_cmake_write_version_file(include/kvikio/version_config.hpp)

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# build options
option(BUILD_EXAMPLES "Configure CMake to build examples" ON)

# find packages we depend on
rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET kvikio-exports
  INSTALL_EXPORT_SET kvikio-exports)

rapids_find_package(
  Threads REQUIRED
  BUILD_EXPORT_SET kvikio-exports
  INSTALL_EXPORT_SET kvikio-exports
)

rapids_cpm_init()

# find cuFile
rapids_find_package(
  cuFile REQUIRED
  BUILD_EXPORT_SET kvikio-exports
  INSTALL_EXPORT_SET kvikio-exports
)

# library targets
add_library(kvikio INTERFACE)
add_library(kvikio::kvikio ALIAS kvikio)

target_include_directories(
  kvikio INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                   "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(kvikio INTERFACE Threads::Threads)
target_link_libraries(kvikio INTERFACE CUDA::nvml)
target_link_libraries(kvikio INTERFACE cuda)
target_link_libraries(kvikio INTERFACE cufile::cuFile_interface)
target_link_libraries(kvikio INTERFACE dl)
target_compile_features(kvikio INTERFACE cxx_std_17)


# optionally build examples
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()


# optionally build tests
if(BUILD_TESTS AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(cmake/thirdparty/get_gtest.cmake)
  include(CTest) # calls enable_testing()

  add_subdirectory(tests)
endif()

include(CPack)

# install export targets
install(TARGETS kvikio EXPORT kvikio-exports)
install(DIRECTORY include/kvikio/ DESTINATION include/kvikio)
install(FILES ${KvikIO_BINARY_DIR}/include/kvikio/version_config.hpp DESTINATION include/kvikio)

set(doc_string
    [=[
Provide targets for KvikIO: C++ bindings for cuFile.
]=])

rapids_export(
  INSTALL kvikio
  EXPORT_SET kvikio-exports
  GLOBAL_TARGETS kvikio
  namespace kvikio::
  DOCUMENTATION doc_string
)

# build export targets
rapids_export(
  BUILD kvikio
  EXPORT_SET kvikio-exports
  GLOBAL_TARGETS kvikio
  namespace kvikio::
  DOCUMENTATION doc_string
)

# make documentation

# add_custom_command(
#   OUTPUT KvikIO_DOXYGEN
#   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
#   COMMAND doxygen Doxyfile
#   VERBATIM
#   COMMENT "Custom command for KvikIO doxygen docs")

# add_custom_target(
#   kvikio_doc
#   DEPENDS KvikIO_DOXYGEN
#   COMMENT "Target for the custom command to build the KvikIO doxygen docs")


