# =============================================================================
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

include(cmake/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

project(
  KvikIO
  VERSION 23.06.00
  LANGUAGES CXX
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

# Write the version header
rapids_cmake_write_version_file(include/kvikio/version_config.hpp)

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# build options
option(KvikIO_BUILD_EXAMPLES "Configure CMake to build examples" ON)

# find packages we depend on
rapids_cpm_init()

rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET kvikio-exports
  INSTALL_EXPORT_SET kvikio-exports
)

rapids_find_package(
  Threads REQUIRED
  BUILD_EXPORT_SET kvikio-exports
  INSTALL_EXPORT_SET kvikio-exports
)

rapids_find_package(
  cuFile
  BUILD_EXPORT_SET kvikio-exports
  INSTALL_EXPORT_SET kvikio-exports
)
if(NOT cuFile_FOUND)
  message(WARNING "Building KvikIO without cuFile")
endif()

# library targets
add_library(kvikio INTERFACE)
add_library(kvikio::kvikio ALIAS kvikio)

target_include_directories(
  kvikio INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                   "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(kvikio INTERFACE Threads::Threads)
target_link_libraries(kvikio INTERFACE CUDA::toolkit)
if(cuFile_FOUND)
  target_link_libraries(kvikio INTERFACE cufile::cuFile_interface)
endif()
target_link_libraries(kvikio INTERFACE ${CMAKE_DL_LIBS})
target_compile_features(kvikio INTERFACE cxx_std_17)

# optionally build examples
if(KvikIO_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# optionally build tests
if(KvikIO_BUILD_TESTS AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(cmake/thirdparty/get_gtest.cmake)
  include(CTest) # calls enable_testing()
  add_subdirectory(tests)
endif()

include(CPack)

# install export targets
install(TARGETS kvikio EXPORT kvikio-exports)
install(DIRECTORY include/kvikio/ DESTINATION include/kvikio)
install(FILES ${KvikIO_BINARY_DIR}/include/kvikio/version_config.hpp DESTINATION include/kvikio)

include("${rapids-cmake-dir}/export/find_package_file.cmake")
rapids_export_find_package_file(
  BUILD "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/FindcuFile.cmake" kvikio-exports
)
rapids_export_find_package_file(
  INSTALL "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/FindcuFile.cmake" kvikio-exports
)

set(doc_string
    [=[
Provide targets for KvikIO: C++ bindings for cuFile.
]=]
)

rapids_export(
  INSTALL kvikio
  EXPORT_SET kvikio-exports
  GLOBAL_TARGETS kvikio
  NAMESPACE kvikio::
  DOCUMENTATION doc_string
)

# build export targets
rapids_export(
  BUILD kvikio
  EXPORT_SET kvikio-exports
  GLOBAL_TARGETS kvikio
  NAMESPACE kvikio::
  DOCUMENTATION doc_string
)

# make documentation

# add_custom_command( OUTPUT KvikIO_DOXYGEN WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
# COMMAND doxygen Doxyfile VERBATIM COMMENT "Custom command for KvikIO doxygen docs")

# add_custom_target( kvikio_doc DEPENDS KvikIO_DOXYGEN COMMENT "Target for the custom command to
# build the KvikIO doxygen docs")
