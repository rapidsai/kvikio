# =============================================================================
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

include(cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

project(
  KvikIO
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

# Write the version header
rapids_cmake_write_version_file(include/kvikio/version_config.hpp)

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# build options
option(KvikIO_BUILD_EXAMPLES "Configure CMake to build examples" ON)

rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# find packages we depend on
rapids_cpm_init()

rapids_find_package(
  Threads REQUIRED
  BUILD_EXPORT_SET kvikio-exports
  INSTALL_EXPORT_SET kvikio-exports
)

rapids_find_package(
  CUDAToolkit
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
else()
  file(READ "${cuFile_INCLUDE_DIRS}/cufile.h" CUFILE_H_STR)
  string(FIND "${CUFILE_H_STR}" "cuFileBatchIOSetUp" cuFileBatchIOSetUp_location)
  if(cuFileBatchIOSetUp_location EQUAL "-1")
    set(cuFile_BATCH_API_FOUND FALSE)
  else()
    set(cuFile_BATCH_API_FOUND TRUE)
  endif()
  message(STATUS "Found cuFile Batch API: ${cuFile_BATCH_API_FOUND}")
  string(FIND "${CUFILE_H_STR}" "cuFileReadAsync" cuFileReadAsync_location)
  if(cuFileReadAsync_location EQUAL "-1")
    set(cuFile_STREAM_API_FOUND FALSE)
  else()
    set(cuFile_STREAM_API_FOUND TRUE)
  endif()
  message(STATUS "Found cuFile Stream API: ${cuFile_STREAM_API_FOUND}")
endif()

# library targets
add_library(kvikio INTERFACE)
add_library(kvikio::kvikio ALIAS kvikio)

# We enable CUDA and cuFile both here and in the FINAL_CODE_BLOCK export block. While the code block
# below (in FINAL_CODE_BLOCK) sets this information when KvikIO is imported from a
# kvikio-config.cmake file, this code block is intended to be used by projects that include KvikIO's
# source directory in their own CMake build.
#
# Normally we would just set the below without using $<BUILD_LOCAL_INTERFACE:...>, and without the
# final_code_string, but in this case we want to conditionally set these things at import time, not
# export time, since KvikIO is a header-only library that can adapt to different build environments.

# Enable CUDA in KvikIO
if(CUDAToolkit_FOUND)
  if(CUDA_STATIC_RUNTIME)
    target_link_libraries(kvikio INTERFACE $<BUILD_LOCAL_INTERFACE:CUDA::cudart_static>
      $<BUILD_LOCAL_INTERFACE:CUDA::cuFile_static>)
  else()
    target_link_libraries(kvikio INTERFACE $<BUILD_LOCAL_INTERFACE:CUDA::cudart>
      $<BUILD_LOCAL_INTERFACE:CUDA::cuFile>)
  endif()
  target_compile_definitions(kvikio INTERFACE $<BUILD_LOCAL_INTERFACE:KVIKIO_CUDA_FOUND>)
else()
  message(WARNING "Building KvikIO without CUDA")
endif()

# Enable supported cuFile features in KvikIO
if(cuFile_FOUND)
  target_link_libraries(kvikio INTERFACE $<BUILD_LOCAL_INTERFACE:cufile::cuFile_interface>)
  target_compile_definitions(kvikio INTERFACE $<BUILD_LOCAL_INTERFACE:KVIKIO_CUFILE_FOUND>)
  if(cuFile_BATCH_API_FOUND)
    target_compile_definitions(
      kvikio INTERFACE $<BUILD_LOCAL_INTERFACE:KVIKIO_CUFILE_BATCH_API_FOUND>
    )
  endif()
  if(cuFile_STREAM_API_FOUND)
    target_compile_definitions(
      kvikio INTERFACE $<BUILD_LOCAL_INTERFACE:KVIKIO_CUFILE_STREAM_API_FOUND>
    )
  endif()
endif()

target_include_directories(
  kvikio INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                   "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(kvikio INTERFACE Threads::Threads)
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
  BUILD "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/FindcuFile.cmake" EXPORT_SET kvikio-exports
)
rapids_export_find_package_file(
  INSTALL "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/FindcuFile.cmake" EXPORT_SET kvikio-exports
)

set(doc_string
    [=[
Provide targets for KvikIO: C++ bindings for cuFile.
]=]
)

set(final_code_string
    [=[
get_property(already_set_kvikio DIRECTORY PROPERTY kvikio_already_set_defines SET)
if(NOT already_set_kvikio)
  set_property(DIRECTORY PROPERTY kvikio_already_set_defines "ON")

  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    if(CUDA_STATIC_RUNTIME)
      target_link_libraries(kvikio::kvikio INTERFACE CUDA::cudart_static CUDA::cuFile_static)
    else()
      target_link_libraries(kvikio::kvikio INTERFACE CUDA::cudart CUDA::cuFile)
    endif()
    target_compile_definitions(kvikio::kvikio INTERFACE KVIKIO_CUDA_FOUND)
  else()
    message(WARNING "Building KvikIO without CUDA")
  endif()

  # Find cuFile and determine which features are supported
  find_package(cuFile QUIET)
  if(NOT cuFile_FOUND)
    message(WARNING "KvikIO: cuFile not found")
  else()
    file(READ "${cuFile_INCLUDE_DIRS}/cufile.h" CUFILE_H_STR)
    string(FIND "${CUFILE_H_STR}" "cuFileBatchIOSetUp" cuFileBatchIOSetUp_location)
    if(cuFileBatchIOSetUp_location EQUAL "-1")
      set(cuFile_BATCH_API_FOUND FALSE)
    else()
      set(cuFile_BATCH_API_FOUND TRUE)
    endif()
    message(STATUS "KvikIO: Found cuFile Batch API: ${cuFile_BATCH_API_FOUND}")
    string(FIND "${CUFILE_H_STR}" "cuFileReadAsync" cuFileReadAsync_location)
    if(cuFileReadAsync_location EQUAL "-1")
      set(cuFile_STREAM_API_FOUND FALSE)
    else()
      set(cuFile_STREAM_API_FOUND TRUE)
    endif()
    message(STATUS "KvikIO: Found cuFile Stream API: ${cuFile_STREAM_API_FOUND}")
  endif()

  # Enable supported cuFile features in KvikIO
  if(cuFile_FOUND)
    target_link_libraries(kvikio::kvikio INTERFACE cufile::cuFile_interface)
    target_compile_definitions(kvikio::kvikio INTERFACE KVIKIO_CUFILE_FOUND)
    if(cuFile_BATCH_API_FOUND)
      target_compile_definitions(kvikio::kvikio INTERFACE KVIKIO_CUFILE_BATCH_API_FOUND)
    endif()
    if(cuFile_STREAM_API_FOUND)
      target_compile_definitions(kvikio::kvikio INTERFACE KVIKIO_CUFILE_STREAM_API_FOUND)
    endif()
  endif()
endif()
]=]
)

rapids_export(
  INSTALL kvikio
  EXPORT_SET kvikio-exports
  GLOBAL_TARGETS kvikio
  NAMESPACE kvikio::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK final_code_string
)

rapids_export(
  BUILD kvikio
  EXPORT_SET kvikio-exports
  GLOBAL_TARGETS kvikio
  NAMESPACE kvikio::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK final_code_string
)
