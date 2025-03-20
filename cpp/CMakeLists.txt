# =============================================================================
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

# ##################################################################################################
# * build options ----------------------------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build KvikIO shared library" ON)
option(KvikIO_BUILD_BENCHMARKS "Configure CMake to build benchmarks" ON)
option(KvikIO_BUILD_EXAMPLES "Configure CMake to build examples" ON)
option(KvikIO_BUILD_TESTS "Configure CMake to build tests" ON)
option(KvikIO_REMOTE_SUPPORT "Configure CMake to build with remote IO support" ON)
option(KvikIO_CUDA_SUPPORT "Configure CMake to build with CUDA support" ON)
option(KvikIO_EXPORT_NVCOMP "Export NVCOMP as a dependency" ON)

# ##################################################################################################
# * conda environment ------------------------------------------------------------------------------
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# ##################################################################################################
# * dependencies -----------------------------------------------------------------------------------

rapids_cpm_init()

rapids_find_package(
  Threads REQUIRED
  BUILD_EXPORT_SET kvikio-exports
  INSTALL_EXPORT_SET kvikio-exports
)

if(KvikIO_REMOTE_SUPPORT)
  include(cmake/thirdparty/get_libcurl.cmake)
  if(TARGET libcurl_static)
    set_target_properties(libcurl_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()
endif()

set(cuFile_FOUND 0)
if(KvikIO_CUDA_SUPPORT)
  rapids_find_package(
    CUDAToolkit REQUIRED
    BUILD_EXPORT_SET kvikio-exports
    INSTALL_EXPORT_SET kvikio-exports
  )
  include(cmake/thirdparty/get_nvtx.cmake)

  if(NOT TARGET CUDA::cuFile)
    message(
      WARNING "Cannot find cuFile - KvikIO will still work but won't use GPUDirect Storage (GDS)"
    )
  else()
    set(cuFile_FOUND 1)

    # Check API support
    try_compile(
      cuFile_BATCH_API_FOUND SOURCE_FROM_CONTENT
      batch.cpp
      [[#include <cufile.h>
      int main() {
        cuFileBatchIOSetUp(nullptr, 0);
        return 0;
      }
      ]]
      LINK_LIBRARIES CUDA::cuFile rt ${CMAKE_DL_LIBS}
      OUTPUT_VARIABLE batch_output
    )
    message(STATUS "Found cuFile Batch API: ${cuFile_BATCH_API_FOUND}")
    try_compile(
      cuFile_STREAM_API_FOUND SOURCE_FROM_CONTENT
      stream.cpp
      [[#include <cufile.h>
      int main() {
        CUfileHandle_t fh;
        CUstream stream;
        cuFileReadAsync(fh, nullptr, nullptr, nullptr, nullptr, nullptr, stream);
        return 0;
      }
      ]]
      LINK_LIBRARIES CUDA::cuFile rt ${CMAKE_DL_LIBS}
      OUTPUT_VARIABLE stream_output
    )
    message(STATUS "Found cuFile Stream API: ${cuFile_STREAM_API_FOUND}")
    try_compile(
      cuFile_VERSION_API_FOUND SOURCE_FROM_CONTENT
      version.cpp
      [[#include <cufile.h>
      int main() {
        int version;
        cuFileGetVersion(&version);
        return 0;
      }
      ]]
      LINK_LIBRARIES CUDA::cuFile rt ${CMAKE_DL_LIBS}
      OUTPUT_VARIABLE version_output
    )
    message(STATUS "Found cuFile Version API: ${cuFile_VERSION_API_FOUND}")
  endif()
endif()

include(cmake/thirdparty/get_thread_pool.cmake)

# ##################################################################################################
# * library targets --------------------------------------------------------------------------------

set(SOURCES
    "src/batch.cpp"
    "src/bounce_buffer.cpp"
    "src/buffer.cpp"
    "src/compat_mode.cpp"
    "src/http_status_codes.cpp"
    "src/cufile/config.cpp"
    "src/cufile/driver.cpp"
    "src/defaults.cpp"
    "src/error.cpp"
    "src/file_handle.cpp"
    "src/file_utils.cpp"
    "src/nvtx.cpp"
    "src/posix_io.cpp"
    "src/shim/cuda.cpp"
    "src/shim/cufile.cpp"
    "src/shim/utils.cpp"
    "src/stream.cpp"
    "src/utils.cpp"
)

if(KvikIO_REMOTE_SUPPORT)
  list(APPEND SOURCES "src/remote_handle.cpp" "src/shim/libcurl.cpp")
endif()

add_library(kvikio ${SOURCES})

# To avoid symbol conflicts when statically linking to libcurl.a (see get_libcurl.cmake) and its
# dependency OpenSSL, we exclude them when building libkvikio.so. This way, libkvikio.so will not
# expose any OpenSSL symbols that could conflict with downstream users like CPython that also links
# to (another version of) OpenSSL.
target_link_options(kvikio PRIVATE "LINKER:--exclude-libs,ALL")

add_library(kvikio::kvikio ALIAS kvikio)

target_include_directories(
  kvikio
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:$<$<BOOL:${KvikIO_CUDA_SUPPORT}>:${CUDAToolkit_INCLUDE_DIRS}>>"
  INTERFACE "$<INSTALL_INTERFACE:include>"
)

# Notice, we do not link to cuda or cufile since KvikIO opens them manually using `dlopen()`.
target_link_libraries(
  kvikio
  PUBLIC Threads::Threads BS::thread_pool ${CMAKE_DL_LIBS} $<TARGET_NAME_IF_EXISTS:nvtx3::nvtx3-cpp>
  PRIVATE $<TARGET_NAME_IF_EXISTS:CURL::libcurl>
)

target_compile_definitions(
  kvikio
  PUBLIC $<$<BOOL:${KvikIO_REMOTE_SUPPORT}>:KVIKIO_LIBCURL_FOUND>
         $<$<BOOL:${KvikIO_CUDA_SUPPORT}>:KVIKIO_CUDA_FOUND>
         $<$<BOOL:${cuFile_FOUND}>:KVIKIO_CUFILE_FOUND>
         $<$<BOOL:${cuFile_BATCH_API_FOUND}>:KVIKIO_CUFILE_BATCH_API_FOUND>
         $<$<BOOL:${cuFile_STREAM_API_FOUND}>:KVIKIO_CUFILE_STREAM_API_FOUND>
         $<$<BOOL:${cuFile_VERSION_API_FOUND}>:KVIKIO_CUFILE_VERSION_API_FOUND>
)

set_target_properties(
  kvikio
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

# ##################################################################################################
# * add benchmarks --------------------------------------------------------------------------------

if(KvikIO_BUILD_BENCHMARKS)
  # Find or install GoogleBench
  include(${rapids-cmake-dir}/cpm/gbench.cmake)
  rapids_cpm_gbench(BUILD_STATIC)

  add_subdirectory(benchmarks)
endif()

# ##################################################################################################
# * add examples -----------------------------------------------------------------------------------

# optionally build examples
if(KvikIO_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(CUDAToolkit_FOUND
   AND KvikIO_BUILD_TESTS
   AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME
)
  include(cmake/thirdparty/get_gtest.cmake)

  # include CTest module -- automatically calls enable_testing()
  include(CTest)

  # Always print verbose output when tests fail if run using `make test`.
  list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
  add_subdirectory(tests)
endif()

# ##################################################################################################
# * install targets --------------------------------------------------------------------------------

rapids_cmake_install_lib_dir(lib_dir)
include(CPack)
include(GNUInstallDirs)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME kvikio)

install(
  TARGETS kvikio
  DESTINATION ${lib_dir}
  EXPORT kvikio-exports
)

install(DIRECTORY include/kvikio/ DESTINATION include/kvikio)
install(FILES ${KvikIO_BINARY_DIR}/include/kvikio/version_config.hpp DESTINATION include/kvikio)

set(doc_string
    [=[
Provide targets for KvikIO.
]=]
)

set(final_code_string
    "
set(KvikIO_CUDA_SUPPORT [=[${KvikIO_CUDA_SUPPORT}]=])
set(KvikIO_CUFILE_SUPPORT [=[${cuFile_FOUND}]=])
set(KvikIO_REMOTE_SUPPORT [=[${KvikIO_REMOTE_SUPPORT}]=])
"
)
string(
  APPEND
  final_code_string
  [=[
if(KvikIO_CUDA_SUPPORT)
  find_package(CUDAToolkit REQUIRED QUIET)
  target_include_directories(kvikio::kvikio INTERFACE ${CUDAToolkit_INCLUDE_DIRS})

  if(KvikIO_CUFILE_SUPPORT AND NOT TARGET CUDA::cuFile)
    message(FATAL_ERROR "Compiled with cuFile support but cuFile not found")
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
