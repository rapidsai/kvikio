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
  CUFILECXX
  VERSION 21.12.00
  LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Write the version header
rapids_cmake_write_version_file(include/cufile/version_config.hpp)

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# build options
option(BUILD_EXAMPLES "Configure CMake to build examples" ON)

# find packages we depend on
rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET cufilecxx-exports
  INSTALL_EXPORT_SET cufilecxx-exports)
rapids_cpm_init()

# find cuFile
rapids_find_package(
  cuFile REQUIRED
  BUILD_EXPORT_SET cufilecxx-exports
  INSTALL_EXPORT_SET cufilecxx-exports
)

# library targets
add_library(cufilecxx INTERFACE)
add_library(cufilecxx::cufilecxx ALIAS cufilecxx)

target_include_directories(cufilecxx INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                               "$<INSTALL_INTERFACE:include>")

target_link_libraries(cufilecxx INTERFACE CUDA::cudart)
target_link_libraries(cufilecxx INTERFACE CUDA::nvml)
target_link_libraries(cufilecxx INTERFACE cuda)
target_link_libraries(cufilecxx INTERFACE cuFile::cuFile_interface)
target_compile_features(cufilecxx INTERFACE cxx_std_17)


# if((BUILD_EXAMPLES OR BUILD_TESTS OR BUILD_BENCHMARKS) AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#   include(rapids-cuda)
#   rapids_cuda_init_architectures(CUFILECXX)
#   enable_language(CUDA)

#   # Since CuFile only enables CUDA optionally we need to manually include the file that
#   # rapids_cuda_init_architectures relies on `project` calling
#   include("${CMAKE_PROJECT_CUFILECXX_INCLUDE}")
#   message(STATUS "CuFile: Building with GPU Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
# endif()


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
install(TARGETS cufilecxx EXPORT cufilecxx-exports)
install(DIRECTORY include/cufile/ DESTINATION include/cufile)
install(FILES ${CUFILECXX_BINARY_DIR}/include/cufile/version_config.hpp DESTINATION include/cufile)

set(doc_string
    [=[
Provide targets for CuFile++: C++ bindings for cuFile.
]=])

rapids_export(
  INSTALL cufilecxx
  EXPORT_SET cufilecxx-exports
  GLOBAL_TARGETS cufilecxx
  NAMESPACE cufile::
  DOCUMENTATION doc_string
)

# build export targets
rapids_export(
  BUILD cufilecxx
  EXPORT_SET cufilecxx-exports
  GLOBAL_TARGETS cufilecxx
  NAMESPACE cufile::
  DOCUMENTATION doc_string
)

# make documentation

# add_custom_command(
#   OUTPUT CUFILECXXCXX_DOXYGEN
#   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
#   COMMAND doxygen Doxyfile
#   VERBATIM
#   COMMENT "Custom command for RMM doxygen docs")

# add_custom_target(
#   cufilecxx_doc
#   DEPENDS RMM_DOXYGEN
#   COMMENT "Target for the custom command to build the RMM doxygen docs")


