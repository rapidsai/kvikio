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

enable_testing()

include(rapids-test)
rapids_test_init()

#[=======================================================================[.rst:
kvikio_add_test
--------------------

Create a KvikIO test.

.. code-block:: cmake

   kvikio_add_test(NAME <name> SOURCES <sources>)

   ``NAME``
   Test name. Single-value argument.

   ``SOURCES``
   List of source files for the test. Multi-value argument.
#]=======================================================================]
function(kvikio_add_test)
  cmake_parse_arguments(
    _KVIKIO # prefix
    "" # optional
    "NAME" # single value
    "SOURCES" # multi-value
    ${ARGN}
  )

  if(DEFINED _KVIKIO_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown argument: ${_KVIKIO_UNPARSED_ARGUMENTS}")
  endif()

  add_executable(${_KVIKIO_NAME} ${_KVIKIO_SOURCES})
  set_target_properties(
    ${_KVIKIO_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${KvikIO_BINARY_DIR}/gtests>"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               # For std:: support of __int128_t. Can be removed once using cuda::std
               CXX_EXTENSIONS ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
  )
  target_link_libraries(
    ${_KVIKIO_NAME} PRIVATE kvikio::kvikio GTest::gmock GTest::gmock_main GTest::gtest
                            GTest::gtest_main CUDA::cudart
  )

  rapids_test_add(
    NAME ${_KVIKIO_NAME}
    COMMAND ${_KVIKIO_NAME}
    GPUS 1
    INSTALL_COMPONENT_SET testing
  )
endfunction()

kvikio_add_test(NAME BASIC_IO_TEST SOURCES test_basic_io.cpp)

kvikio_add_test(NAME DEFAULTS_TEST SOURCES test_defaults.cpp)

kvikio_add_test(NAME ERROR_TEST SOURCES test_error.cpp)

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/tests/libkvikio)
