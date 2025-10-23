# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
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
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )
  target_link_libraries(
    ${_KVIKIO_NAME} PRIVATE kvikio::kvikio GTest::gmock GTest::gmock_main GTest::gtest
                            GTest::gtest_main CUDA::cudart $<TARGET_NAME_IF_EXISTS:CURL::libcurl>
  )

  rapids_test_add(
    NAME ${_KVIKIO_NAME}
    COMMAND ${_KVIKIO_NAME}
    GPUS 1
    INSTALL_COMPONENT_SET testing
  )
endfunction()

kvikio_add_test(NAME BASIC_IO_TEST SOURCES test_basic_io.cpp)

kvikio_add_test(NAME DEFAULTS_TEST SOURCES test_defaults.cpp utils/env.cpp)

kvikio_add_test(NAME ERROR_TEST SOURCES test_error.cpp)

kvikio_add_test(NAME MMAP_TEST SOURCES test_mmap.cpp)

if(KvikIO_REMOTE_SUPPORT)
  kvikio_add_test(NAME REMOTE_HANDLE_TEST SOURCES test_remote_handle.cpp utils/env.cpp)
  kvikio_add_test(NAME HDFS_TEST SOURCES test_hdfs.cpp utils/hdfs_helper.cpp)
  kvikio_add_test(NAME TLS_TEST SOURCES test_tls.cpp utils/env.cpp)
  kvikio_add_test(NAME URL_TEST SOURCES test_url.cpp)
endif()

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/tests/libkvikio)
