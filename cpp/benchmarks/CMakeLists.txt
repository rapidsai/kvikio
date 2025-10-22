# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

#[=======================================================================[.rst:
kvikio_add_benchmark
--------------------

Create a KvikIO benchmark.

.. code-block:: cmake

   kvikio_add_benchmark(NAME <name> SOURCES <sources>)

   ``NAME``
   Benchmark name. Single-value argument.

   ``SOURCES``
   List of source files for the benchmark. Multi-value argument.
#]=======================================================================]
function(kvikio_add_benchmark)
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
    PROPERTIES INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )

  target_link_libraries(${_KVIKIO_NAME} PUBLIC benchmark::benchmark kvikio::kvikio)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(KVIKIO_CXX_FLAGS "-Wall;-Werror;-Wno-unknown-pragmas")
    target_compile_options(${_KVIKIO_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${KVIKIO_CXX_FLAGS}>")
  endif()

  install(
    TARGETS ${_KVIKIO_NAME}
    COMPONENT testing
    DESTINATION bin/benchmarks/libkvikio
    EXCLUDE_FROM_ALL
  )
endfunction()

kvikio_add_benchmark(NAME THREADPOOL_BENCHMARK SOURCES "threadpool/threadpool_benchmark.cpp")
