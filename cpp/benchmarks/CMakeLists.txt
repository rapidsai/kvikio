# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
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
  set_target_properties(${_KVIKIO_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/../../lib")
  target_include_directories(${_KVIKIO_NAME} PRIVATE ../include ${cuFile_INCLUDE_DIRS})
  target_link_libraries(${_KVIKIO_NAME} PUBLIC benchmark::benchmark kvikio)

  if(CMAKE_COMPILER_IS_GNUCXX)
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
