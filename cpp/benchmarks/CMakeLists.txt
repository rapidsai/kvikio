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

add_executable(THREADPOOL_BENCHMARK "threadpool/threadpool_benchmark.cpp")
set_target_properties(THREADPOOL_BENCHMARK PROPERTIES INSTALL_RPATH "\$ORIGIN/../../lib")
target_include_directories(THREADPOOL_BENCHMARK PRIVATE ../include ${cuFile_INCLUDE_DIRS})
target_link_libraries(THREADPOOL_BENCHMARK PUBLIC benchmark::benchmark kvikio)

if(CMAKE_COMPILER_IS_GNUCXX)
  set(KVIKIO_CXX_FLAGS "-Wall;-Werror;-Wno-unknown-pragmas")
  target_compile_options(
    THREADPOOL_BENCHMARK PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${KVIKIO_CXX_FLAGS}>"
  )
endif()

install(
  TARGETS THREADPOOL_BENCHMARK
  COMPONENT testing
  DESTINATION bin/benchmarks/libkvikio
  EXCLUDE_FROM_ALL
)
