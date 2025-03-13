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

set(TEST_INSTALL_PATH bin/tests/libkvikio)

# Example: basic_io

if(CUDAToolkit_FOUND)
  add_executable(BASIC_IO_EXAMPLE basic_io.cpp)
  set_target_properties(BASIC_IO_EXAMPLE PROPERTIES INSTALL_RPATH "\$ORIGIN/../../../lib")
  target_include_directories(BASIC_IO_EXAMPLE PRIVATE ../include ${cuFile_INCLUDE_DIRS})
  target_link_libraries(BASIC_IO_EXAMPLE PRIVATE kvikio CUDA::cudart)

  if(CMAKE_COMPILER_IS_GNUCXX)
    set(KVIKIO_CXX_FLAGS "-Wall;-Werror;-Wno-unknown-pragmas")
    target_compile_options(
      BASIC_IO_EXAMPLE PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${KVIKIO_CXX_FLAGS}>"
    )
  endif()

  install(
    TARGETS BASIC_IO_EXAMPLE
    COMPONENT testing
    DESTINATION ${TEST_INSTALL_PATH}
    EXCLUDE_FROM_ALL
  )
else()
  message(STATUS "Cannot build the basic_io example when CUDA is not found")
endif()

# Example: basic_no_cuda

add_executable(BASIC_NO_CUDA_EXAMPLE basic_no_cuda.cpp)
set_target_properties(BASIC_NO_CUDA_EXAMPLE PROPERTIES INSTALL_RPATH "\$ORIGIN/../../../lib")
target_include_directories(BASIC_NO_CUDA_EXAMPLE PRIVATE ../include)
target_link_libraries(BASIC_NO_CUDA_EXAMPLE PRIVATE kvikio)

if(CMAKE_COMPILER_IS_GNUCXX)
  set(KVIKIO_CXX_FLAGS "-Wall;-Werror;-Wno-unknown-pragmas")
  target_compile_options(
    BASIC_NO_CUDA_EXAMPLE PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${KVIKIO_CXX_FLAGS}>"
  )
endif()

install(
  TARGETS BASIC_NO_CUDA_EXAMPLE
  COMPONENT testing
  DESTINATION ${TEST_INSTALL_PATH}
  EXCLUDE_FROM_ALL
)
