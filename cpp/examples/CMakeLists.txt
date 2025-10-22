# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(TEST_INSTALL_PATH bin/tests/libkvikio)

# Example: basic_io

if(CUDAToolkit_FOUND)
  add_executable(BASIC_IO_EXAMPLE basic_io.cpp)
  set_target_properties(BASIC_IO_EXAMPLE PROPERTIES INSTALL_RPATH "\$ORIGIN/../../../lib")
  target_link_libraries(BASIC_IO_EXAMPLE PRIVATE kvikio::kvikio CUDA::cudart)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
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
target_link_libraries(BASIC_NO_CUDA_EXAMPLE PRIVATE kvikio::kvikio)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
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
